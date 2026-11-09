"""NetworkX-backed hardware topology for the dataflow modeling flow.

Hardware is an attribute ``networkx.MultiGraph`` wrapped by
:class:`HardwareGraph`. There are no resource or link classes: nodes and edges
are plain attribute mappings, and the model adds only canonical link names.

Vocabulary
----------
node ``name``               resource name
node ``type``               ``"mem"`` or ``"compute"``
node ``capacity``           memory capacity in bytes
node ``compute_peak``       compute throughput in FLOP/s
node ``flop_capability``    mapping of requirement unit -> FLOPs consumed per unit
edge ``name``               link id, ``source::target`` or ``source::target::key``
edge ``bw``                 bandwidth in bytes/s, or ``None`` for an ideal link
edge ``latency``            fixed per-transfer overhead in seconds, or ``0.0``
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import networkx as nx

MEMORY_KIND = "mem"
COMPUTE_KIND = "compute"

NAME_KEY = "name"
KIND_KEY = "type"
CAPACITY_KEY = "capacity"
THROUGHPUT_KEY = "compute_peak"
CAPABILITY_KEY = "flop_capability"
BANDWIDTH_KEY = "bw"
LATENCY_KEY = "latency"

# One physical link as (source, target, data); the canonical id is data[NAME_KEY].
HardwareEdge = tuple[str, str, Mapping[str, Any]]


def link_name(graph: nx.MultiGraph, source: str, target: str, key: object) -> str:
    """Canonical link id; the key disambiguates parallel edges."""
    if graph.number_of_edges(source, target) > 1:
        return f"{source}::{target}::{key}"
    return f"{source}::{target}"


@dataclass(frozen=True)
class HardwareGraph:
    """Read-only handle around a NetworkX hardware topology."""

    graph: nx.MultiGraph

    @classmethod
    def from_networkx(cls, graph: nx.Graph) -> HardwareGraph:
        """Wrap an attribute multigraph, assigning canonical link names in place."""
        if not graph.is_multigraph():
            raise ValueError("hardware topology must be a MultiGraph")
        for source, target, key, data in graph.edges(keys=True, data=True):
            if not data.get(NAME_KEY):
                data[NAME_KEY] = link_name(graph, source, target, key)
        return cls(graph)

    def validate(self) -> None:
        """Assert the topology is a named multigraph with the documented vocabulary."""
        graph = self.graph
        if not graph.is_multigraph():
            raise ValueError("hardware topology must be a MultiGraph")
        for node_id, data in graph.nodes(data=True):
            kind = data.get(KIND_KEY)
            if kind == MEMORY_KIND:
                capacity = data.get(CAPACITY_KEY)
                if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
                    raise ValueError(
                        f"memory node {node_id!r} requires a positive integer capacity"
                    )
            elif kind == COMPUTE_KIND:
                throughput = data.get(THROUGHPUT_KEY)
                if (
                    isinstance(throughput, bool)
                    or not isinstance(throughput, (int, float))
                    or throughput <= 0
                ):
                    raise ValueError(
                        f"compute node {node_id!r} requires a positive compute_peak"
                    )
                capability = data.get(CAPABILITY_KEY)
                if not isinstance(capability, Mapping) or not capability:
                    raise ValueError(
                        f"compute node {node_id!r} requires a flop_capability mapping"
                    )
            else:
                raise ValueError(f"hardware node {node_id!r} has invalid type {kind!r}")
        for source, target, _, data in graph.edges(keys=True, data=True):
            name = data.get(NAME_KEY)
            if not name:
                raise ValueError(f"hardware link {source!r} -- {target!r} requires a name")
            bandwidth = data.get(BANDWIDTH_KEY)
            if bandwidth is not None and (
                isinstance(bandwidth, bool)
                or not isinstance(bandwidth, (int, float))
                or bandwidth <= 0
            ):
                raise ValueError(f"hardware link {name!r} has invalid bandwidth")
            latency = data.get(LATENCY_KEY)
            if latency is not None and (
                isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency < 0
            ):
                raise ValueError(f"hardware link {name!r} has invalid latency")

    def nodes_by_id(self) -> dict[str, Mapping[str, Any]]:
        """Node attribute mappings keyed by resource id."""
        return {node_id: data for node_id, data in self.graph.nodes(data=True)}

    def is_memory(self, node_id: str) -> bool:
        """Whether the resource is a memory node."""
        return self.graph.nodes[node_id].get(KIND_KEY) == MEMORY_KIND

    def is_compute(self, node_id: str) -> bool:
        """Whether the resource is a compute node."""
        return self.graph.nodes[node_id].get(KIND_KEY) == COMPUTE_KIND

    def memory_capacity(self, node_id: str) -> int | None:
        """Memory capacity in bytes, or ``None`` when unbounded/absent."""
        return self.graph.nodes[node_id].get(CAPACITY_KEY)

    def compute_peak(self, node_id: str) -> float:
        """Compute throughput in FLOP/s."""
        return self.graph.nodes[node_id].get(THROUGHPUT_KEY)

    def flop_capability(self, node_id: str) -> Mapping[str, float]:
        """Per-unit FLOP costs for a compute node; empty when absent."""
        return self.graph.nodes[node_id].get(CAPABILITY_KEY, {})

    def links_by_id(self) -> dict[str, HardwareEdge]:
        """Physical links keyed by their canonical name."""
        return {
            data[NAME_KEY]: (source, target, data)
            for source, target, _key, data in self.graph.edges(keys=True, data=True)
        }

    def links_by_endpoint(self) -> dict[str, list[HardwareEdge]]:
        """Physical links indexed by endpoint, each bucket sorted by link name."""
        result: dict[str, list[HardwareEdge]] = {}
        for source, target, _key, data in self.graph.edges(keys=True, data=True):
            edge = (source, target, data)
            result.setdefault(source, []).append(edge)
            if target != source:  # don't double-count a self-loop
                result.setdefault(target, []).append(edge)
        for bucket in result.values():
            bucket.sort(key=lambda edge: edge[2][NAME_KEY])  # deterministic tie-breaks
        return result

    def link_other_endpoint(self, link: HardwareEdge, node_id: str) -> str:
        """The endpoint of ``link`` opposite ``node_id``."""
        source, target, _data = link
        return target if node_id == source else source

    def link_seconds(self, link: HardwareEdge, size_bytes: int) -> float:
        """Fixed link overhead plus transfer time; ``bw is None`` is an ideal link."""
        _source, _target, data = link
        latency = float(data.get(LATENCY_KEY) or 0.0)
        bandwidth = data.get(BANDWIDTH_KEY)
        return latency if bandwidth is None else latency + size_bytes / bandwidth

    def shortest_path(
        self,
        source: str,
        destination: str,
        size_bytes: int,
    ) -> tuple[HardwareEdge, ...] | None:
        """Lowest-transfer-time link path between two resources.

        Intermediate hops must be memory nodes; only the destination may be a compute
        node. Ties break on the link-name sequence. Returns ``()`` when source and
        destination are the same and ``None`` when no path exists.
        """
        if source == destination:
            return ()
        links_by_endpoint = self.links_by_endpoint()
        queue: list[tuple[float, tuple[str, ...], str, tuple[HardwareEdge, ...]]] = [
            (0.0, (), source, ())
        ]
        best: dict[str, tuple[float, tuple[str, ...]]] = {source: (0.0, ())}
        while queue:
            cost, ids, node, path = heapq.heappop(queue)
            if best.get(node) != (cost, ids):
                continue
            if node == destination:
                return path
            for link in links_by_endpoint.get(node, ()):
                neighbor = self.link_other_endpoint(link, node)
                if neighbor != destination and not self.is_memory(neighbor):
                    continue
                candidate = (
                    cost + self.link_seconds(link, size_bytes),
                    (*ids, link[2][NAME_KEY]),
                )
                if best.get(neighbor) is None or candidate < best[neighbor]:
                    best[neighbor] = candidate
                    heapq.heappush(queue, (*candidate, neighbor, (*path, link)))
        return None


__all__ = [
    "BANDWIDTH_KEY",
    "CAPABILITY_KEY",
    "CAPACITY_KEY",
    "COMPUTE_KIND",
    "KIND_KEY",
    "LATENCY_KEY",
    "MEMORY_KIND",
    "NAME_KEY",
    "THROUGHPUT_KEY",
    "HardwareEdge",
    "HardwareGraph",
    "link_name",
]
