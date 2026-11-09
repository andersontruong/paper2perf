"""Route selection for placed dataflow.

Routing answers *how* values move between placed endpoints. It owns hardware
path selection and annotates each task-graph edge with its chosen route: the
link-id sequence, per-hop transfer bytes, and transfer seconds.
Route selection never mutates placement.
"""

from __future__ import annotations

import networkx as nx

from hw_model.model import NAME_KEY, HardwareGraph


def task_key(node_id: str) -> str:
    """Collapse source/sink endpoint ids into a single per-value task id."""
    return (
        f"boundary:{node_id.removeprefix('source:').removeprefix('sink:')}"
        if node_id.startswith(("source:", "sink:"))
        else node_id
    )


def movement_resource(graph: nx.MultiDiGraph, task: str, value_id: str) -> str:
    """True resource a movement leaves or enters for ``value_id``.

    A wrapper is an aggregate scope with no single compute resource: the value
    crosses its boundary at the memory the child reads or writes it, recorded in
    ``scope_boundaries``. Every other task uses its placed resource.
    """
    data = graph.nodes[task]
    boundaries = data.get("scope_boundaries")
    if boundaries and value_id in boundaries:
        return boundaries[value_id]
    parent_boundaries = data.get("parent_scope_boundaries")
    if parent_boundaries and value_id in parent_boundaries:
        return parent_boundaries[value_id]
    return data["resource"]


def select_routes(
    graph: nx.MultiDiGraph,
    hardware: HardwareGraph,
) -> None:
    """Annotate every task-graph edge with its selected hardware route.

    Each edge must connect placed endpoints (``graph.nodes[...]["resource"]``).
    Adds ``route`` (link ids), ``transfer_bytes``, and ``seconds`` to the edge
    attributes.
    """
    for source_task, destination_task, key, data in graph.edges(keys=True, data=True):
        size_bytes = data["size_bytes"]
        value_id = data["value_id"]
        source_resource = movement_resource(graph, source_task, value_id)
        destination_resource = movement_resource(graph, destination_task, value_id)
        # An aliased sink reuses the destination buffer. If the producer already
        # sits in that memory the hand-off is free; otherwise the producer must
        # still stage its result there.
        if graph.nodes[destination_task].get("alias"):
            if source_resource == destination_resource:
                data["route"] = ()
                data["transfer_bytes"] = ()
                data["seconds"] = 0.0
                continue
        path = hardware.shortest_path(source_resource, destination_resource, size_bytes)
        if path is None:
            raise ValueError(
                f"no route for edge {key!r}: {source_task!r} -> {destination_task!r}"
            )
        data["route"] = tuple(link[2][NAME_KEY] for link in path)
        data["transfer_bytes"] = tuple(size_bytes for _ in path)
        data["seconds"] = sum(hardware.link_seconds(link, size_bytes) for link in path)
