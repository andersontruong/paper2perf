"""Optional collection of mapper artifacts for visualization.

A :class:`MappingTrace` is passed through the mapping pipeline as the ``trace``
keyword. Each scope reached by ``map_fx_graph`` records its placed task graph,
routed task graph, physical event graph, hierarchical schedule, and modulo
placements. Passing ``None`` (the default) disables collection with no change in
behavior.

A value's residency is carried by the storage nodes in the physical event graph
(``kind="storage"``), which is the single storage representation; this trace does
not keep a parallel copy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import networkx as nx

from dataflow_mapper.scheduling.hierarchical import FXHierarchicalEventSchedule
from dataflow_mapper.scheduling.model import ScopeKey, TilingScope
from dataflow_mapper.scheduling.modulo import EventModuloPlacement

# Scope key: the path-only scope identity. It alternates wrapper node names with
# the tile coordinate of the branch that enters each wrapper.


@dataclass
class ScopeTrace:
    """Artifacts recorded for one ``map_fx_graph`` scope."""

    scope_path: ScopeKey
    module_axis: str | None
    placed_graph: nx.MultiDiGraph | None = None
    routed_graph: nx.MultiDiGraph | None = None
    event_graph: nx.DiGraph | None = None
    schedule: FXHierarchicalEventSchedule | None = None
    modulo_placements: Mapping[tuple[TilingScope, ...], EventModuloPlacement] = field(
        default_factory=dict
    )


class MappingTrace:
    """Collects per-scope mapping artifacts when passed as ``trace=``.

    A scope is keyed by its path-only ``scope_path``. The first record for a key
    creates its :class:`ScopeTrace`; later records overwrite the same scope, so a
    capacity-retiling pass keeps its final successful attempt.
    """

    def __init__(self) -> None:
        self._scopes: list[ScopeTrace] = []
        self._by_key: dict[ScopeKey, ScopeTrace] = {}

    @property
    def scopes(self) -> tuple[ScopeTrace, ...]:
        """Recorded scope traces, in first-visited order."""
        return tuple(self._scopes)

    def _scope(self, scope_path: ScopeKey, module_axis: str | None) -> ScopeTrace:
        """Return the trace for one scope, creating it on first use."""
        key = scope_path
        scope = self._by_key.get(key)
        if scope is None:
            scope = ScopeTrace(scope_path, module_axis)
            self._by_key[key] = scope
            self._scopes.append(scope)
        return scope

    def record_placed(
        self,
        scope_path: ScopeKey,
        module_axis: str | None,
        graph: nx.MultiDiGraph,
    ) -> None:
        """Record the placed task graph, before routes are selected."""
        self._scope(scope_path, module_axis).placed_graph = graph

    def record_routed(
        self,
        scope_path: ScopeKey,
        module_axis: str | None,
        graph: nx.MultiDiGraph,
    ) -> None:
        """Record the routed task graph, after route selection."""
        self._scope(scope_path, module_axis).routed_graph = graph

    def record_scheduled(
        self,
        scope_path: ScopeKey,
        module_axis: str | None,
        event_graph: nx.DiGraph,
        schedule: FXHierarchicalEventSchedule,
        modulo_placements: (
            Mapping[tuple[TilingScope, ...], EventModuloPlacement] | None
        ) = None,
    ) -> None:
        """Record the physical event graph, schedule, and hidden scheduling data.

        Args:
            scope_path: Ancestor tile-coordinate path for this scope.
            module_axis: The scope module's tiling axis, or ``None`` for root.
            event_graph: Physical event graph with absolute event intervals and
                the storage nodes that carry value residency.
            schedule: Hierarchical schedule (makespan, per-scope durations).
            modulo_placements: Modulo placements per innermost scope path.
        """
        scope = self._scope(scope_path, module_axis)
        scope.event_graph = event_graph
        scope.schedule = schedule
        scope.modulo_placements = dict(modulo_placements or {})


__all__ = ["MappingTrace", "ScopeKey", "ScopeTrace"]
