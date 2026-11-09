"""Recursive scheduling for FX event scope trees.

Builds a validated scope tree for the mapper's tiled event graph and composes
its scopes into a makespan: parent setup, modulo-placed leaf bodies, and serial
composition around nested child scopes. All times are seconds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import networkx as nx

from dataflow_mapper.scheduling.model import (
    EventInterval,
    TilingScope,
    earliest_free_interval,
)
from dataflow_mapper.scheduling.modulo import EventModuloPlacement


@dataclass(frozen=True)
class FXEventScopeNode:
    """One exact mapper-owned scope containing scheduled events and child scopes."""

    scope_path: tuple[TilingScope, ...]
    # Event-graph node ids; the event graph owns their attributes.
    activities: tuple[str, ...] = ()
    children: tuple[FXEventScopeNode, ...] = ()

    @property
    def axis(self) -> str | None:
        """Axis owning this scope, or ``None`` for the root setup scope."""
        return self.scope_path[-1].axis if self.scope_path else None

    @property
    def repetitions(self) -> int:
        """Number of tile coordinates represented by the innermost scope."""
        return self.scope_path[-1].factor if self.scope_path else 1


@dataclass(frozen=True)
class FXEventScopeTree:
    """Hierarchical grouping of mapper events without an application-IR mirror."""

    root: FXEventScopeNode
    axis_order: tuple[str, ...]
    repetitions: Mapping[str, int]

    def __post_init__(self) -> None:
        """Freeze the axis repetition map for downstream hierarchical scheduling."""
        object.__setattr__(
            self, "repetitions", MappingProxyType(dict(self.repetitions))
        )


def build_fx_event_scope_tree(
    event_graph: nx.DiGraph,
    axis_order: tuple[str, ...],
    repetitions: Mapping[str, int],
) -> FXEventScopeTree:
    """Group exact-scope events and lifetimes into a validated hierarchy tree.

    Steps:
        1. Reject mismatched axis order and repetition maps.
        2. Register each event under its exact scope path, creating the prefix
           scopes and their parent -> child links.
        3. Validate cross-scope dependencies: parent-owned edges and ordered
           same-contracting-axis accumulator edges are allowed, sibling or
           unrelated parent crossings are rejected.
        4. Freeze the records into a recursive ``FXEventScopeNode`` tree.

    Args:
        event_graph: Event graph whose nodes carry ``scopes`` paths.
        axis_order: Ordered axis names that every scope path must prefix.
        repetitions: Tile factor per axis.

    Returns:
        The root ``FXEventScopeTree`` with per-scope activities and children.

    Raises:
        ValueError: When axes and repetitions disagree, or a dependency crosses
            sibling or parent scopes.

    Example:
        >>> tree = build_fx_event_scope_tree(event_graph, ("M",), {"M": 2})
    """
    if len(set(axis_order)) != len(axis_order) or set(axis_order) != set(repetitions):
        raise ValueError(
            "FX event hierarchy requires unique axes and matching repetitions"
        )
    records: dict[tuple[TilingScope, ...], dict[str, object]] = {
        (): {"activities": [], "children": set()}
    }
    event_paths = {
        event_id: tuple(data["scopes"])
        for event_id, data in event_graph.nodes(data=True)
    }

    def register(path: tuple[TilingScope, ...], event_id: str) -> None:
        positions = [axis_order.index(scope.axis) for scope in path]
        if positions != list(range(len(path))):
            raise ValueError(
                "FX event scopes must be ordered prefixes of the configured axes"
            )
        for index, scope in enumerate(path):
            if (
                scope.factor != repetitions[scope.axis]
                or scope.tile_index >= scope.factor
            ):
                raise ValueError(
                    "FX event scope conflicts with its configured repetition"
                )
            prefix = path[: index + 1]
            records.setdefault(prefix, {"activities": [], "children": set()})
            parent = path[:index]
            records[parent]["children"].add(prefix)
        records[path]["activities"].append(event_id)

    for event_id in event_graph.nodes:
        register(event_paths[event_id], event_id)
    for event_id in event_graph.nodes:
        owner_path = event_paths[event_id]
        for dependency_id in event_graph.predecessors(event_id):
            dependency_path = event_paths.get(dependency_id)
            if dependency_path is None:
                raise ValueError(
                    f"FX event dependency {dependency_id!r} has no owner scope"
                )
            if owner_path[: len(dependency_path)] == dependency_path:
                continue
            if owner_path == dependency_path[: len(owner_path)] and any(
                scope.contracting for scope in dependency_path[len(owner_path) :]
            ):
                # A completed contraction child may feed an operation owned by
                # its parent scope, which is scheduled after all reduction tiles.
                continue
            common = 0
            while (
                common < min(len(owner_path), len(dependency_path))
                and owner_path[common] == dependency_path[common]
            ):
                common += 1
            if common < min(len(owner_path), len(dependency_path)):
                owner_scope = owner_path[common]
                dependency_scope = dependency_path[common]
                if owner_scope.axis == dependency_scope.axis:
                    # A dependency between sibling tiles of one axis is lifted
                    # to this parent scope. Serial composition below orders the
                    # dependent child scopes; a future modulo parent can instead
                    # lower them as recurrences.
                    continue
            raise ValueError("FX event dependency crosses unrelated parent scopes")

    def freeze(path: tuple[TilingScope, ...]) -> FXEventScopeNode:
        record = records[path]
        child_paths = sorted(
            record["children"],
            key=lambda child: tuple(
                (axis_order.index(scope.axis), scope.tile_index) for scope in child
            ),
        )
        return FXEventScopeNode(
            path,
            tuple(record["activities"]),
            tuple(freeze(child) for child in child_paths),
        )

    return FXEventScopeTree(freeze(()), axis_order, repetitions)


@dataclass(frozen=True)
class FXHierarchicalEventSchedule:
    """Mapper-owned recursive timing summary for a scope-aware FX event tree."""

    event_tree: FXEventScopeTree
    makespan_seconds: float
    innermost_initiation_interval_seconds: float
    scope_makespans_seconds: Mapping[tuple[TilingScope, ...], float]
    peak_memory_bytes: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze per-scope timing summaries."""
        if self.makespan_seconds < 0 or self.innermost_initiation_interval_seconds < 0:
            raise ValueError("FX hierarchical schedule times must be non-negative")
        object.__setattr__(
            self,
            "scope_makespans_seconds",
            MappingProxyType(dict(self.scope_makespans_seconds)),
        )
        object.__setattr__(
            self, "peak_memory_bytes", MappingProxyType(dict(self.peak_memory_bytes))
        )


def schedule_hierarchical_fx_events(
    event_graph: nx.DiGraph,
    event_tree: FXEventScopeTree,
    modulo_placements: Mapping[tuple[TilingScope, ...], EventModuloPlacement],
) -> FXHierarchicalEventSchedule:
    """Compose parent setup, nested child scopes, and modulo-placed leaf bodies.

    Each scope schedules its local events, then composes its children. A leaf
    child uses its modulo placement regardless of whether its axis is
    contracting; a non-leaf child is composed recursively. Multiple child
    coordinates are ordered by their cross-tile dependencies and run serially.
    Capacity is enforced elsewhere -- by ``schedule_modulo_event_body`` for
    periodic residency and by the caller's whole-graph check for composed
    residency -- so this pass only orders events.

    Args:
        event_graph: Physical event graph with per-node intervals and metrics.
        event_tree: Scope tree over the event graph's tiling scopes.
        modulo_placements: Capacity-bounded modulo placement per leaf scope.

    Returns:
        The composed ``FXHierarchicalEventSchedule``. Every non-storage event is
        stamped with its absolute ``start_seconds``/``finish_seconds`` so the
        caller can rebuild residency from the composed timeline;
        ``peak_memory_bytes`` is left empty for the caller to fill.

    Example:
        Compose a two-tile tree with one modulo placement per tile::

            schedule = schedule_hierarchical_fx_events(event_graph, tree, placements)
            schedule.makespan_seconds
    """
    scope_durations: dict[tuple[TilingScope, ...], float] = {}
    initiation_intervals: list[float] = []

    def schedule_node(node: FXEventScopeNode, offset: float) -> float:
        """Schedule one scope's events and recursively compose its child scopes.

        Steps:
            1. Split the scope's own events into pre-child and post-child groups.
            2. Schedule the pre-child group at ``offset``.
            3. Compose children: recursively for non-leaf scopes, or stamp the
               modulo placement for a leaf scope. Multiple child coordinates
               are ordered by cross-tile dependencies and run serially.
            4. Schedule the post-child group after the children and return the
               scope's duration.

        Args:
            node: Scope-tree node being scheduled.
            offset: Absolute time at which the scope may begin.

        Returns:
            The scope's duration in seconds (finish minus ``offset``), not an
            absolute time.

        Example:
            >>> makespan = schedule_node(event_tree.root, 0.0)
        """
        descendant_event_ids = {
            event_id
            for child in node.children
            for descendant in _walk_fx_event_scopes(child)
            for event_id in descendant.activities
        }
        # An event must run after the children when it depends on them, directly
        # or through another event in this scope (e.g. a reduction that reads the
        # child partials, then a transfer that reads the reduction).
        deferred = set(descendant_event_ids)
        changed = True
        while changed:
            changed = False
            for event_id in node.activities:
                if event_id in deferred:
                    continue
                if any(
                    dependency in deferred
                    for dependency in event_graph.predecessors(event_id)
                ):
                    deferred.add(event_id)
                    changed = True
        pre_children = tuple(e for e in node.activities if e not in deferred)
        post_children = tuple(e for e in node.activities if e in deferred)
        pre_finish = _schedule_fx_event_group(event_graph, pre_children, offset)
        if not node.children and len(node.scope_path) < len(event_tree.axis_order):
            raise ValueError("FX hierarchy is missing a nested tile scope")
        if not node.children:
            scope_durations[node.scope_path] = pre_finish
            return pre_finish

        child_axes = {child.axis for child in node.children}
        if len(child_axes) != 1 or None in child_axes:
            raise ValueError("FX hierarchy children must share one ordered tile axis")
        child_axis = next(iter(child_axes))
        expected_axis = event_tree.axis_order[len(node.scope_path)]
        if child_axis != expected_axis:
            raise ValueError(
                "FX hierarchy scope axes do not match the configured order"
            )
        # One logical child scope represents the whole axis, regardless of its
        # parfactor. Branch instances are events inside that scope, not child
        # scope coordinates. Multiple child coordinates are ordered by their
        # cross-tile dependencies before serial composition.
        children = tuple(node.children)
        if len(children) > 1:
            # A completed contraction child may feed a parent operation, but
            # sibling child coordinates must still respect their data edges.
            child_of_event: dict[str, int] = {}
            for index, child in enumerate(children):
                for descendant in _walk_fx_event_scopes(child):
                    for event_id in descendant.activities:
                        child_of_event[event_id] = index
            dependencies_by_child: dict[int, set[int]] = {
                index: set() for index in range(len(children))
            }
            for event_id, owner_index in child_of_event.items():
                for dependency_id in event_graph.predecessors(event_id):
                    dependency_index = child_of_event.get(dependency_id)
                    if (
                        dependency_index is not None
                        and dependency_index != owner_index
                    ):
                        dependencies_by_child[owner_index].add(dependency_index)
            remaining = set(range(len(children)))
            ordered_children: list[int] = []
            while remaining:
                ready = [
                    index
                    for index in remaining
                    if dependencies_by_child[index].isdisjoint(remaining)
                ]
                if not ready:
                    raise ValueError("FX cross-tile dependency cycle")
                index = min(ready)
                ordered_children.append(index)
                remaining.remove(index)
            start = offset + pre_finish
            for index in ordered_children:
                start += schedule_node(children[index], start)
            child_finish = start - (offset + pre_finish)
        # A leaf scope's modulo placement covers all of its activities, so use
        # it regardless of whether the axis is contracting. A non-leaf scope
        # still needs its children composed, so recurse and serialize around
        # them; its own activities are placed by the recursive call.
        elif children[0].children or (
            modulo_placements.get(children[0].scope_path) is None
        ):
            # Serial composition: the child starts where the parent setup ended.
            start = offset + pre_finish
            start += schedule_node(children[0], start)
            child_finish = start - (offset + pre_finish)
        else:
            placement = modulo_placements[children[0].scope_path]
            scope_durations[children[0].scope_path] = (
                placement.body_finish_seconds
            )
            initiation_intervals.append(placement.initiation_interval_seconds)
            # One logical scope owns all branch instances; the modulo placement
            # already covers them, so only stamp their body starts once.
            tile_offset = offset + pre_finish
            for event_id, local_start in placement.starts.items():
                event_data = event_graph.nodes[event_id]
                start = tile_offset + local_start
                event_data["start_seconds"] = start
                event_data["finish_seconds"] = start + float(event_data["duration"])
            child_finish = placement.body_finish_seconds
        post_finish = _schedule_fx_event_group(
            event_graph, post_children, offset + pre_finish + child_finish
        )
        finish = pre_finish + child_finish + post_finish
        scope_durations[node.scope_path] = finish
        return finish

    makespan = schedule_node(event_tree.root, 0.0)
    return FXHierarchicalEventSchedule(
        event_tree,
        makespan,
        max(initiation_intervals, default=0.0),
        scope_durations,
    )


def _walk_fx_event_scopes(root: FXEventScopeNode) -> tuple[FXEventScopeNode, ...]:
    """Return every scope node in depth-first order, starting at ``root``.

    Args:
        root: Scope node to walk.

    Returns:
        ``root`` followed by every descendant scope, depth-first.

    Example:
        >>> _walk_fx_event_scopes(tree.root)
    """
    return (
        root,
        *(node for child in root.children for node in _walk_fx_event_scopes(child)),
    )


def _schedule_fx_event_group(
    event_graph: nx.DiGraph, event_ids: tuple[str, ...], offset: float
) -> float:
    """Schedule one scope's event group in absolute time, recording intervals.

    Events are placed no earlier than ``offset`` and respect their internal
    dependencies; each event's absolute ``start_seconds``/``finish_seconds`` is
    written back onto the graph so the caller can rebuild residency from the
    composed timeline.

    Args:
        event_graph: Event graph holding the events and their resources.
        event_ids: Event ids owned by this scope, scheduled as one group.
        offset: Absolute time at which the group may begin.

    Returns:
        The group's duration (``0.0`` when ``event_ids`` is empty).

    Example:
        Schedule a scope's setup events at the scope's start::

            span = _schedule_fx_event_group(event_graph, pre_children, offset)
    """
    # Only intra-group dependencies drive the order; edges from outside this
    # group are honored by the caller-chosen ``offset``.
    ids = set(event_ids)
    dependencies = {
        event_id: {dep for dep in event_graph.predecessors(event_id) if dep in ids}
        for event_id in event_ids
    }
    calendars: dict[str, list[EventInterval]] = {}
    finishes: dict[str, float] = {}
    # Greedy list schedule within the group, never earlier than ``offset``.
    while len(finishes) < len(event_ids):
        # Ready = all intra-group predecessors placed (sorted for determinism).
        ready = sorted(
            event_id
            for event_id in ids - set(finishes)
            if dependencies[event_id].issubset(finishes)
        )
        if not ready:
            raise ValueError("FX scope events contain a cyclic dependency")
        event_id = ready[0]
        data = event_graph.nodes[event_id]
        # Start once dependencies finish, then wait for free resources. Storage
        # nodes are passive hand-offs: they order the chain but reserve nothing.
        earliest = max(
            (finishes[item] for item in dependencies[event_id]), default=offset
        )
        if data.get("kind") == "storage":
            duration = 0.0
            resource_ids = ()
        else:
            duration = float(data["duration"])
            resource_ids = tuple(
                data.get("resources") or (str(data["resource_id"]),)
            )
        interval = earliest_free_interval(calendars, resource_ids, earliest, duration)
        for resource_id in resource_ids:
            calendars.setdefault(resource_id, []).append(interval)
        finishes[event_id] = interval.finish_seconds
        # Write absolute times back so residency can be rebuilt from the graph.
        data["start_seconds"] = interval.start_seconds
        data["finish_seconds"] = interval.finish_seconds
    if not finishes:
        return 0.0
    return max(finishes.values()) - offset
