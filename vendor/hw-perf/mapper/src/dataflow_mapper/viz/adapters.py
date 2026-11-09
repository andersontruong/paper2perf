"""Translate mapping-pipeline artifacts into the display schema.

Each public adapter turns one phase's data into a
:class:`~dataflow_mapper.viz.schema.HierarchicalGraph` (a drill-down scope tree)
or a set of per-scope :class:`~dataflow_mapper.viz.schema.GanttChart` charts.
Nothing here renders; the renderers consume the schema.

Scope tree
----------
Tiling nests one ``call_module`` per axis and ``ParallelModel.concretize``
replicates each tile body, so walking the FX module tree yields one scope per
module instance. A node whose name points into a child module becomes a
drill-down link. Scope ids are ``/``-joined ``axis+tile`` paths, e.g.
``root/M0/N1/K0``. The walk replays the same scope-key rule as
``dataflow_mapper.execution._map_child_scopes`` so a scope can be matched to its
:class:`~dataflow_mapper.viz.trace.ScopeTrace`.

Phases
------
* :func:`flat_graph_from_graph_module` - one flat PyTorch/ATen graph (Phase 1).
* :func:`hierarchy_from_module_tree` - the tiled model's scope tree (Phase 2).
* :func:`placed_hierarchy` / :func:`routed_hierarchy` - the mapper task graph per
  scope, annotated with placement (6a) or routes (6b).
* :func:`event_hierarchy` - the physical event graph per scope (6c).
* :func:`gantt_charts` - per-scope resource timelines and storage lifetimes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import sympy as sp
import torch

from dataflow_mapper.scheduling.model import ScopeKey, TilingScope
from dataflow_mapper.viz.schema import (
    ABSOLUTE_TIMELINE,
    COMPUTE_TRACK,
    LINK_TRACK,
    MEMORY_TRACK,
    RELATIVE_TIMELINE,
    FlatGraph,
    GanttBar,
    GanttChart,
    GanttTrack,
    GraphEdge,
    GraphNode,
    HierarchicalGraph,
)
from dataflow_mapper.viz.trace import MappingTrace, ScopeTrace

# FX ``node.op`` -> display kind, matching the renderer's fill palette.
_KIND_BY_OP: Mapping[str, str] = {
    "placeholder": "placeholder",
    "get_attr": "get_attr",
    "call_function": "op",
    "call_module": "call_module",
    "output": "output",
}


@dataclass(frozen=True)
class _Scope:
    """One visited FX module: its graph, drill links, and trace identity."""

    scope_id: str
    parent: str | None
    gm: torch.fx.GraphModule
    trace_key: ScopeKey
    children: Mapping[str, str]


def _referenced_nodes(value: object) -> tuple[torch.fx.Node, ...]:
    """Flatten one FX argument to the nodes it references, in traversal order."""
    found: list[torch.fx.Node] = []
    torch.fx.node.map_arg(
        value,
        lambda item: found.append(item) if isinstance(item, torch.fx.Node) else item,
    )
    return tuple(found)


def _bind(value: object, bindings: Mapping[str, int] | None) -> object:
    """Substitute concrete bindings into a symbolic shape/FLOP expression."""
    if bindings and isinstance(value, sp.Basic):
        substitutions = {
            symbol: bindings[symbol.name]
            for symbol in value.free_symbols
            if symbol.name in bindings
        }
        if substitutions:
            value = value.subs(substitutions)
    return value


def _format_shape(shape: object, bindings: Mapping[str, int] | None) -> str:
    """Format an FX shape (optionally nested) as ``[d0, d1, ...]``."""
    dimensions = shape
    if (
        isinstance(dimensions, (list, tuple))
        and len(dimensions) == 1
        and isinstance(dimensions[0], (list, tuple))
    ):
        dimensions = dimensions[0]
    if isinstance(dimensions, (list, tuple)):
        return "[" + ", ".join(str(_bind(dim, bindings)) for dim in dimensions) + "]"
    return str(_bind(dimensions, bindings))


def _format_sequence(values: object, bindings: Mapping[str, int] | None) -> str:
    """Format a tuple of symbolic values as ``(a, b, c)``."""
    if isinstance(values, (list, tuple)):
        return "(" + ", ".join(str(_bind(value, bindings)) for value in values) + ")"
    return str(_bind(values, bindings))


def _fx_kind(node: torch.fx.Node) -> str:
    """Display kind for one FX node."""
    return _KIND_BY_OP.get(node.op, "op")


def _fx_annotations(
    node: torch.fx.Node, bindings: Mapping[str, int] | None
) -> dict[str, str]:
    """Display annotations for one FX node: op target, shape, FLOPs, dtype."""
    annotations: dict[str, str] = {}
    target = getattr(node, "target", None)
    if node.op == "call_function" and target is not None:
        annotations["op"] = str(target)
    elif node.op == "call_module":
        annotations["module"] = str(target)
    shape = node.meta.get("output_size")
    if shape is not None:
        annotations["shape"] = _format_shape(shape, bindings)
    flop_req = node.meta.get("flop_req")
    if flop_req is not None:
        annotations["flop_req"] = _format_sequence(flop_req, bindings)
    dtype = node.meta.get("application_dtype")
    if dtype:
        annotations["dtype"] = str(dtype)
    return annotations


def _scopes(root_gm: torch.fx.GraphModule, root_id: str = "root") -> dict[str, _Scope]:
    """Walk the FX module tree into one :class:`_Scope` per module instance.

    Args:
        root_gm: Root FX module, symbolic or concretized.
        root_id: Scope id for the root module.

    Returns:
        Scope id -> scope record, in depth-first visit order. Shared submodules
        are visited per instance, so a concretized tile tree yields one scope per
        tile rather than one per template.
    """
    scopes: dict[str, _Scope] = {}

    def visit(
        scope_id: str,
        gm: torch.fx.GraphModule,
        parent_id: str | None,
        scope_path: ScopeKey,
        module_axis: str | None,
    ) -> None:
        module_meta = gm.meta.get("tiling")
        children: dict[str, str] = {}
        ordinals: dict[str, int] = {}
        for node in gm.graph.nodes:
            if node.op != "call_module":
                continue
            child_gm = gm.get_submodule(node.target)
            child_meta = child_gm.meta.get("tiling")
            child_axis = str(child_meta["axis"]) if child_meta is not None else None
            child_scope_path = scope_path
            if module_meta is not None:
                child_scope_path = scope_path + (
                    TilingScope(
                        str(module_meta["axis"]),
                        int(module_meta.get("factor", 2)),
                        0,
                        bool(module_meta.get("contracting", False)),
                        int(module_meta.get("split", 0)),
                    ),
                )
            # One logical scope per axis; repeated calls to the same child scope
            # share the same path and display id.
            child_scope_path = (*child_scope_path, node.target)
            if child_meta is not None:
                base = (
                    f"{child_meta['axis']}"
                    f".{int(child_meta.get('split', 0))}"
                )
                child_id = f"{scope_id}/{base}"
            elif module_meta is not None:
                base = (
                    f"leaf-{module_meta['axis']}"
                    f".{int(module_meta.get('split', 0))}"
                )
                child_id = f"{scope_id}/{base}"
            else:
                base = "scope"
                ordinals[base] = ordinals.get(base, 0) + 1
                suffix = "" if ordinals[base] == 1 else f"-{ordinals[base]}"
                child_id = f"{scope_id}/{base}{suffix}"
            children[node.name] = child_id
            visit(child_id, child_gm, scope_id, child_scope_path, child_axis)
        scopes[scope_id] = _Scope(
            scope_id, parent_id, gm, scope_path, children
        )

    visit(root_id, root_gm, None, (), None)
    return scopes


def _fx_flat(
    scope: _Scope,
    bindings: Mapping[str, int] | None,
    *,
    title: str | None = None,
) -> FlatGraph:
    """Flat display graph of one scope's FX nodes, with child drill links."""
    nodes = tuple(
        GraphNode(
            id=node.name,
            label=node.name,
            kind=_fx_kind(node),
            annotations=_fx_annotations(node, bindings),
            child_scope=scope.children.get(node.name),
        )
        for node in scope.gm.graph.nodes
    )
    edges: list[GraphEdge] = []
    seen: set[tuple[str, str]] = set()
    for node in scope.gm.graph.nodes:
        for argument in (*node.args, *node.kwargs.values()):
            for source in _referenced_nodes(argument):
                key = (source.name, node.name)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(GraphEdge(source=source.name, target=node.name))
    return FlatGraph(
        scope_id=scope.scope_id,
        title=title or scope.scope_id,
        nodes=nodes,
        edges=tuple(edges),
    )


def _task_flat(scope: _Scope, graph: nx.MultiDiGraph, *, routed: bool) -> FlatGraph:
    """Flat display graph of one scope's mapper task graph.

    Compute tasks carry their placement (``resource``/``seconds``) and, when
    ``routed``, each edge carries its selected link route and transfer time.
    ``call_module`` wrapper tasks become drill links into their child scope.
    """
    nodes = tuple(
        GraphNode(
            id=name,
            label=name,
            kind=str(data.get("kind", "compute")),
            annotations=_task_annotations(data),
            child_scope=scope.children.get(name),
        )
        for name, data in graph.nodes(data=True)
    )
    edges = tuple(
        GraphEdge(
            source=source,
            target=target,
            label=str(data.get("value_id", "")),
            annotations=_route_annotations(data) if routed else {},
        )
        for source, target, _key, data in graph.edges(keys=True, data=True)
    )
    return FlatGraph(
        scope_id=scope.scope_id, title=scope.scope_id, nodes=nodes, edges=edges
    )


def _task_annotations(data: Mapping[str, Any]) -> dict[str, str]:
    """Display annotations for one task node."""
    annotations: dict[str, str] = {}
    resource = data.get("resource")
    if resource is not None:
        annotations["resource"] = str(resource)
    seconds = data.get("seconds")
    if seconds is not None:
        annotations["seconds"] = f"{float(seconds):.3e}"
    return annotations


def _route_annotations(data: Mapping[str, Any]) -> dict[str, str]:
    """Display annotations for one routed edge."""
    annotations: dict[str, str] = {}
    route = data.get("route") or ()
    if route:
        annotations["route"] = " -> ".join(str(hop) for hop in route)
    seconds = data.get("seconds")
    if seconds is not None:
        annotations["seconds"] = f"{float(seconds):.3e}"
    return annotations


def _event_flat(scope: _Scope, event_graph: nx.DiGraph) -> FlatGraph:
    """Flat display graph of one scope's physical event graph.

    Compute events for ``call_module`` wrappers become drill links into the child
    scope; transfer events carry their link and duration.
    """
    nodes = tuple(
        GraphNode(
            id=name,
            label=name,
            kind=str(data.get("kind", "compute")),
            annotations=_event_annotations(data),
            child_scope=_event_child(scope, name),
        )
        for name, data in event_graph.nodes(data=True)
    )
    edges = tuple(
        GraphEdge(source=source, target=target)
        for source, target in event_graph.edges()
    )
    return FlatGraph(
        scope_id=scope.scope_id, title=scope.scope_id, nodes=nodes, edges=edges
    )


def _event_child(scope: _Scope, name: str) -> str | None:
    """Drill link for a compute event naming a child ``call_module``."""
    if name.startswith("compute:"):
        return scope.children.get(name[len("compute:") :])
    return None


def _event_annotations(data: Mapping[str, Any]) -> dict[str, str]:
    """Display annotations for one event node."""
    annotations: dict[str, str] = {}
    resource = data.get("resource_id")
    if resource is not None:
        annotations["resource"] = str(resource)
    duration = data.get("duration")
    if duration is not None:
        annotations["duration"] = f"{float(duration):.3e}"
    start = data.get("start_seconds")
    if start is not None:
        annotations["start"] = f"{float(start):.3e}"
    finish = data.get("finish_seconds")
    if finish is not None:
        annotations["finish"] = f"{float(finish):.3e}"
    return annotations


def _trace_by_key(
    trace: MappingTrace,
) -> dict[ScopeKey, ScopeTrace]:
    """Index a trace's scopes by their path-only scope key."""
    return {scope.scope_path: scope for scope in trace.scopes}


def flat_graph_from_graph_module(
    gm: torch.fx.GraphModule,
    *,
    scope_id: str = "model",
    title: str | None = None,
    bindings: Mapping[str, int] | None = None,
) -> FlatGraph:
    """Display one flat PyTorch/ATen graph (Phase 1).

    Args:
        gm: FX module to display; no tiling wrappers are expected.
        scope_id: Scope id for the single scope.
        title: Optional section title; defaults to ``scope_id``.
        bindings: Concrete symbol bindings; when given, symbolic shapes and FLOP
            expressions are substituted so the same graph can be shown symbolic
            (``None``) or concrete.

    Returns:
        A one-scope :class:`FlatGraph` with op/shape/FLOP annotations.

    Example:
        >>> symbolic = flat_graph_from_graph_module(model.gm, title="symbolic")
        >>> concrete = flat_graph_from_graph_module(model.gm, bindings={"M": 32})
    """
    scope = _Scope(scope_id, None, gm, (), {})
    return _fx_flat(scope, bindings, title=title)


def single_scope(flat: FlatGraph) -> HierarchicalGraph:
    """Wrap one flat graph as the sole scope of a hierarchical graph.

    Use this to display a phase that has no nested scopes (a symbolic or
    concretized ATen graph) through the same drill-down viewer as a tiled
    model.

    Args:
        flat: Flat display graph to use as the root scope.

    Returns:
        A :class:`HierarchicalGraph` whose root is ``flat`` and which has no
        child scopes.

    Example:
        >>> flat = flat_graph_from_graph_module(model.gm, title="symbolic")
        >>> graph = single_scope(flat)
    """
    return HierarchicalGraph(root=flat.scope_id, scopes={flat.scope_id: flat})


def hierarchy_from_module_tree(
    gm: torch.fx.GraphModule,
    *,
    root_id: str = "root",
    bindings: Mapping[str, int] | None = None,
    title: str | None = None,
) -> HierarchicalGraph:
    """Display the tiled model's ``call_module`` scope tree (Phase 2).

    Args:
        gm: Root of a tiled FX module tree, symbolic (``apply_tiling_plan``) or
            concretized.
        root_id: Scope id for the root module.
        bindings: Optional concrete symbol bindings for shape/FLOP display.
        title: Optional root scope title.

    Returns:
        A drill-down :class:`HierarchicalGraph`; each ``call_module`` node links
        to the scope holding its branch body.

    Example:
        >>> tiled = model.apply_tiling_plan(("M",))
        >>> tree = hierarchy_from_module_tree(tiled.gm)
    """
    scopes = _scopes(gm, root_id)
    graphs = {
        scope_id: _fx_flat(
            scope, bindings, title=title if scope_id == root_id else None
        )
        for scope_id, scope in scopes.items()
    }
    return HierarchicalGraph(root=root_id, scopes=graphs)


def placed_hierarchy(
    gm: torch.fx.GraphModule,
    trace: MappingTrace,
    *,
    root_id: str = "root",
    bindings: Mapping[str, int] | None = None,
) -> HierarchicalGraph:
    """Display the placed mapper artifact per scope (6a).

    Args:
        gm: Root of the FX module tree that was mapped (its concrete form).
        trace: Trace collected while mapping that same model.
        root_id: Scope id for the root module.
        bindings: Fallback bindings for scopes with no recorded task graph.

    Returns:
        A :class:`HierarchicalGraph` whose compute nodes carry
        ``resource``/``seconds``.
    """
    return _annotated_hierarchy(gm, trace, root_id, routed=False, bindings=bindings)


def routed_hierarchy(
    gm: torch.fx.GraphModule,
    trace: MappingTrace,
    *,
    root_id: str = "root",
    bindings: Mapping[str, int] | None = None,
) -> HierarchicalGraph:
    """Display the routed mapper artifact per scope (6b).

    Like :func:`placed_hierarchy`, with each data movement annotated with its
    link route and transfer time.

    Args:
        gm: Root of the mapped FX module tree.
        trace: Trace collected while mapping that same model.
        root_id: Scope id for the root module.
        bindings: Fallback bindings for scopes with no recorded task graph.

    Returns:
        A :class:`HierarchicalGraph` whose edges carry ``route``/``seconds``.
    """
    return _annotated_hierarchy(gm, trace, root_id, routed=True, bindings=bindings)


def _annotated_hierarchy(
    gm: torch.fx.GraphModule,
    trace: MappingTrace,
    root_id: str,
    *,
    routed: bool,
    bindings: Mapping[str, int] | None,
) -> HierarchicalGraph:
    """Build a scope tree from the module walk, annotating from the trace."""
    scopes = _scopes(gm, root_id)
    by_key = _trace_by_key(trace)
    graphs: dict[str, FlatGraph] = {}
    for scope_id, scope in scopes.items():
        recorded = by_key.get(scope.trace_key)
        graph = None
        if recorded is not None:
            graph = recorded.routed_graph if routed else recorded.placed_graph
        if graph is None:
            graphs[scope_id] = _fx_flat(scope, bindings)
        else:
            graphs[scope_id] = _task_flat(scope, graph, routed=routed)
    return HierarchicalGraph(root=root_id, scopes=graphs)


def event_hierarchy(
    gm: torch.fx.GraphModule,
    trace: MappingTrace,
    *,
    root_id: str = "root",
    bindings: Mapping[str, int] | None = None,
) -> HierarchicalGraph:
    """Display the physical event graph per scope (6c).

    Args:
        gm: Root of the mapped FX module tree.
        trace: Trace collected while mapping that same model.
        root_id: Scope id for the root module.
        bindings: Fallback bindings for scopes with no recorded event graph.

    Returns:
        A :class:`HierarchicalGraph` whose nodes carry event timing; compute
        events for child wrappers drill into their scope.
    """
    scopes = _scopes(gm, root_id)
    by_key = _trace_by_key(trace)
    graphs: dict[str, FlatGraph] = {}
    for scope_id, scope in scopes.items():
        recorded = by_key.get(scope.trace_key)
        event_graph = recorded.event_graph if recorded is not None else None
        if event_graph is None:
            graphs[scope_id] = _fx_flat(scope, bindings)
        else:
            graphs[scope_id] = _event_flat(scope, event_graph)
    return HierarchicalGraph(root=root_id, scopes=graphs)


def gantt_charts(
    gm: torch.fx.GraphModule,
    trace: MappingTrace,
    *,
    root_id: str = "root",
    timelines: Sequence[str] = (ABSOLUTE_TIMELINE, RELATIVE_TIMELINE),
) -> tuple[GanttChart, ...]:
    """Build one resource timeline per scope and timeline selector (6c).

    Compute events occupy their compute resource, transfer events their link, and
    storage windows their memory (as tensor lifetimes with byte sizes). The
    windows a scope carries have already been chained across the scopes that
    share a value by the global storage ledger, so a shared input's bar can span
    from its birth scope onward.

    Args:
        gm: Root of the mapped FX module tree.
        trace: Trace collected while mapping that same model.
        root_id: Scope id for the root module.
        timelines: Timeline selectors to emit per scope.

    Returns:
        One :class:`GanttChart` per (scope, timeline) that has a recorded event
        graph, in scope visit order.
    """
    scopes = _scopes(gm, root_id)
    by_key = _trace_by_key(trace)
    # Resource class seen anywhere in the trace, so a multi-resource event's
    # reserved lanes can be classified even when this scope carries no event on
    # that resource.
    resource_kinds: dict[str, str] = {}
    for scope in scopes.values():
        recorded = by_key.get(scope.trace_key)
        if recorded is None or recorded.event_graph is None:
            continue
        for _name, data in recorded.event_graph.nodes(data=True):
            resource_id = str(data.get("resource_id", ""))
            if not resource_id:
                continue
            kind = str(data.get("kind", "compute"))
            track = (
                MEMORY_TRACK
                if kind == "storage"
                else LINK_TRACK if kind == "transfer" else COMPUTE_TRACK
            )
            resource_kinds.setdefault(resource_id, track)
    charts: list[GanttChart] = []
    for scope_id, scope in scopes.items():
        recorded = by_key.get(scope.trace_key)
        if recorded is None or recorded.event_graph is None:
            continue
        for timeline in timelines:
            charts.append(_gantt_chart(scope_id, recorded, timeline, resource_kinds))
    return tuple(charts)


def _transfer_endpoint(
    event_graph: nx.DiGraph, event_id: str, *, upstream: bool
) -> str | None:
    """Resource a transfer leaves from (``upstream``) or arrives at.

    A transfer is mediated by storage nodes after scheduling, so its neighbor in
    the event graph names the memory (or compute node) on that side.
    """
    neighbors = (
        event_graph.predecessors(event_id)
        if upstream
        else event_graph.successors(event_id)
    )
    for neighbor in neighbors:
        node = event_graph.nodes[neighbor]
        if node.get("kind") in ("compute", "storage"):
            return str(node["resource_id"])
    return None


# Track kind -> the bar kind that gives a lane's native fill color and hover.
_BAR_KIND_BY_TRACK = {
    COMPUTE_TRACK: "compute",
    MEMORY_TRACK: "storage",
    LINK_TRACK: "transfer",
}


def _resident_peak(bars: list[GanttBar]) -> int:
    """Peak concurrent bytes held by a lane's (non-reserved) storage bars."""
    deltas: list[tuple[float, int]] = []
    for bar in bars:
        if bar.reserved:
            continue
        size = bar.annotations.get("bytes")
        if size is None:
            continue
        deltas.append((bar.start, int(size)))
        deltas.append((bar.finish, -int(size)))
    deltas.sort()
    peak = current = 0
    for _time, delta in deltas:
        current += delta
        peak = max(peak, current)
    return peak


def _gantt_chart(
    scope_id: str,
    recorded: ScopeTrace,
    timeline: str,
    resource_kinds: dict[str, str] | None = None,
) -> GanttChart:
    """Build one scope's resource timeline.

    The absolute timeline renders on the **global** (whole-model) timeline by
    adding the scope's start offset, so charts from different scopes line up.
    The relative timeline folds a modulo-pipelined scope into one initiation
    interval and otherwise shifts the scope to begin at zero.
    """
    event_graph = recorded.event_graph
    assert event_graph is not None
    starts = [float(data["start_seconds"]) for _, data in event_graph.nodes(data=True)]
    origin = min(starts, default=0.0)
    offset = (
        float(event_graph.graph.get("global_offset", 0.0))
        if timeline == ABSOLUTE_TIMELINE
        else 0.0
    )
    period = None
    if timeline == RELATIVE_TIMELINE and recorded.schedule is not None:
        interval = recorded.schedule.innermost_initiation_interval_seconds
        period = interval if interval > 0 else None

    def place(start: float, finish: float) -> tuple[float, float]:
        duration = finish - start
        if timeline == ABSOLUTE_TIMELINE:
            return start + offset, finish + offset
        if period is not None:
            folded = start % period
            return folded, min(folded + duration, period)
        return start - origin, finish - origin

    bars_by_resource: dict[str, list[GanttBar]] = {}
    kind_by_resource: dict[str, str] = {}

    def add(resource_id: str, kind: str, bar: GanttBar) -> None:
        bars_by_resource.setdefault(resource_id, []).append(bar)
        kind_by_resource.setdefault(resource_id, kind)

    # Every lane comes from the event graph: compute events on compute resources,
    # transfer events on *directional* link lanes, and storage nodes on memories.
    # Splitting a link into one lane per direction is display-only; the schedule
    # still reserves the physical link as a single resource.
    for name, data in event_graph.nodes(data=True):
        kind = str(data.get("kind", "compute"))
        # An aggregate scope reserves several resources and has no single owner,
        # so it is drawn only as reserved bars (the second pass) instead of a
        # solid primary bar on one arbitrary lane.
        if kind == "compute" and len(tuple(data.get("resources") or ())) > 1:
            continue
        start, finish = place(
            float(data["start_seconds"]), float(data["finish_seconds"])
        )
        label = str(data.get("source_node_name", name))
        annotations: dict[str, str] = {}
        if kind == "storage":
            track = MEMORY_TRACK
            lane = str(data["resource_id"])
            if "size_bytes" in data:
                annotations = {"bytes": str(data["size_bytes"])}
        elif kind == "transfer":
            track = LINK_TRACK
            source_resource = _transfer_endpoint(event_graph, name, upstream=True)
            destination_resource = _transfer_endpoint(event_graph, name, upstream=False)
            lane = (
                f"{source_resource} -> {destination_resource}"
                if source_resource and destination_resource
                else str(data["resource_id"])
            )
        else:
            track = COMPUTE_TRACK
            lane = str(data["resource_id"])
        add(
            lane,
            track,
            GanttBar(start, finish, label=label, kind=kind, annotations=annotations),
        )

    # A multi-resource event (an aggregate scope) holds every resource its body
    # uses for the whole makespan, not just the representative one. Mirror its
    # bar onto each reserved lane that exists, flagged ``reserved`` so the
    # renderer de-emphasizes it. A physical link mirrors onto whichever
    # directional lanes this chart carries.
    def reserved_lanes(resource_id: str) -> tuple[tuple[str, str], ...]:
        if resource_id in bars_by_resource:
            return ((resource_id, kind_by_resource[resource_id]),)
        kind = (resource_kinds or {}).get(resource_id)
        if kind == LINK_TRACK or (kind is None and "::" in resource_id):
            head, tail = resource_id.split("::")[:2]
            candidates = (f"{head} -> {tail}", f"{tail} -> {head}")
            existing = tuple(
                (lane, LINK_TRACK) for lane in candidates if lane in bars_by_resource
            )
            return existing or tuple((lane, LINK_TRACK) for lane in candidates)
        if kind in (COMPUTE_TRACK, MEMORY_TRACK):
            return ((resource_id, kind),)
        return ()

    for name, data in event_graph.nodes(data=True):
        if data.get("kind") != "compute":
            continue
        reserved = [str(resource_id) for resource_id in data.get("resources", ())]
        if len(reserved) <= 1:
            continue
        memory_peaks: dict[str, int] = {}
        start, finish = place(
            float(data["start_seconds"]), float(data["finish_seconds"])
        )
        label = str(data.get("source_node_name", name))
        for resource_id in reserved:
            for lane, lane_kind in reserved_lanes(resource_id):
                annotations: dict[str, str] = {}
                if lane_kind == MEMORY_TRACK:
                    # An aliased child boundary may reserve a memory without
                    # leaving any storage bar in this scope.
                    lane_bars = bars_by_resource.get(lane)
                    if not lane_bars:
                        continue
                    # The reservation carries no value of its own, so show how
                    # much the memory holds while it is held.
                    if lane not in memory_peaks:
                        memory_peaks[lane] = _resident_peak(lane_bars)
                    annotations = {"bytes": str(memory_peaks[lane])}
                add(
                    lane,
                    lane_kind,
                    GanttBar(
                        start,
                        finish,
                        label=label,
                        kind=_BAR_KIND_BY_TRACK[lane_kind],
                        annotations=annotations,
                        reserved=True,
                    ),
                )
    tracks = tuple(
        GanttTrack(
            resource_id=resource_id,
            kind=kind_by_resource[resource_id],
            bars=tuple(sorted(bars, key=lambda bar: (bar.start, bar.finish))),
        )
        for resource_id, bars in sorted(bars_by_resource.items())
    )
    makespan = (
        recorded.schedule.makespan_seconds if recorded.schedule is not None else None
    )
    if makespan is not None and timeline == ABSOLUTE_TIMELINE:
        makespan += offset
    frame = "global" if timeline == ABSOLUTE_TIMELINE else "scope-relative"
    return GanttChart(
        scope_id=scope_id,
        title=f"{scope_id} ({timeline}, {frame} time)",
        tracks=tracks,
        timeline=timeline,
        makespan=makespan,
    )


__all__ = [
    "event_hierarchy",
    "flat_graph_from_graph_module",
    "gantt_charts",
    "hierarchy_from_module_tree",
    "placed_hierarchy",
    "routed_hierarchy",
]
