"""Domain-agnostic display model for mapper visualizations.

These structures are the contract between the per-phase *adapters* (which read FX
graphs, task graphs, and event graphs) and the *renderer* (which writes a
self-contained HTML report). Nothing here imports FX, hardware, or scheduling
types: a scope is an opaque string id, a node or edge carries pre-formatted
display annotations, and a Gantt chart is a set of timed bars grouped by
resource.

A :class:`HierarchicalGraph` is a scope tree. The root scope is the whole graph;
each :class:`GraphNode` whose ``child_scope`` is set encapsulates a nested scope
(a fused/tiled ``call_module``) that the renderer exposes as a drill-down link.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

# Opaque id for one drill-down scope: the root graph or a nested scope body.
ScopeId = str

# Gantt track kinds: one Y-axis lane category per hardware resource class.
COMPUTE_TRACK = "compute"
MEMORY_TRACK = "memory"
LINK_TRACK = "link"

# Gantt timelines: absolute composed time, or the scope's own relative time.
ABSOLUTE_TIMELINE = "absolute"
RELATIVE_TIMELINE = "relative"


def _freeze_annotations(annotations: Mapping[str, str]) -> Mapping[str, str]:
    """Return an immutable copy of a node/edge annotation map.

    Annotations are display strings; callers format numbers before passing them.

    Args:
        annotations: Display key -> pre-formatted value.

    Returns:
        A read-only mapping with the same entries.

    Raises:
        TypeError: If any key or value is not a string.
    """
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in annotations.items()
    ):
        raise TypeError("display annotations must map strings to strings")
    return MappingProxyType(dict(annotations))


@dataclass(frozen=True)
class GraphNode:
    """One node of a flat scope graph.

    ``child_scope`` is set when the node encapsulates a nested scope (a fused
    ``call_module``): the renderer turns the node into a drill-down link.
    """

    id: str
    label: str
    kind: str = "op"
    annotations: Mapping[str, str] = field(default_factory=dict)
    child_scope: ScopeId | None = None

    def __post_init__(self) -> None:
        """Validate identity and freeze the display annotations."""
        if not self.id:
            raise ValueError("graph node requires a non-empty id")
        if not self.label:
            raise ValueError(f"graph node {self.id!r} requires a non-empty label")
        object.__setattr__(self, "annotations", _freeze_annotations(self.annotations))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "annotations": dict(self.annotations),
            "child_scope": self.child_scope,
        }


@dataclass(frozen=True)
class GraphEdge:
    """One directed value edge between two nodes of a flat scope graph."""

    source: str
    target: str
    label: str = ""
    annotations: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the display annotations."""
        object.__setattr__(self, "annotations", _freeze_annotations(self.annotations))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "annotations": dict(self.annotations),
        }


@dataclass(frozen=True)
class FlatGraph:
    """One drill-down scope: a self-contained flat graph.

    Hierarchy position is not stored here; :class:`HierarchicalGraph` derives it
    from the nodes' ``child_scope`` links.
    """

    scope_id: ScopeId
    title: str
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()

    def __post_init__(self) -> None:
        """Validate node identity and that every edge references known nodes."""
        if not self.scope_id:
            raise ValueError("flat graph requires a non-empty scope id")
        ids = [node.id for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"scope {self.scope_id!r} has duplicate node ids")
        known = set(ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(
                    f"scope {self.scope_id!r} edge "
                    f"{edge.source!r}->{edge.target!r} references an unknown node"
                )

    @property
    def child_scopes(self) -> dict[str, ScopeId]:
        """Node id -> nested scope id for every drillable node."""
        return {
            node.id: node.child_scope
            for node in self.nodes
            if node.child_scope is not None
        }

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "scope_id": self.scope_id,
            "title": self.title,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass(frozen=True)
class HierarchicalGraph:
    """A scope tree: one root scope plus every nested scope, keyed by id.

    The tree is defined solely by the nodes' ``child_scope`` links; each scope's
    single parent and the root-to-node breadcrumb are derived from them.
    """

    root: ScopeId
    scopes: Mapping[ScopeId, FlatGraph]
    _parents: Mapping[ScopeId, ScopeId] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        """Freeze the scope map, derive parents, and validate the tree."""
        scopes = dict(self.scopes)
        if self.root not in scopes:
            raise ValueError(f"root scope {self.root!r} is not among the graph scopes")
        for scope_id, graph in scopes.items():
            if graph.scope_id != scope_id:
                raise ValueError(
                    f"scope key {scope_id!r} does not match graph id "
                    f"{graph.scope_id!r}"
                )
        # Derive each non-root scope's single parent from the drill links.
        parents: dict[ScopeId, ScopeId] = {}
        for scope_id, graph in scopes.items():
            for child in graph.child_scopes.values():
                if child not in scopes:
                    raise ValueError(
                        f"scope {scope_id!r} drills into unknown scope {child!r}"
                    )
                if child == self.root:
                    raise ValueError(f"scope {scope_id!r} drills into the root scope")
                existing = parents.get(child)
                if existing is not None and existing != scope_id:
                    raise ValueError(
                        f"scope {child!r} is drilled into by both {existing!r} "
                        f"and {scope_id!r}"
                    )
                parents[child] = scope_id
        # Every scope must be reachable from the root through drill links.
        reachable = {self.root}
        frontier = [self.root]
        while frontier:
            for child in scopes[frontier.pop()].child_scopes.values():
                if child not in reachable:
                    reachable.add(child)
                    frontier.append(child)
        if reachable != set(scopes):
            unreachable = sorted(set(scopes) - reachable)
            raise ValueError(
                f"scopes unreachable from root {self.root!r}: {unreachable}"
            )
        object.__setattr__(self, "scopes", MappingProxyType(scopes))
        object.__setattr__(self, "_parents", MappingProxyType(parents))

    def parent_of(self, scope_id: ScopeId) -> ScopeId | None:
        """Return the scope that drills into ``scope_id``, or ``None`` for root.

        Args:
            scope_id: A scope present in this graph.

        Returns:
            The enclosing scope id, or ``None`` when ``scope_id`` is the root.

        Raises:
            ValueError: If ``scope_id`` is unknown.
        """
        if scope_id not in self.scopes:
            raise ValueError(f"unknown scope {scope_id!r}")
        return self._parents.get(scope_id)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "root": self.root,
            "scopes": {
                scope_id: graph.to_dict() for scope_id, graph in self.scopes.items()
            },
        }


@dataclass(frozen=True)
class GanttBar:
    """One timed interval on a Gantt track.

    ``reserved`` marks a bar mirrored onto a lane the event holds but does not
    actively work, so the renderer can draw it de-emphasized.
    """

    start: float
    finish: float
    label: str = ""
    kind: str = "event"
    annotations: Mapping[str, str] = field(default_factory=dict)
    reserved: bool = False

    def __post_init__(self) -> None:
        """Reject intervals that finish before they start.

        ``start`` may be negative on a scope's absolute timeline when a chained
        storage window predates the scope (its value was born in an ancestor).
        """
        if self.finish < self.start:
            raise ValueError("invalid gantt bar interval")
        object.__setattr__(self, "annotations", _freeze_annotations(self.annotations))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "start": self.start,
            "finish": self.finish,
            "label": self.label,
            "kind": self.kind,
            "annotations": dict(self.annotations),
            "reserved": self.reserved,
        }


@dataclass(frozen=True)
class GanttTrack:
    """One Y-axis lane: a compute resource, a memory, or a link."""

    resource_id: str
    kind: str
    bars: tuple[GanttBar, ...] = ()

    def __post_init__(self) -> None:
        """Validate the resource identity."""
        if not self.resource_id:
            raise ValueError("gantt track requires a non-empty resource id")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "resource_id": self.resource_id,
            "kind": self.kind,
            "bars": [bar.to_dict() for bar in self.bars],
        }


@dataclass(frozen=True)
class GanttChart:
    """A timed chart for one scope: resource tracks plus a timeline selector.

    ``timeline`` is :data:`ABSOLUTE_TIMELINE` for the composed whole-graph time or
    :data:`RELATIVE_TIMELINE` for the scope's own body time.
    """

    scope_id: ScopeId
    title: str
    tracks: tuple[GanttTrack, ...] = ()
    timeline: str = ABSOLUTE_TIMELINE
    makespan: float | None = None

    def __post_init__(self) -> None:
        """Validate the timeline selector, makespan, and track identity."""
        if self.timeline not in (ABSOLUTE_TIMELINE, RELATIVE_TIMELINE):
            raise ValueError(f"unknown gantt timeline {self.timeline!r}")
        if self.makespan is not None and self.makespan < 0:
            raise ValueError("gantt makespan must be non-negative")
        resource_ids = [track.resource_id for track in self.tracks]
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError(f"gantt scope {self.scope_id!r} has duplicate tracks")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "scope_id": self.scope_id,
            "title": self.title,
            "timeline": self.timeline,
            "makespan": self.makespan,
            "tracks": [track.to_dict() for track in self.tracks],
        }


@dataclass(frozen=True)
class Section:
    """One phase's visual payload in the report.

    A section renders exactly one media kind: a drill-down ``graph`` (optionally
    with per-scope ``gantt`` charts), pre-rendered inline ``svg``, or a raster
    ``image_uri``. ``group`` lets the renderer nest related sections (e.g. the
    6a/6b/6c sections under one "Phase 3" heading).
    """

    title: str
    description: str = ""
    group: str = ""
    graph: HierarchicalGraph | None = None
    gantt: tuple[GanttChart, ...] = ()
    svg: str | None = None
    image_uri: str | None = None

    def __post_init__(self) -> None:
        """Validate that the section carries one coherent payload."""
        if not self.title:
            raise ValueError("section requires a non-empty title")
        if self.svg is not None and self.image_uri is not None:
            raise ValueError(f"section {self.title!r} sets both svg and image_uri")
        if self.graph is None and self.svg is None and self.image_uri is None:
            raise ValueError(f"section {self.title!r} carries no visual payload")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "title": self.title,
            "description": self.description,
            "group": self.group,
            "graph": self.graph.to_dict() if self.graph is not None else None,
            "gantt": [chart.to_dict() for chart in self.gantt],
            "svg": self.svg,
            "image_uri": self.image_uri,
        }


@dataclass(frozen=True)
class Report:
    """The complete self-contained visualization: an ordered list of sections."""

    title: str
    sections: tuple[Section, ...] = ()

    def __post_init__(self) -> None:
        """Validate the report title."""
        if not self.title:
            raise ValueError("report requires a non-empty title")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "title": self.title,
            "sections": [section.to_dict() for section in self.sections],
        }


__all__ = [
    "ABSOLUTE_TIMELINE",
    "COMPUTE_TRACK",
    "LINK_TRACK",
    "MEMORY_TRACK",
    "RELATIVE_TIMELINE",
    "FlatGraph",
    "GanttBar",
    "GanttChart",
    "GanttTrack",
    "GraphEdge",
    "GraphNode",
    "HierarchicalGraph",
    "Report",
    "ScopeId",
    "Section",
]
