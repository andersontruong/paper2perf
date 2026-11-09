"""Visualization support for the dataflow mapper.

Pass a :class:`~dataflow_mapper.viz.trace.MappingTrace` to
``dataflow_mapper.execution.map_fx_graph`` as ``trace=`` to collect the placed,
routed, and scheduled artifacts of every tiled scope, then translate them into
the display structures in :mod:`dataflow_mapper.viz.schema` for rendering.
"""

from dataflow_mapper.viz.adapters import (
    event_hierarchy,
    flat_graph_from_graph_module,
    gantt_charts,
    hierarchy_from_module_tree,
    placed_hierarchy,
    routed_hierarchy,
    single_scope,
)
from dataflow_mapper.viz.graphviz import flat_graph_to_svg
from dataflow_mapper.viz.html import GanttRenderer, render_report, write_report
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
    Report,
    ScopeId,
    Section,
)
from dataflow_mapper.viz.trace import (
    MappingTrace,
    ScopeKey,
    ScopeTrace,
)

__all__ = [
    "ABSOLUTE_TIMELINE",
    "COMPUTE_TRACK",
    "LINK_TRACK",
    "MEMORY_TRACK",
    "RELATIVE_TIMELINE",
    "FlatGraph",
    "GanttBar",
    "GanttChart",
    "GanttRenderer",
    "GanttTrack",
    "GraphEdge",
    "GraphNode",
    "HierarchicalGraph",
    "MappingTrace",
    "Report",
    "ScopeId",
    "ScopeKey",
    "ScopeTrace",
    "Section",
    "event_hierarchy",
    "flat_graph_from_graph_module",
    "flat_graph_to_svg",
    "gantt_charts",
    "hierarchy_from_module_tree",
    "placed_hierarchy",
    "render_report",
    "routed_hierarchy",
    "single_scope",
    "write_report",
]
