"""Render the display schema to inline Graphviz SVG.

This renderer is deliberately domain-agnostic: it consumes the display
structures in :mod:`dataflow_mapper.viz.schema` and emits SVG. A
:class:`~dataflow_mapper.viz.schema.GraphNode` whose ``child_scope`` is set
becomes a drill-down link pointing at the href supplied by ``drill_href``, so the
same renderer serves the symbolic, tiled, placed, routed, and event graphs alike.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

import pydot

from dataflow_mapper.viz.schema import FlatGraph, GraphEdge, GraphNode

# Fill color per node kind; an unlisted kind uses ``_DEFAULT_FILL``.
_FILL_BY_KIND: Mapping[str, str] = {
    "placeholder": "#dcfce7",
    "get_attr": "#fef9c3",
    "op": "#eef2ff",
    "call_function": "#eef2ff",
    "call_module": "#dbeafe",
    "boundary": "#fef3c7",
    "source": "#fef3c7",
    "sink": "#fef3c7",
    "compute": "#e0e7ff",
    "transfer": "#fae8ff",
    "memory": "#cffafe",
}
_DEFAULT_FILL = "#f3f4f6"
# Drill-down nodes override the kind fill to stand out as expandable.
_DRILL_FILL = "#bfdbfe"

# Kinds that read more like values than operations.
_OVAL_KINDS = frozenset({"placeholder", "get_attr", "source", "sink"})

_SVG_TAG = re.compile(r"<svg[^>]*>")
_SVG_WIDTH = re.compile(r'width="[^"]*"')
_SVG_HEIGHT = re.compile(r'height="[^"]*"')


def _format_annotations(annotations: Mapping[str, str]) -> str:
    """Render display annotations as one ``key: value`` line each."""
    return "\n".join(f"{key}: {value}" for key, value in annotations.items())


def _node_label(node: GraphNode) -> str:
    """Node label plus its annotation lines, or just the label when bare."""
    details = _format_annotations(node.annotations)
    return f"{node.label}\n{details}" if details else node.label


def _edge_label(edge: GraphEdge) -> str:
    """Edge label and annotations joined by newlines; empty when both absent."""
    details = _format_annotations(edge.annotations)
    return "\n".join(part for part in (edge.label, details) if part)


def _quoted(identifier: str) -> str:
    """Force-quote a Graphviz id.

    pydot treats an unquoted ``:`` as a node port and silently truncates the id
    (``source:value`` becomes ``source``), so every node id and edge endpoint
    must be quoted even when pydot's own quoting heuristic would leave it bare.
    """
    return pydot.make_quoted(identifier)


def _node(node: GraphNode, drill_href: Callable[[str], str] | None) -> pydot.Node:
    """Build the Graphviz node, adding drill-down attributes when expandable."""
    attributes: dict[str, str] = {
        "label": _node_label(node),
        "shape": "ellipse" if node.kind in _OVAL_KINDS else "box",
        "style": "filled",
        "fillcolor": _FILL_BY_KIND.get(node.kind, _DEFAULT_FILL),
        "fontname": "Helvetica",
        "fontsize": "11",
    }
    if node.child_scope is not None and drill_href is not None:
        attributes.update(
            {
                "fillcolor": _DRILL_FILL,
                "penwidth": "2",
                "tooltip": f"expand {node.child_scope}",
                "URL": drill_href(node.child_scope),
                # `class` is a keyword, so it must enter through the mapping.
                "class": "drill",
            }
        )
    return pydot.Node(_quoted(node.id), **attributes)


def _strip_envelope(svg: str) -> str:
    """Drop the XML prolog/DOCTYPE so the SVG can be inlined into HTML."""
    start = svg.find("<svg")
    end = svg.rfind("</svg>")
    if start == -1 or end == -1:
        return svg
    return svg[start : end + len("</svg>")]


def _responsive_svg(svg: str) -> str:
    """Make the root ``<svg>`` scale to its container width."""
    match = _SVG_TAG.search(svg)
    if match is None:
        return svg
    tag = _SVG_WIDTH.sub('width="100%"', match.group(0), count=1)
    tag = _SVG_HEIGHT.sub('height="100%"', tag, count=1)
    tag = tag[:-1] + ' style="max-width:100%;height:auto">'
    return svg[: match.start()] + tag + svg[match.end() :]


def flat_graph_to_svg(
    graph: FlatGraph,
    *,
    drill_href: Callable[[str], str] | None = None,
    rankdir: str = "TB",
) -> str:
    """Render one flat scope graph to an inline, responsive Graphviz SVG.

    Args:
        graph: The flat scope graph to draw.
        drill_href: Maps a node's ``child_scope`` to the href of its target
            scope (e.g. ``"#gv0-s3"``). When ``None``, drillable nodes are drawn
            as ordinary nodes.
        rankdir: Graphviz rank direction; ``"TB"`` (default) draws the dataflow
            top-to-bottom and ``"LR"`` left-to-right.

    Returns:
        A ``<svg>`` document string without the XML prolog, or ``""`` when the
        graph has no nodes.

    Example:
        Render a placed scope whose ``mm`` node expands into scope ``"M0"``::

            svg = flat_graph_to_svg(flat, drill_href=lambda scope: f"#{scope}")
    """
    if not graph.nodes:
        return ""
    dot = pydot.Dot(graph_type="digraph", rankdir=rankdir, bgcolor="transparent")
    edge_defaults = {
        "fontname": "Helvetica",
        "fontsize": "9",
        "color": "#6b7280",
        "arrowsize": "0.7",
    }
    dot.set_edge_defaults(**edge_defaults)
    for node in graph.nodes:
        dot.add_node(_node(node, drill_href))
    for edge in graph.edges:
        attributes = dict(edge_defaults)
        label = _edge_label(edge)
        if label:
            attributes["label"] = label
        dot.add_edge(
            pydot.Edge(_quoted(edge.source), _quoted(edge.target), **attributes)
        )
    svg = dot.create_svg().decode("utf-8")
    return _responsive_svg(_strip_envelope(svg))


__all__ = ["flat_graph_to_svg"]
