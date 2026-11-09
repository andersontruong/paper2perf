"""Render display Gantt charts as self-contained Plotly HTML fragments.

Each :class:`~dataflow_mapper.viz.schema.GanttChart` becomes one Plotly figure: a
horizontal timeline with hardware resources on the Y axis and time in seconds on
the X axis. Lanes are ordered so every link sits beside the resources it
connects. Compute and transfer events fill their resource for their duration;
memory lanes additionally carry tensor storage lifetimes sized in bytes.
Overlapping storage lifetimes on one memory are packed into staggered sub-lanes so
each can be seen, rather than piled onto a single line.

Fragments omit Plotly's JavaScript so a multi-chart report embeds the library
once. :func:`plotly_script` returns that shared ``<script>`` tag and the report
renderer injects it a single time.
"""

from __future__ import annotations

import plotly.graph_objects as go

from dataflow_mapper.viz.schema import (
    ABSOLUTE_TIMELINE,
    MEMORY_TRACK,
    GanttBar,
    GanttChart,
    GanttTrack,
)

# One fill color per event kind.
_BAR_COLORS = {
    "compute": "#0969da",
    "transfer": "#bc4c00",
    "storage": "#1a7f37",
}
_DEFAULT_BAR_COLOR = "#8250df"

# Two windows touching within this tolerance may share a sub-lane.
_LANE_EPSILON = 1e-15


def _slug(value: str) -> str:
    """Return an HTML-id-safe token derived from ``value``."""
    slug = "".join(character if character.isalnum() else "-" for character in value)
    return slug.strip("-") or "scope"


def _bar_color(bar: GanttBar) -> str:
    """Return the fill color for one bar, keyed by its event kind."""
    return _BAR_COLORS.get(bar.kind, _DEFAULT_BAR_COLOR)


def _directional_endpoints(resource_id: str) -> tuple[str, str] | None:
    """Split a directional link lane ``"a -> b"`` into its two endpoints."""
    if " -> " in resource_id:
        source, target = resource_id.split(" -> ", 1)
        if source and target:
            return source, target
    return None


def _ordered_tracks(chart: GanttChart) -> tuple[GanttTrack, ...]:
    """Order lanes so every link sits beside the resources it connects.

    Lanes are laid out by walking the hardware resource graph: each resource is
    emitted, then for each neighbor its down-link, the neighbor, and its up-link.
    A link lane is therefore vertically adjacent to both its endpoints. A
    bidirectional link becomes two lanes (the adapter splits directions); this is
    display-only and does not change what the scheduler reserves.
    """
    tracks_by_id = {track.resource_id: track for track in chart.tracks}
    adjacency: dict[str, set[str]] = {}
    for track in chart.tracks:
        endpoints = _directional_endpoints(track.resource_id)
        if endpoints is None:
            adjacency.setdefault(track.resource_id, set())
        else:
            source, target = endpoints
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)

    def degree(node: str) -> int:
        return len(adjacency.get(node, ()))

    order: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        visited.add(node)
        # Link adjacency can mention a memory that has no storage lane in this
        # scope after a child-boundary alias.
        if node in tracks_by_id:
            order.append(node)
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor in visited:
                continue
            down = f"{node} -> {neighbor}"
            if down in tracks_by_id:
                order.append(down)
            visit(neighbor)
            up = f"{neighbor} -> {node}"
            if up in tracks_by_id:
                order.append(up)

    # Start from the most peripheral memory (the DRAM side) so the flow reads
    # top-to-bottom, then sweep any remaining components.
    memories = [
        track.resource_id for track in chart.tracks if track.kind == MEMORY_TRACK
    ]
    if memories:
        visit(min(memories, key=lambda node: (degree(node), node)))
    for node in sorted(adjacency):
        visit(node)
    emitted = set(order)
    for resource_id in sorted(tracks_by_id):
        if resource_id not in emitted:
            order.append(resource_id)
    return tuple(tracks_by_id[node] for node in order)


def _pack_lanes(bars: tuple[GanttBar, ...]) -> tuple[dict[int, int], int]:
    """Greedily pack storage intervals into the fewest sub-lanes.

    Two lifetimes may share a sub-lane when they do not overlap, so a memory with
    staggered traffic stays compact while overlapping buffers each get their own
    row. Returns ``bar index -> lane`` and the lane count.
    """
    lane_finishes: list[float] = []
    lane_of: dict[int, int] = {}
    order = sorted(
        range(len(bars)), key=lambda index: (bars[index].start, bars[index].finish)
    )
    for index in order:
        bar = bars[index]
        for lane, finish in enumerate(lane_finishes):
            if bar.start >= finish - _LANE_EPSILON:
                lane_finishes[lane] = bar.finish
                lane_of[index] = lane
                break
        else:
            lane_finishes.append(bar.finish)
            lane_of[index] = len(lane_finishes) - 1
    return lane_of, len(lane_finishes)


def _track_rows(track: GanttTrack) -> tuple[dict[int, str], list[str]]:
    """Return each bar's Y label and the lane labels in display order.

    Compute and link resources are exclusive, so their bars share one row.
    Storage on a shared memory is packed into staggered sub-rows so each lifetime
    is visible instead of all overlapping on the memory's single line.
    """
    count = len(track.bars)
    if track.kind != MEMORY_TRACK or count == 0:
        return {index: track.resource_id for index in range(count)}, [track.resource_id]
    lane_of, lanes = _pack_lanes(track.bars)
    if lanes <= 1:
        return {index: track.resource_id for index in range(count)}, [track.resource_id]
    labels = [f"{track.resource_id} [{lane}]" for lane in range(lanes)]
    return {index: labels[lane_of[index]] for index in range(count)}, labels


def _hover_text(track: GanttTrack, bar: GanttBar) -> str:
    """Return the hover body for one bar."""
    body = (
        f"{bar.label or track.resource_id}<br>"
        f"kind={bar.kind}<br>"
        f"start={bar.start:.4g}s &nbsp; dur={bar.finish - bar.start:.4g}s"
    )
    if bar.reserved:
        body += f"<br>reserved on {track.resource_id}"
    size = bar.annotations.get("bytes")
    if size is not None:
        body += f"<br>bytes={size}"
    return body


def _track_trace(track: GanttTrack, rows: dict[int, str]) -> go.Bar:
    """Build the horizontal-bar trace for one resource lane's bars."""
    return go.Bar(
        orientation="h",
        y=[rows[index] for index in range(len(track.bars))],
        base=[bar.start for bar in track.bars],
        x=[bar.finish - bar.start for bar in track.bars],
        marker={
            "color": [_bar_color(bar) for bar in track.bars],
            "opacity": [0.25 if bar.reserved else 1.0 for bar in track.bars],
        },
        hovertext=[_hover_text(track, bar) for bar in track.bars],
        hovertemplate="%{hovertext}<extra></extra>",
        meta={"track_kind": track.kind},
        showlegend=False,
    )


def _figure(chart: GanttChart) -> go.Figure:
    """Build the Plotly figure for one scope's Gantt chart."""
    tracks = _ordered_tracks(chart)
    categories: list[str] = []
    figure = go.Figure()
    for track in tracks:
        if not track.bars:
            continue
        rows, labels = _track_rows(track)
        categories.extend(labels)
        figure.add_trace(_track_trace(track, rows))
    figure.update_layout(
        title=chart.title,
        barmode="overlay",
        template="plotly_white",
        showlegend=False,
        height=max(220, 34 * len(categories) + 140),
        margin={"l": 8, "r": 16, "t": 48, "b": 40},
        xaxis={"title": "time (s)", "showgrid": True, "zeroline": False},
        yaxis={
            "type": "category",
            "categoryorder": "array",
            "categoryarray": list(reversed(categories)),
            "showgrid": False,
            "title": "",
        },
    )
    if chart.timeline == ABSOLUTE_TIMELINE and chart.makespan is not None:
        figure.add_vline(
            x=chart.makespan,
            line_dash="dash",
            line_color="#59636e",
            annotation_text="makespan",
            annotation_position="top right",
        )
    return figure


def render_gantt_chart(chart: GanttChart) -> str:
    """Render one :class:`GanttChart` as an HTML fragment without Plotly's JS.

    Args:
        chart: The Gantt chart to render.

    Returns:
        A ``<div>`` plus a ``Plotly.newPlot`` script, requiring the shared
        Plotly bundle from :func:`plotly_script`.

    Example:
        >>> html = render_gantt_chart(chart)
    """
    figure = _figure(chart)
    div_id = f"gantt-{_slug(chart.scope_id)}-{chart.timeline}"
    return figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"displaylogo": False, "responsive": True},
    )


def plotly_script() -> str:
    """Return the inline Plotly bundle to include once per report page."""
    from plotly.offline import get_plotlyjs

    return f"<script>{get_plotlyjs()}</script>"


__all__ = ["plotly_script", "render_gantt_chart"]
