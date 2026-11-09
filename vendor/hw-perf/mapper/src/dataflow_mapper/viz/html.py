"""Assemble a display :class:`~dataflow_mapper.viz.schema.Report` into HTML.

The output is a single self-contained page: every scope graph is inlined as
Graphviz SVG, and a small vanilla-JavaScript controller swaps the visible scope
and maintains a breadcrumb. No external assets are required. Gantt charts are
delegated to an optional renderer so the scheduling phase can supply Plotly
figures without this module depending on Plotly. When a report carries Gantt
charts, the page splits into two independently scrolling panes: dataflow graphs
on the left and Gantt charts on the right.
"""

from __future__ import annotations

from collections.abc import Callable
from html import escape
from pathlib import Path

from dataflow_mapper.viz.graphviz import flat_graph_to_svg
from dataflow_mapper.viz.schema import GanttChart, HierarchicalGraph, Report, Section

# A Gantt chart renderer returns an HTML fragment for one chart.
GanttRenderer = Callable[[GanttChart], str]

_CSS = """
:root {
  --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --border: #d1d9e0;
  --surface: #f6f8fa; --accent: #0969da; --accent-soft: #ddf4ff;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}
/* Split layout: a fixed header over two independently scrolling panes. */
body.split { display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
.report-header { padding: 20px 24px 14px; border-bottom: 1px solid var(--border); }
.report-header h1 { margin: 0; font-size: 1.5rem; }
body.split .report-header { flex: 0 0 auto; }
.report { max-width: 1200px; margin: 0 auto; padding: 16px 24px 80px; }
body.split .report {
  display: flex; flex: 1 1 auto; min-height: 0; max-width: none; margin: 0; padding: 0;
}
.pane { padding: 16px 20px 64px; }
body.split .pane { min-width: 0; overflow-y: auto; }
body.split .pane-graphs { flex: 0 0 var(--split, 50%); }
body.split .pane-gantt { flex: 1 1 auto; background: #fcfdfe; }
body.split .pane-divider {
  flex: 0 0 6px; cursor: col-resize; background: var(--border); touch-action: none;
}
body.split .pane-divider:hover, body.split .pane-divider:active { background: var(--accent); }
body.hide-graphs .pane-graphs,
body.hide-gantt .pane-gantt { display: none; }
.pane-label {
  margin: 0 0 4px; font-size: 0.75rem; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted); font-weight: 700;
}
.group-heading {
  margin: 32px 0 8px; font-size: 1.2rem; color: var(--accent);
  border-bottom: 1px solid var(--border); padding-bottom: 6px;
}
.report-section { margin: 24px 0; }
.section-heading { margin: 0 0 4px; font-size: 1.05rem; }
.description { margin: 0 0 12px; color: var(--muted); }
.graph-viewer {
  border: 1px solid var(--border); border-radius: 8px;
  padding: 12px; background: var(--surface);
}
.graph-toolbar { display: flex; gap: 6px; margin-bottom: 8px; }
.graph-toolbar button {
  border: 1px solid var(--border); background: #ffffff; color: var(--fg);
  border-radius: 6px; padding: 3px 10px; font: inherit; font-size: 0.85rem; cursor: pointer;
}
.graph-toolbar button:hover { background: var(--accent-soft); border-color: var(--accent); }
.breadcrumb { display: flex; flex-wrap: wrap; align-items: center; gap: 2px; margin-bottom: 10px; }
.crumb {
  border: none; background: none; color: var(--accent); cursor: pointer;
  font: inherit; padding: 2px 4px; border-radius: 4px;
}
.crumb:hover { background: var(--accent-soft); }
.crumb.current { color: var(--fg); font-weight: 600; cursor: default; }
.crumb-sep { color: var(--muted); }
.graph-canvas {
  position: relative; overflow: hidden; height: 560px; max-height: 70vh;
  border: 1px solid var(--border); border-radius: 6px; background: #ffffff;
  cursor: grab; touch-action: none;
}
.graph-canvas:active { cursor: grabbing; }
.scope-panel { transform-origin: 0 0; }
.scope-panel[hidden] { display: none; }
.graph-viewer svg { max-width: 100%; height: auto; }
.graph-viewer svg a { cursor: pointer; }
.section-svg { overflow: auto; }
.section-image { max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 8px; }
.gantt-block { border: 1px solid var(--border); border-radius: 8px; padding: 8px; }
.gantt-toolbar { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.gantt-scale-label {
  display: flex; align-items: center; gap: 6px;
  font-size: 0.8rem; color: var(--muted);
}
.gantt-scale-label input[type="range"] { width: 160px; }
.gantt-scroll { overflow-x: auto; }
.gantt-view[hidden] { display: none; }
"""

_JS = """
(function () {
  var memoryVisible = true;

  function applyGanttMemory(show) {
    memoryVisible = show;
    document.querySelectorAll(".gantt-view").forEach(function (view) {
      var toggle = view.querySelector("[data-gantt-memory]");
      if (toggle) { toggle.checked = show; }
      var plot = view.querySelector(".plotly-graph-div");
      var fullCategories = view._ganttCategories;
      if (!plot || !window.Plotly || !plot.layout || !fullCategories) { return; }
      var memoryLabels = {};
      var indices = [];
      plot.data.forEach(function (trace, index) {
        if (trace.meta && trace.meta.track_kind === "memory") {
          indices.push(index);
          (trace.y || []).forEach(function (label) { memoryLabels[label] = true; });
        }
      });
      if (indices.length) {
        window.Plotly.restyle(plot, { visible: show }, indices);
      }
      window.Plotly.relayout(plot, {
        "yaxis.categoryarray": show
          ? fullCategories
          : fullCategories.filter(function (label) {
              return !memoryLabels[label];
            }),
      });
    });
  }

  function setupGanttControls(view) {
    var slider = view.querySelector("[data-gantt-scale]");
    var memoryToggle = view.querySelector("[data-gantt-memory]");
    var scroll = view.querySelector(".gantt-scroll");
    var plot = view.querySelector(".plotly-graph-div");
    if (!slider || !scroll || !plot || !window.Plotly) { return; }
    function apply() {
      if (!scroll.clientWidth) { return; }
      var scale = parseFloat(slider.value) || 1;
      var width = Math.max(1, Math.round(scroll.clientWidth * scale));
      plot.style.width = width + "px";
      window.Plotly.relayout(plot, { width: width, autosize: false });
    }
    var frame = 0;
    function schedule() {
      if (frame) { return; }
      frame = requestAnimationFrame(function () {
        frame = 0;
        apply();
      });
    }
    slider.addEventListener("input", apply);
    window.addEventListener("resize", schedule);
    if (typeof ResizeObserver !== "undefined") {
      // Track the pane's width so the chart always fits its split, whether the
      // divider is dragged, a pane is hidden, or the window resizes.
      view._ganttResize = new ResizeObserver(schedule);
      view._ganttResize.observe(scroll);
    }

    if (!memoryToggle) { return; }
    view._ganttCategories =
      (plot.layout.yaxis && plot.layout.yaxis.categoryarray) || [];
    memoryToggle.checked = memoryVisible;
    memoryToggle.addEventListener("change", function () {
      applyGanttMemory(memoryToggle.checked);
    });
  }

  function downloadPng(panel) {
    var svg = panel && panel.querySelector("svg");
    if (!svg) { return; }
    var box = svg.viewBox && svg.viewBox.baseVal;
    var width = box && box.width ? box.width : svg.getBoundingClientRect().width;
    var height = box && box.height ? box.height : svg.getBoundingClientRect().height;
    var clone = svg.cloneNode(true);
    clone.removeAttribute("style");
    clone.setAttribute("width", width);
    clone.setAttribute("height", height);
    var markup = new XMLSerializer().serializeToString(clone);
    var blob = new Blob([markup], { type: "image/svg+xml;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var name = (panel.dataset.scope || "graph").replace(/[^a-z0-9._-]+/gi, "_");
    var image = new Image();
    image.onload = function () {
      var scale = 2;
      var canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));
      var context = canvas.getContext("2d");
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      canvas.toBlob(function (png) {
        var link = document.createElement("a");
        link.href = URL.createObjectURL(png);
        link.download = name + ".png";
        link.click();
        setTimeout(function () { URL.revokeObjectURL(link.href); }, 1000);
      });
    };
    image.onerror = function () {
      URL.revokeObjectURL(url);
      var link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = name + ".svg";
      link.click();
    };
    image.src = url;
  }

  function setupGraphControls(viewer) {
    var canvas = viewer.querySelector(".graph-canvas");
    if (!canvas) { return; }
    var state = { scale: 1, x: 0, y: 0 };
    function activePanel() {
      return canvas.querySelector(".scope-panel:not([hidden])");
    }
    function apply() {
      var panel = activePanel();
      if (panel) {
        panel.style.transform =
          "translate(" + state.x + "px," + state.y + "px) scale(" + state.scale + ")";
      }
    }
    function fit() {
      var panel = activePanel();
      if (!panel) { return; }
      panel.style.transform = "none";
      var svg = panel.querySelector("svg");
      var rect = (svg || panel).getBoundingClientRect();
      var width = rect.width;
      var height = rect.height;
      if (!width || !height) { return; }
      state.scale = Math.min(
        8,
        Math.max(
          0.2,
          Math.min(1, canvas.clientWidth / width, canvas.clientHeight / height)
        )
      );
      state.x = (canvas.clientWidth - width * state.scale) / 2;
      state.y = (canvas.clientHeight - height * state.scale) / 2;
      apply();
    }
    function reset() {
      fit();
    }
    function zoom(factor) {
      state.scale = Math.min(8, Math.max(0.2, state.scale * factor));
      apply();
    }
    viewer._graphFit = fit;
    window.addEventListener("resize", fit);
    viewer.addEventListener("click", function (event) {
      var zoomButton = event.target.closest("[data-graph-zoom]");
      if (zoomButton) {
        var action = zoomButton.dataset.graphZoom;
        if (action === "in") { zoom(1.25); }
        else if (action === "out") { zoom(0.8); }
        else { reset(); }
        return;
      }
      if (event.target.closest("[data-graph-shot]")) {
        downloadPng(activePanel());
      }
    });
    canvas.addEventListener("wheel", function (event) {
      if (!event.ctrlKey && !event.metaKey) { return; }
      event.preventDefault();
      zoom(event.deltaY < 0 ? 1.1 : 1 / 1.1);
    }, { passive: false });
    var drag = null;
    canvas.addEventListener("pointerdown", function (event) {
      if (event.target.closest("a")) { return; }
      drag = { x: event.clientX, y: event.clientY, ox: state.x, oy: state.y };
      if (canvas.setPointerCapture) {
        try { canvas.setPointerCapture(event.pointerId); }
        catch (error) { /* pointer already released */ }
      }
    });
    canvas.addEventListener("pointermove", function (event) {
      if (!drag) { return; }
      state.x = drag.ox + (event.clientX - drag.x);
      state.y = drag.oy + (event.clientY - drag.y);
      apply();
    });
    canvas.addEventListener("pointerup", function () { drag = null; });
    canvas.addEventListener("pointercancel", function () { drag = null; });
  }

  function showGantt(scope) {
    var views = document.querySelectorAll(".gantt-view");
    if (!views.length) { return; }
    views.forEach(function (view) {
      view.hidden = view.dataset.ganttScope !== scope;
    });
  }

  function setupViewer(viewer) {
    var panels = {};
    viewer.querySelectorAll(".scope-panel").forEach(function (panel) {
      panels[panel.dataset.scope] = panel;
    });
    var breadcrumb = viewer.querySelector(".breadcrumb");

    function show(scope) {
      var target = panels[scope];
      if (!target) { return; }
      Object.keys(panels).forEach(function (key) {
        panels[key].hidden = key !== scope;
      });
      var chain = [];
      var current = scope;
      while (current) {
        chain.unshift(current);
        var panel = panels[current];
        current = panel && panel.dataset.parentScope ? panel.dataset.parentScope : "";
      }
      breadcrumb.innerHTML = "";
      chain.forEach(function (key, index) {
        if (index > 0) {
          var separator = document.createElement("span");
          separator.className = "crumb-sep";
          separator.textContent = " / ";
          breadcrumb.appendChild(separator);
        }
        var button = document.createElement("button");
        button.type = "button";
        button.className = "crumb" + (key === scope ? " current" : "");
        button.textContent = key;
        button.dataset.scope = key;
        breadcrumb.appendChild(button);
      });
      viewer.dataset.current = scope;
      if (viewer._graphFit) { viewer._graphFit(); }
      if (viewer.dataset.ganttDriver) { showGantt(scope); }
    }

    viewer.addEventListener("click", function (event) {
      var link = event.target.closest("a");
      if (link) {
        var href = link.getAttribute("href") || link.getAttribute("xlink:href") || "";
        if (href.charAt(0) === "#") {
          var element = document.getElementById(href.slice(1));
          if (element && element.classList.contains("scope-panel") && viewer.contains(element)) {
            event.preventDefault();
            show(element.dataset.scope);
            return;
          }
        }
      }
      var button = event.target.closest(".crumb");
      if (button && viewer.contains(button)) {
        show(button.dataset.scope);
      }
    });

    show(viewer.dataset.root);
  }

  function setupLayout() {
    var report = document.querySelector("body.split .report");
    if (!report) { return; }
    var divider = report.querySelector("[data-pane-divider]");
    if (!divider) { return; }
    var dragging = false;
    // Dragging to within this fraction of an edge hides that pane instead.
    var edge = 0.03;
    function refresh() { window.dispatchEvent(new Event("resize")); }
    function resizeTo(fraction) {
      if (fraction <= edge) {
        document.body.classList.add("hide-graphs");
        document.body.classList.remove("hide-gantt");
        report.style.setProperty("--split", "0%");
      } else if (fraction >= 1 - edge) {
        document.body.classList.add("hide-gantt");
        document.body.classList.remove("hide-graphs");
        report.style.setProperty("--split", "100%");
      } else {
        document.body.classList.remove("hide-graphs", "hide-gantt");
        report.style.setProperty("--split", fraction * 100 + "%");
      }
    }
    divider.addEventListener("pointerdown", function (event) {
      dragging = true;
      if (divider.setPointerCapture) {
        try { divider.setPointerCapture(event.pointerId); }
        catch (error) { /* pointer already released */ }
      }
      event.preventDefault();
    });
    divider.addEventListener("pointermove", function (event) {
      if (!dragging) { return; }
      var bounds = report.getBoundingClientRect();
      if (!bounds.width) { return; }
      resizeTo((event.clientX - bounds.left) / bounds.width);
    });
    function stop(event) {
      if (!dragging) { return; }
      dragging = false;
      if (divider.releasePointerCapture) {
        try { divider.releasePointerCapture(event.pointerId); }
        catch (error) { /* pointer already released */ }
      }
      refresh();
    }
    divider.addEventListener("pointerup", stop);
    divider.addEventListener("pointercancel", stop);
  }

  document.querySelectorAll(".graph-viewer").forEach(function (viewer) {
    setupGraphControls(viewer);
    setupViewer(viewer);
  });
  document.querySelectorAll(".gantt-view").forEach(setupGanttControls);
  setupLayout();
})();
"""


def _default_gantt_renderer() -> GanttRenderer | None:
    """Return the Plotly Gantt renderer when the optional module is present."""
    try:
        from dataflow_mapper.viz.gantt import render_gantt_chart
    except ImportError:
        return None
    return render_gantt_chart


def _default_gantt_script() -> str:
    """Return the shared Plotly bundle mark-up when the module is present."""
    try:
        from dataflow_mapper.viz.gantt import plotly_script
    except ImportError:
        return ""
    return plotly_script()


def _render_graph(
    hierarchical: HierarchicalGraph, index: int, *, gantt_driver: bool = False
) -> str:
    """Render one scope tree as inlined SVGs with a drill-down controller.

    When ``gantt_driver`` is set, the controller also reveals the Gantt chart
    whose scope matches the viewer's active scope.
    """
    viewer_id = f"gv{index}"
    scope_ids = list(hierarchical.scopes)
    element_ids = {
        scope: f"{viewer_id}-s{position}" for position, scope in enumerate(scope_ids)
    }

    def drill_href(child_scope: str) -> str:
        return "#" + element_ids[child_scope]

    panels: list[str] = []
    for scope in scope_ids:
        svg = flat_graph_to_svg(hierarchical.scopes[scope], drill_href=drill_href)
        parent = hierarchical.parent_of(scope) or ""
        hidden = "" if scope == hierarchical.root else " hidden"
        panels.append(
            f'<div class="scope-panel" id="{element_ids[scope]}"'
            f' data-scope="{escape(scope)}"'
            f' data-parent-scope="{escape(parent)}"'
            f' data-viewer="{viewer_id}"{hidden}>{svg}</div>'
        )
    driver = ' data-gantt-driver="1"' if gantt_driver else ""
    toolbar = (
        '<div class="graph-toolbar">'
        '<button type="button" data-graph-zoom="in" title="Zoom in">Zoom +</button>'
        '<button type="button" data-graph-zoom="out" title="Zoom out">Zoom -</button>'
        '<button type="button" data-graph-zoom="reset" title="Reset view">Reset</button>'
        '<button type="button" data-graph-shot title="Download the graph as PNG">'
        "PNG</button></div>"
    )
    return (
        f'<div class="graph-viewer" data-root="{escape(hierarchical.root)}"'
        f' data-viewer="{viewer_id}"{driver}>'
        + toolbar
        + '<nav class="breadcrumb" aria-label="scope path"></nav>'
        + '<div class="graph-canvas">'
        + "".join(panels)
        + "</div>"
        + "</div>"
    )


def _has_media(section: Section) -> bool:
    """Whether a section carries dataflow-graph media (svg, image, or graph)."""
    return (
        section.graph is not None
        or section.svg is not None
        or section.image_uri is not None
    )


def _render_media(section: Section, index: int) -> str:
    """Render a section's dataflow-graph media (image, inline svg, or graph)."""
    out: list[str] = []
    if section.image_uri is not None:
        out.append(
            f'<img class="section-image" src="{escape(section.image_uri)}"'
            f' alt="{escape(section.title)}">'
        )
    if section.svg is not None:
        out.append(f'<div class="section-svg">{section.svg}</div>')
    if section.graph is not None:
        out.append(
            _render_graph(section.graph, index, gantt_driver=bool(section.gantt))
        )
    return "".join(out)


def _render_gantt(section: Section, gantt_renderer: GanttRenderer | None) -> str:
    """Render a section's Gantt charts as scope-linked views.

    Each chart is wrapped in a ``.gantt-view`` tagged with its scope id; only the
    chart for the graph's root scope is visible up front, so the event-graph
    viewer can reveal the chart for whichever scope the user drills into.
    """
    if not section.gantt:
        return ""
    if gantt_renderer is None:
        return "<!-- gantt charts available but no gantt renderer installed -->"
    root = section.graph.root if section.graph is not None else None
    out: list[str] = []
    for chart in section.gantt:
        hidden = "" if root is None or chart.scope_id == root else " hidden"
        out.append(
            f'<div class="gantt-view" data-gantt-scope="{escape(chart.scope_id)}"{hidden}>'
            '<div class="gantt-toolbar">'
            '<label class="gantt-scale-label">width'
            '<input type="range" min="1" max="8" step="0.25" value="1"'
            " data-gantt-scale></label>"
            '<label class="gantt-scale-label">'
            '<input type="checkbox" data-gantt-memory checked> memory</label>'
            "</div>"
            '<div class="gantt-scroll">'
            f'<div class="gantt-block">{gantt_renderer(chart)}</div>'
            "</div>"
            "</div>"
        )
    return "".join(out)


def _section_heading(section: Section) -> str:
    """Render one section's heading and description."""
    level = "h3" if section.group else "h2"
    out = [f'<{level} class="section-heading">{escape(section.title)}</{level}>']
    if section.description:
        out.append(f'<p class="description">{escape(section.description)}</p>')
    return "".join(out)


def _render_section(
    section: Section, index: int, gantt_renderer: GanttRenderer | None
) -> str:
    """Render one section with both its graph media and its Gantt charts."""
    return (
        '<section class="report-section">'
        + _section_heading(section)
        + _render_media(section, index)
        + _render_gantt(section, gantt_renderer)
        + "</section>"
    )


def _render_sections(
    sections: tuple[tuple[int, Section], ...],
    *,
    gantt_renderer: GanttRenderer | None,
) -> str:
    """Render every section with its full payload (stacked layout)."""
    out: list[str] = []
    current_group: str | None = None
    for index, section in sections:
        if section.group and section.group != current_group:
            out.append(f'<h2 class="group-heading">{escape(section.group)}</h2>')
        current_group = section.group
        out.append(_render_section(section, index, gantt_renderer))
    return "".join(out)


def _render_pane(
    sections: tuple[tuple[int, Section], ...],
    *,
    side: str,
    gantt_renderer: GanttRenderer | None,
) -> str:
    """Render the sections contributing to one split pane, keeping headings.

    ``side`` is ``"graphs"`` (dataflow graphs, images, inline SVGs) or
    ``"gantt"`` (Gantt charts); sections without that payload are skipped.
    """
    out: list[str] = []
    current_group: str | None = None
    for index, section in sections:
        if side == "graphs" and not _has_media(section):
            continue
        if side == "gantt" and not section.gantt:
            continue
        if section.group and section.group != current_group:
            out.append(f'<h2 class="group-heading">{escape(section.group)}</h2>')
        current_group = section.group
        if side == "graphs":
            body = _render_media(section, index)
        else:
            body = _render_gantt(section, gantt_renderer)
        out.append(
            '<section class="report-section">'
            + _section_heading(section)
            + body
            + "</section>"
        )
    return "".join(out)


def render_report(
    report: Report,
    *,
    gantt_renderer: GanttRenderer | None = None,
) -> str:
    """Render a display report to a self-contained HTML document string.

    Args:
        report: The report to render.
        gantt_renderer: Optional callable returning the HTML for one Gantt chart.
            When ``None``, :func:`dataflow_mapper.viz.gantt.render_gantt_chart` is
            used if that module is importable; otherwise Gantt charts are skipped.

    Returns:
        A complete HTML page with inline CSS, inlined scope SVGs, the
        drill-down controller, and -- when Gantt charts are present -- the shared
        Plotly bundle inlined once.

    Example:
        >>> html = render_report(Report("Linear", (Section("Tiling", graph=tree),)))
    """
    default_gantt_renderer = _default_gantt_renderer()
    if gantt_renderer is None:
        gantt_renderer = default_gantt_renderer
    # Inline the Plotly bundle once when the default renderer will draw charts.
    head_scripts = ""
    if (
        gantt_renderer is not None
        and gantt_renderer is default_gantt_renderer
        and any(section.gantt for section in report.sections)
    ):
        head_scripts = _default_gantt_script()
    sections = tuple(enumerate(report.sections))
    # With Gantt charts present, split into two independently scrolling panes:
    # dataflow graphs on the left, Gantt charts on the right.
    split = gantt_renderer is not None and any(
        section.gantt for section in report.sections
    )
    if split:
        body_class = "split"
        content = (
            '<div class="pane pane-graphs">'
            '<p class="pane-label">Dataflow graphs</p>'
            + _render_pane(sections, side="graphs", gantt_renderer=gantt_renderer)
            + "</div>"
            '<div class="pane-divider" data-pane-divider role="separator"'
            ' aria-orientation="vertical"'
            ' title="Drag to resize; drag to an edge to hide a pane"></div>'
            '<div class="pane pane-gantt">'
            '<p class="pane-label">Gantt charts</p>'
            + _render_pane(sections, side="gantt", gantt_renderer=gantt_renderer)
            + "</div>"
        )
    else:
        body_class = "stacked"
        content = _render_sections(sections, gantt_renderer=gantt_renderer)
    title = escape(report.title)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        f"<style>{_CSS}</style>\n"
        f"{head_scripts}\n"
        "</head>\n"
        f'<body class="{body_class}">\n'
        f'<header class="report-header"><h1>{title}</h1></header>\n'
        f'<main class="report">{content}</main>\n'
        f"<script>{_JS}</script>\n"
        "</body>\n"
        "</html>\n"
    )


def write_report(
    report: Report,
    path: str | Path,
    *,
    gantt_renderer: GanttRenderer | None = None,
) -> Path:
    """Render ``report`` and write it to ``path``, creating parent directories.

    Args:
        report: The report to render.
        path: Destination file path (conventionally ``index.html``).
        gantt_renderer: Optional Gantt renderer, forwarded to :func:`render_report`.

    Returns:
        The written path.

    Example:
        >>> write_report(report, "outputs/linear/index.html")
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_report(report, gantt_renderer=gantt_renderer), encoding="utf-8"
    )
    return destination


__all__ = ["GanttRenderer", "render_report", "write_report"]
