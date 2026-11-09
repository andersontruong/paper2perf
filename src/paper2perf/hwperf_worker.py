"""Trusted adapter to APIs inspected in the supplied hw-perf source archive."""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import sys

from .hwperf import source_metadata, validate_metrics, verify_bundled_source
from .schemas import Proposal


def jsonable(value, depth=0):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if depth > 12:
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [jsonable(v, depth + 1) for v in value]
    if is_dataclass(value):
        return {f.name: jsonable(getattr(value, f.name), depth + 1) for f in fields(value)}
    return str(value)


def build_hardware(hardware):
    import networkx as nx
    from hw_model import HardwareGraph

    graph = nx.MultiGraph()
    for memory in hardware.memories:
        if memory.capacity.value != int(memory.capacity.value):
            raise ValueError("Memory capacities must be integral bytes")
        graph.add_node(memory.id, name=memory.id, type="mem", capacity=int(memory.capacity.value))
    for compute in hardware.computes:
        # Simulator conventions: one MAC is two FLOPs; one ALU add is one FLOP.
        capability = {"mac": 2} if compute.kind == "matrix" else {"alu": 1}
        graph.add_node(compute.id, name=compute.id, type="compute",
                       compute_peak=compute.throughput.value, flop_capability=capability)
    for link in hardware.links:
        graph.add_edge(link.source, link.target, bw=link.bandwidth.value, latency=0.0)
    model = HardwareGraph.from_networkx(graph)
    model.validate()
    return model


def execute(proposal, output_dir):
    import torch
    from .torch_compat import shape_env_compatibility

    # Keep compatibility active through capture, tiling, and concretization.
    # The worker is a separate process; upstream source remains unchanged.
    with shape_env_compatibility() as compatibility:
        result = _execute(proposal, output_dir)
        result["runtime"] = {"torch_version": torch.__version__,
                             "shape_env_compatibility": compatibility}
        return result


def _execute(proposal, output_dir):
    import networkx as nx
    import sympy as sp
    from dataflow_mapper.execution import map_fx_graph, map_with_capacity_retiling
    from dataflow_mapper.viz import ABSOLUTE_TIMELINE, gantt_charts
    from dataflow_mapper.viz.gantt import _figure
    from dataflow_mapper.viz.trace import MappingTrace
    from parabolic.dataflow.frontend.pytorch import create_parabolic_parallel_model

    from .workload import gemm_class, validate_numerically

    verify_bundled_source()
    workload = proposal.workload
    numeric = validate_numerically(workload)
    hardware = build_hardware(proposal.hardware)
    symbols = {axis: sp.Symbol(axis, positive=True, integer=True) for axis in ("M", "K", "N")}
    symbolic = create_parabolic_parallel_model(
        gemm_class(), symbols, (("M", "K"),), dtype=workload.dtype
    )
    bindings = {axis: getattr(workload, axis) for axis in ("M", "N", "K")}
    # K tiling introduces ALU reduction; never fabricate a vector core to support it.
    candidates = ("M", "N", "K") if any(
        c.kind == "vector" for c in proposal.hardware.computes
    ) else ("M", "N")
    ordering = tuple(axis for axis in candidates if bindings[axis] >= 2)
    trace = MappingTrace()
    if ordering:
        tiled = symbolic.apply_tiling_plan(ordering)
        _, factors, result = map_with_capacity_retiling(
            tiled, hardware, bindings, max_attempts=8, trace=trace
        )
    else:
        factors = {}
        result = map_fx_graph(symbolic.gm, hardware, bindings, trace=trace)
    model = tiled if ordering else symbolic
    scopes, events, peak = [], [], {}
    for scope in trace.scopes:
        scopes.append({"path": str(scope.scope_path), "axis": scope.module_axis,
                       "schedule": jsonable(scope.schedule)})
        if scope.schedule and not scope.scope_path:
            peak = dict(scope.schedule.peak_memory_bytes)
        if scope.event_graph is not None:
            for key, attrs in scope.event_graph.nodes(data=True):
                events.append({"scope": str(scope.scope_path), "id": str(key),
                               "attributes": jsonable(attrs)})
    folder = Path(output_dir)
    for name, value in {
        "hardware-graph.json": jsonable(nx.node_link_data(hardware.graph)),
        "mapping-graph.json": jsonable(nx.node_link_data(result.graph)),
        "scopes.json": scopes, "events.json": events, "numerical-validation.json": numeric,
    }.items():
        (folder / name).write_text(json.dumps(value, indent=2, allow_nan=False))
    gantt_files = []
    for index, chart in enumerate(gantt_charts(model.gm, trace,
                                               timelines=(ABSOLUTE_TIMELINE,)), 1):
        slug = "".join(character if character.isalnum() else "-"
                       for character in chart.scope_id).strip("-") or "scope"
        name = f"gantt-{index:02d}-{slug}.html"
        _figure(chart).write_html(folder / name, include_plotlyjs=True, full_html=True)
        gantt_files.append(name)
    metrics = {"backend": "hw-perf", "makespan_seconds": float(result.latency),
               "gantt_charts": gantt_files,
               "resources": sorted(result.resources), "tiling": list(ordering),
               "parallelization_factors": factors, "peak_memory_bytes": peak,
               "recorded_scope_count": len(scopes), "recorded_event_node_count": len(events),
               "source": source_metadata(),
               "modeling_conventions": ["Single-device aggregate graph", "Undirected links",
                                        "Zero fixed link latency; bandwidth costs remain modeled",
                                        "No independent SM, cache-hit, occupancy, or instruction-latency model",
                                        "One MAC = two FLOPs; one ALU add = one FLOP"],
               "request_sha256": hashlib.sha256(proposal.model_dump_json().encode()).hexdigest()}
    return validate_metrics(metrics)


def main():
    request, output = map(Path, sys.argv[1:])
    proposal = Proposal.model_validate_json(request.read_text())
    result = execute(proposal, output.parent)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
