"""Placement of compute operations and value movements onto hardware.

Placement answers *where* computation and boundary values are assigned. It owns
operation/resource compatibility, per-resource compute cost, and the HEFT
assignment over compute and memory resources. The result is written onto the
task graph: placement stamps each node with its chosen ``resource`` and, for
compute nodes, its ``seconds``. Route selection is owned by
:mod:`dataflow_mapper.routing`.
"""

from __future__ import annotations

import operator
from collections import defaultdict
from collections.abc import Mapping

import networkx as nx
import numpy as np
import scipy.sparse
import sympy as sp
import torch

from dataflow_mapper.heft import HEFTMapper
from dataflow_mapper.scheduling.model import TilingScope
from hw_model.model import HardwareGraph
from parabolic.dataflow.costs import (
    normalized_flop_expression,
    requirements_supported,
)

COMPUTE_KIND = "compute"
BOUNDARY_KIND = "boundary"
SOURCE_PREFIX = "source:"
SINK_PREFIX = "sink:"

# Element width in bytes for the tensor dtypes that reach the mapper.
_DTYPE_BYTES = {
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "half": 2,
    "float32": 4,
    "float": 4,
    "fp32": 4,
    "int32": 4,
    "int64": 8,
    "float64": 8,
    "fp64": 8,
}


def place_fx_graph(
    graph_module: torch.fx.GraphModule,
    hardware: HardwareGraph,
    concrete_bindings: Mapping[str, int],
    *,
    boundary_memory_resources: Mapping[str, str] | None = None,
    module_meta: Mapping[str, object] | None = None,
    is_root: bool = False,
) -> nx.MultiDiGraph:
    """Place compute and boundary tasks, writing resources onto the task graph.

    Resource columns are ordered memory-first, then compute, matching the HEFT
    matrix contract. Compute tasks get a per-resource cost from their FLOP
    requirements; ``call_module`` wrappers are aggregate tasks whose cost is the
    mapped child scope's makespan and whose ``resources`` is the child's full
    resource set. Boundary tasks cost zero and are pinned by the internal
    ``boundary_memory_resources`` map when given, else left free for HEFT.

    Args:
        graph_module: FX module carrying ``flop_req`` and shape metadata.
        hardware: validated hardware graph supplying capabilities and links.
        concrete_bindings: concrete values for the symbolic shape symbols.
        boundary_memory_resources: boundary value name -> memory pin supplied by
            the enclosing scope.
        is_root: Whether this is the outermost scope. Non-root wrapper outputs
            share their child buffer: the sink remains, but its transfer is
            suppressed and its storage is counted once.

    Returns:
        The task graph from :func:`fx_task_graph`, each node stamped with its
        chosen ``resource`` and, for compute nodes, ``seconds``.

    Example:
        Place a concretized graph, pinning its input to the scope's root memory::

            graph = place_fx_graph(
                graph_module,
                hardware,
                {"M": 32, "K": 128, "N": 128},
                boundary_memory_resources={"arg0_1": root_memory},
            )
    """
    # Build the task graph (compute + boundary tasks) from FX data dependencies.
    graph = fx_task_graph(
        graph_module,
        concrete_bindings,
        module_meta=module_meta,
        is_root=is_root,
    )
    by_name = {node.name: node for node in graph_module.graph.nodes}

    # Deterministic task/resource ordering defines the HEFT matrix rows/columns.
    task_ids = tuple(sorted(graph.nodes))
    task_index = {task: index for index, task in enumerate(task_ids)}
    memory_ids = sorted(
        rid for rid in hardware.nodes_by_id() if hardware.is_memory(rid)
    )
    compute_ids = sorted(
        rid for rid in hardware.nodes_by_id() if hardware.is_compute(rid)
    )
    # Column order is memory resources first, then compute resources.
    resource_ids = tuple(memory_ids + compute_ids)
    resource_index = {rid: index for index, rid in enumerate(resource_ids)}

    # computation[task][resource] = compute seconds; inf marks an infeasible pair.
    computation = np.full((len(task_ids), len(resource_ids)), np.inf, dtype=float)
    # Candidate resources per task, used to enumerate communication pairs.
    candidate_resources: dict[str, set[str]] = defaultdict(set)

    # Compute tasks: one cost per resource able to service the node's FLOP units.
    for task in sorted(t for t in task_ids if graph.nodes[t]["kind"] == COMPUTE_KIND):
        fx_node = by_name[task]
        # A call_module wrapper is an aggregate task: its cost is the mapped
        # child scope's makespan, and it may use several resources.
        scope = fx_node.meta.get("scope")
        if scope is not None:
            makespan, scope_resources = scope
            graph.nodes[task]["resources"] = frozenset(scope_resources)
            # Only resources the child actually used are feasible; each costs the
            # same aggregate makespan.
            for resource_id in compute_ids:
                if resource_id not in scope_resources:
                    continue
                computation[task_index[task], resource_index[resource_id]] = float(
                    makespan
                )
                candidate_resources[task].add(resource_id)
            continue
        # Ordinary node: cost is its FLOP count divided by the resource's peak.
        requirements = _fx_requirements(fx_node, concrete_bindings)
        for resource_id in compute_ids:
            capability = hardware.flop_capability(resource_id)
            # Skip resources that cannot service any required FLOP unit.
            if not requirements_supported(requirements, capability):
                continue
            flops = float(normalized_flop_expression(requirements, capability))
            computation[task_index[task], resource_index[resource_id]] = (
                flops / hardware.compute_peak(resource_id)
            )
            candidate_resources[task].add(resource_id)

    # Boundary tasks: an explicit map pins a value to one memory; otherwise the
    # value (a nested scope's own boundary) is placed on any memory by HEFT.
    for task in sorted(t for t in task_ids if graph.nodes[t]["kind"] == BOUNDARY_KIND):
        # Boundary task ids are "source:<value>"/"sink:<value>"; recover the value.
        prefix = SOURCE_PREFIX if task.startswith(SOURCE_PREFIX) else SINK_PREFIX
        value_id = task[len(prefix) :]
        resource_id = (boundary_memory_resources or {}).get(value_id)
        if resource_id is None:
            # Unpinned: any memory is feasible at zero compute cost.
            for candidate in memory_ids:
                computation[task_index[task], resource_index[candidate]] = 0.0
                candidate_resources[task].add(candidate)
            continue
        # Pinned: the named memory must exist in the hardware graph.
        if resource_id not in memory_ids:
            raise ValueError(
                f"boundary value {value_id!r} selects unknown memory {resource_id!r}"
            )
        # Boundary residency has no compute cost of its own.
        computation[task_index[task], resource_index[resource_id]] = 0.0
        candidate_resources[task].add(resource_id)

    if not candidate_resources:
        raise ValueError("FX graph has no placeable compute or boundary tasks")

    # dependencies[source][destination] = True for every value movement.
    dependencies = scipy.sparse.lil_matrix((len(task_ids), len(task_ids)), dtype=bool)
    for source, destination in graph.edges():
        dependencies[task_index[source], task_index[destination]] = True

    # communication[(src_task, dst_task, src_resource, dst_resource)] = seconds,
    # summed over every value movement and candidate (source, destination) pair.
    communication: dict[tuple[int, int, int, int], float] = defaultdict(float)
    for source, destination, data in graph.edges(data=True):
        size_bytes = data["size_bytes"]
        for src_res in sorted(candidate_resources[source]):
            for dst_res in sorted(candidate_resources[destination]):
                # Cheapest hardware path for this value between the two choices;
                # a missing path leaves that resource pair infeasible.
                path = hardware.shortest_path(src_res, dst_res, size_bytes)
                if path is None:
                    continue
                seconds = sum(hardware.link_seconds(link, size_bytes) for link in path)
                key = (
                    task_index[source],
                    task_index[destination],
                    resource_index[src_res],
                    resource_index[dst_res],
                )
                communication[key] += seconds

    # HEFT returns a (tasks x resources) start-time matrix; the single finite
    # column per row is the chosen resource.
    schedule = HEFTMapper().map(dependencies.tocsr(), computation, communication)
    for row, task in enumerate(task_ids):
        column = int(np.flatnonzero(np.isfinite(schedule[row]))[0])
        graph.nodes[task]["resource"] = resource_ids[column]
        # Stamp compute cost so routing/scheduling can read it back.
        if graph.nodes[task]["kind"] == COMPUTE_KIND:
            graph.nodes[task]["seconds"] = float(computation[row, column])
    return graph


def _fx_referenced_nodes(value: object) -> tuple[torch.fx.Node, ...]:
    """FX nodes referenced directly inside one argument value.

    An FX argument may be a bare node, a nested list/tuple/dict of nodes, or a
    mix of nodes and constants. This flattens that structure, in traversal order,
    to just the referenced nodes.

    Args:
        value: One FX ``args``/``kwargs`` entry or output argument, possibly a
            nested container of nodes and non-node values.

    Returns:
        The referenced :class:`torch.fx.Node` objects in traversal order, empty
        when the value holds only constants.

    Example:
        ``_fx_referenced_nodes((x, [y, 1]))`` returns ``(x, y)`` for nodes ``x``
        and ``y``.
    """
    found: list[torch.fx.Node] = []
    torch.fx.node.map_arg(
        value,
        lambda item: found.append(item) if isinstance(item, torch.fx.Node) else item,
    )
    return tuple(found)


def _task_scope_path(
    fx_node: torch.fx.Node,
    module_meta: Mapping[str, object] | None,
) -> tuple[TilingScope, ...]:
    """Local scope coordinate for one task within its enclosing scope.

    The enclosing module carries the axis/factor (its ``meta["tiling"]``).
    ``branch_index`` is execution-instance metadata only; it does not split the
    logical scope. A task in an untiled module has no scope. Ancestor scopes are
    not included: the mapper composes levels by recursing into each submodule,
    so this is the coordinate within the scope.
    """
    if module_meta is None:
        return ()
    return (
        TilingScope(
            str(module_meta["axis"]),
            int(module_meta["factor"]),
            0,
            bool(module_meta.get("contracting", False)),
            int(module_meta.get("split", 0)),
        ),
    )


def _transparent_sources(
    node: torch.fx.Node,
    compute_names: set[str],
    cache: dict[str, tuple[torch.fx.Node, ...]],
) -> tuple[torch.fx.Node, ...]:
    """Graph inputs a value depends on, crossing transparent (non-compute) ops.

    Mirrors the compute-producer walk: where that finds the compute tasks behind
    a view/slice/transpose, this finds the surviving placeholders and buffers, so
    a value read through a transparent op is attributed to its real input.
    """
    if node.name in cache:
        return cache[node.name]
    if node.op in {"placeholder", "get_attr"}:
        result = (node,)
    elif node.name in compute_names:
        result = ()
    else:
        ancestors = [
            item
            for argument in (*node.args, *node.kwargs.values())
            for item in _fx_referenced_nodes(argument)
        ]
        result = tuple(
            dict.fromkeys(
                source
                for ancestor in ancestors
                for source in _transparent_sources(ancestor, compute_names, cache)
            )
        )
    cache[node.name] = result
    return result


def _view_boundary(
    node: torch.fx.Node,
    compute_names: set[str],
) -> torch.fx.Node | None:
    """Return the view value used as a scope boundary, if one is present.

    A slice is a view, so it must stay visible as a boundary value instead of
    being collapsed all the way back to the base tensor. Keeping the view name
    lets placement cache the base once and let inner scopes read alias windows.

    Args:
        node: Value consumed by a compute task or passed to a child scope.
        compute_names: Node names that produce values rather than views.

    Returns:
        The nearest view node on the transparent path to an external source, or
        ``None`` when the value is a direct source or computed value.

    Example:
        A ``slice(t, ...)`` input returns the slice node, not ``t``.
    """
    view_targets = {torch.ops.aten.slice.Tensor}
    if node.name in compute_names or node.op in {"placeholder", "get_attr"}:
        return None
    if node.op == "call_function" and node.target in view_targets:
        return node
    for argument in (*node.args, *node.kwargs.values()):
        for ancestor in _fx_referenced_nodes(argument):
            view = _view_boundary(ancestor, compute_names)
            if view is not None:
                return view
    return None


def fx_task_graph(
    graph_module: torch.fx.GraphModule,
    concrete_bindings: Mapping[str, int],
    *,
    module_meta: Mapping[str, object] | None = None,
    is_root: bool = False,
) -> nx.MultiDiGraph:
    """Derive the mapper task graph from FX producer/consumer edges and outputs.

    Compute tasks are FX nodes with FLOP requirements or ``call_module`` wrappers;
    boundary tasks are ``source:``/``sink:`` nodes for graph inputs, outputs, and
    any intermediate value read across a ``call_module`` scope boundary. Each edge
    carries the moved value's ``value_id``, source/destination ports, and byte size.
    Compute tasks also carry a structural ``scope_path`` (their logical tiling
    scope chain). A branch's ``branch_index`` is copied as execution metadata,
    not as another scope coordinate.

    Args:
        graph_module: FX module carrying shape metadata.
        concrete_bindings: concrete values for the symbolic shape symbols.
        module_meta: Tiling metadata for this scope, if any.
        is_root: Whether this is the outermost scope. Non-root wrapper outputs
            share their child buffer: the sink remains, but its transfer is
            suppressed and its storage is counted once.

    Returns:
        A ``MultiDiGraph`` of compute and boundary tasks (raises ``ValueError``
        when no mapped data dependency survives).

    Example:
        Build the task graph for a concretized Linear scope::

            graph = fx_task_graph(graph_module, {"M": 32, "K": 128, "N": 128})
            graph.nodes["linear"]  # {"kind": "compute"}
    """
    nodes = tuple(graph_module.graph.nodes)
    # Compute tasks: call_functions with FLOP requirements, plus call_module
    # wrappers (whose cost is an aggregate over their nested scope).
    compute_names = {
        node.name
        for node in nodes
        if (node.op == "call_function" and _fx_requirements(node, concrete_bindings))
        or node.op == "call_module"
    }
    graph = nx.MultiDiGraph()
    nodes_by_name = {node.name: node for node in nodes}
    module_scope: tuple[TilingScope, ...] = ()
    if module_meta is not None:
        module_scope = (
            TilingScope(
                str(module_meta["axis"]),
                int(module_meta["factor"]),
                0,
                bool(module_meta.get("contracting", False)),
                int(module_meta.get("split", 0)),
            ),
        )
    for name in compute_names:
        fx_node = nodes_by_name[name]
        attributes: dict[str, object] = {"kind": COMPUTE_KIND}
        # Attach the structural scope path so scheduling need not read FX meta.
        scope_path = _task_scope_path(fx_node, module_meta)
        if scope_path:
            attributes["scope_path"] = scope_path
        branch_index = fx_node.meta.get("branch_index")
        if isinstance(branch_index, int) and not isinstance(branch_index, bool):
            attributes["branch_index"] = branch_index
        # A wrapper's true routing endpoints are its child's boundary memories,
        # not one representative compute resource.
        if fx_node.op == "call_module":
            boundaries = dict(fx_node.meta.get("parent_scope_boundaries", {}))
            attributes["scope_boundaries"] = boundaries
            attributes["parent_scope_boundaries"] = boundaries
        graph.add_node(name, **attributes)
    # Memoizes each node's nearest compute ancestors across the whole traversal.
    producer_cache: dict[str, tuple[torch.fx.Node, ...]] = {}

    def compute_producers(node: torch.fx.Node) -> tuple[torch.fx.Node, ...]:
        """Compute tasks a value ultimately depends on, crossing transparent ops."""
        if node.name in producer_cache:
            return producer_cache[node.name]
        if node.name in compute_names:
            # A compute task produces its own value.
            result = (node,)
        elif node.op in {"placeholder", "get_attr"}:
            # Graph inputs and buffers are external sources, not producers.
            result = ()
        else:
            # A non-compute op (view, getitem, ...) is transparent: walk back to
            # the compute tasks behind it, deduplicating shared ancestors.
            ancestors = [
                item
                for argument in (*node.args, *node.kwargs.values())
                for item in _fx_referenced_nodes(argument)
            ]
            result = tuple(
                dict.fromkeys(
                    producer
                    for ancestor in ancestors
                    for producer in compute_producers(ancestor)
                )
            )
        producer_cache[node.name] = result
        return result

    # Memoizes each node's nearest graph-input ancestors across the traversal.
    source_cache: dict[str, tuple[torch.fx.Node, ...]] = {}

    def external_sources(node: torch.fx.Node) -> tuple[torch.fx.Node, ...]:
        """Graph inputs a value depends on, crossing transparent ops."""
        return _transparent_sources(node, compute_names, source_cache)

    def node_inputs(node: torch.fx.Node):
        """Yield ``(port, tensor_node)`` for each tensor the node consumes."""
        # Positional args are named arg0/arg1/...; keyword args keep their name.
        ports = [(f"arg{index}", argument) for index, argument in enumerate(node.args)]
        ports += [(str(name), argument) for name, argument in node.kwargs.items()]
        for port, argument in ports:
            for input_node in _fx_referenced_nodes(argument):
                # Only shaped tensors move between tasks; scalars/constants do not.
                meta = input_node.meta
                if (
                    meta.get("output_size") is not None
                    or getattr(meta.get("val"), "shape", None) is not None
                ):
                    yield port, input_node

    # One pass over compute tasks, wiring each consumed value into the graph.
    for node in nodes:
        if node.name not in compute_names:
            continue
        # Values a call_module expects to read from its own scope boundary.
        scope_boundaries = (
            node.meta.get("scope_boundaries", {}) if node.op == "call_module" else {}
        )
        for port, input_node in node_inputs(node):
            size_bytes = _fx_value_nbytes(input_node, concrete_bindings)
            producers = compute_producers(input_node)
            if producers and input_node.name in scope_boundaries:
                # Producer lives in the parent but the consumer reads it across a
                # scope boundary: land a boundary task at the child's input memory,
                # then read it inside; the producer drains into that memory.
                port_task = f"{SOURCE_PREFIX}{input_node.name}"
                graph.add_node(port_task, kind=BOUNDARY_KIND)
                graph.add_edge(
                    port_task,
                    node.name,
                    key=f"fx:{input_node.name}:to:{node.name}:{port}:port",
                    value_id=input_node.name,
                    source_port=input_node.name,
                    destination_port=port,
                    size_bytes=0,
                )
                for producer in producers:
                    graph.add_edge(
                        producer.name,
                        port_task,
                        key=f"fx:{input_node.name}:to:port:{port}:{producer.name}",
                        value_id=input_node.name,
                        source_port=producer.name,
                        destination_port=input_node.name,
                        size_bytes=size_bytes,
                    )
            elif producers:
                # Computed within this scope: a direct producer-to-consumer edge.
                # Through a transparent multi-producer op (for example stack ->
                # sum), each producer keeps its own identity and size; the
                # transparent op itself is not a storage buffer.
                multi_value = (
                    len(producers) > 1 and input_node.name not in compute_names
                )
                for producer in producers:
                    edge_value = producer.name if multi_value else input_node.name
                    edge_size = (
                        _fx_value_nbytes(producer, concrete_bindings)
                        if multi_value
                        else size_bytes
                    )
                    graph.add_edge(
                        producer.name,
                        node.name,
                        key=f"fx:{edge_value}:to:{node.name}:{port}:{producer.name}",
                        value_id=edge_value,
                        source_port=producer.name,
                        destination_port=port,
                        size_bytes=edge_size,
                    )
            else:
                # No compute producer: the value comes from a graph input or a
                # view of one. Keep a view as its own boundary so inner scopes
                # can alias the same buffer; only a direct source is attributed
                # to the base name.
                view = _view_boundary(input_node, compute_names)
                if view is not None:
                    bases = external_sources(view)
                    attributes: dict[str, object] = {"kind": BOUNDARY_KIND}
                    if module_scope:
                        attributes["scope_path"] = module_scope
                    if len(bases) == 1:
                        attributes["alias_value"] = bases[0].name
                    source_task = f"{SOURCE_PREFIX}{view.name}"
                    graph.add_node(source_task, **attributes)
                    graph.add_edge(
                        source_task,
                        node.name,
                        key=f"fx:{view.name}:to:{node.name}:{port}",
                        value_id=view.name,
                        source_port=view.name,
                        destination_port=port,
                        size_bytes=size_bytes,
                    )
                    # A view of one external base is materialized here: the base
                    # stays at its boundary, while only the view's bytes cross
                    # into the memory that holds the slice.
                    if len(bases) == 1:
                        base = bases[0]
                        base_task = f"{SOURCE_PREFIX}{base.name}"
                        graph.add_node(base_task, kind=BOUNDARY_KIND)
                        graph.add_edge(
                            base_task,
                            source_task,
                            key=f"fx:{view.name}:from:{base.name}",
                            value_id=view.name,
                            source_port=base.name,
                            destination_port=view.name,
                            size_bytes=size_bytes,
                        )
                else:
                    for source in external_sources(input_node):
                        source_task = f"{SOURCE_PREFIX}{source.name}"
                        graph.add_node(source_task, kind=BOUNDARY_KIND)
                        graph.add_edge(
                            source_task,
                            node.name,
                            key=f"fx:{source.name}:to:{node.name}:{port}",
                            value_id=source.name,
                            source_port=source.name,
                            destination_port=port,
                            size_bytes=size_bytes,
                        )

    # Sink the graph outputs: one boundary task per contributing producer.
    output = next(node for node in nodes if node.op == "output")
    for value in _fx_referenced_nodes(output.args):
        producers = compute_producers(value)
        for producer in producers:
            # Several producers feeding one output (e.g. a concatenation): each
            # drains its own slice, so size it from the producer, not the output.
            cat_join = (
                value.op == "call_function"
                and value.target is torch.ops.aten.cat.default
            )
            partial_output = len(producers) > 1
            source_value = producer if partial_output else value
            sink_task = f"{SINK_PREFIX}{value.name}"
            sink_attrs: dict[str, object] = {"kind": BOUNDARY_KIND}
            if module_scope:
                sink_attrs["scope_path"] = module_scope
            # A non-root wrapper output shares the wrapped child's buffer. Keep
            # the sink for its scope boundary, but mark it as an alias so routing
            # does not emit a second transfer and storage counts it once.
            single_wrapper = (
                not is_root
                and value.op == "call_module"
                and len(value.meta.get("scope_outputs", ())) == 1
                and value.name in value.meta.get("scope_boundaries", {})
            )
            child_getitem = (
                not is_root
                and value.op == "call_function"
                and value.target is operator.getitem
                and bool(value.args)
                and isinstance(value.args[0], torch.fx.Node)
                and value.args[0].op == "call_module"
                and value.meta.get("scope_output") is not None
            )
            cat_join = (
                value.op == "call_function"
                and value.target is torch.ops.aten.cat.default
                and bool(producers)
            )
            if single_wrapper or child_getitem or cat_join:
                sink_attrs["alias"] = True
                if cat_join:
                    sink_attrs["alias_value"] = producers[0].name
                    sink_attrs["alias_inputs"] = tuple(
                        producer.name for producer in producers
                    )
                else:
                    sink_attrs["alias_value"] = (
                        value.meta["scope_outputs"][0]
                        if single_wrapper
                        else value.meta["scope_output"]
                    )
                    sink_attrs["alias_scope"] = (
                        value.meta.get("scope_child_path")
                        if single_wrapper
                        else value.args[0].meta.get("scope_child_path")
                    )
            graph.add_node(sink_task, **sink_attrs)
            if cat_join and sink_attrs.get("alias"):
                producer_data = graph.nodes[producer.name]
                producer_boundaries = producer_data.setdefault(
                    "parent_scope_boundaries", {}
                )
                if producer_boundaries:
                    producer_boundaries[value.name] = next(
                        iter(producer_boundaries.values())
                    )
            graph.add_edge(
                producer.name,
                sink_task,
                key=f"fx:{value.name}:to:sink:{value.name}:{producer.name}",
                value_id=value.name,
                source_port=value.name,
                destination_port=value.name,
                size_bytes=(
                    0
                    if sink_attrs.get("alias")
                    else (
                        _fx_value_nbytes(source_value, concrete_bindings)
                        if source_value.meta.get("output_size") is not None
                        or source_value.meta.get("val") is not None
                        else _fx_value_nbytes(value, concrete_bindings)
                    )
                ),
            )
    if graph.number_of_edges() == 0:
        raise ValueError("FX graph has no mapped data dependencies")
    return graph


def _fx_requirements(
    node: torch.fx.Node,
    concrete_bindings: Mapping[str, int],
) -> dict[str, object]:
    """Concrete named FLOP requirements on one annotated FX node.

    Reads the ``flop_req`` metadata triple ``(alu, exp, mac)``, substitutes the
    concrete bindings, and drops zero entries.

    Args:
        node: An annotated FX node, typically a ``call_function``.
        concrete_bindings: Concrete values for the symbolic shape symbols.

    Returns:
        Non-zero ``{unit: count}`` requirements (``alu``/``exp``/``mac``), or an
        empty dict when the node carries no ``flop_req`` triple.

    Example:
        A node annotating ``(0, 0, M * K * N)`` returns ``{"mac": M * K * N}``,
        with the symbols bound by the caller at costing time.
    """
    vector = node.meta.get("flop_req")
    if not isinstance(vector, (list, tuple)) or len(vector) != 3:
        return {}
    names = ("alu", "exp", "mac")
    requirements = {
        name: _bind_fx_expression(value, concrete_bindings)
        for name, value in zip(names, vector, strict=True)
    }
    return {name: value for name, value in requirements.items() if value != 0}


def _bind_fx_expression(value: object, bindings: Mapping[str, int]) -> object:
    """Substitute concrete bindings into a symbolic FX metadata expression.

    Accepts a plain value, a ``torch.SymInt``, or a SymPy expression whose free
    symbols are named in ``bindings``.

    Args:
        value: The metadata expression to bind to concrete values.
        bindings: Symbolic axis name to concrete extent.

    Returns:
        An ``int`` for symbolic inputs, or the original value when it carries no
        symbols.

    Raises:
        ValueError: If a symbol in the expression has no binding.

    Example:
        ``_bind_fx_expression(M * 4, {"M": 32})`` returns ``128``.
    """
    if isinstance(value, torch.SymInt):
        value = value._sympy_()
    if isinstance(value, sp.Basic):
        substitutions = {
            symbol: bindings[symbol.name]
            for symbol in value.free_symbols
            if symbol.name in bindings
        }
        result = value.subs(substitutions)
        if result.free_symbols:
            raise ValueError(
                f"missing bindings for FX metadata symbols: {result.free_symbols}"
            )
        return int(result)
    return value


def _fx_value_nbytes(node: torch.fx.Node, bindings: Mapping[str, int]) -> int:
    """Concrete byte size for an FX node value from shape and dtype metadata.

    Binds the symbolic shape to concrete dimensions and multiplies by the dtype's
    element width. When the node itself lacks dtype metadata it is inherited from
    the nearest input that has it, since concretized slice/join helpers may omit
    it.

    Args:
        node: An FX node whose value size is needed.
        bindings: Symbolic axis name to concrete extent.

    Returns:
        The value's size in bytes.

    Raises:
        ValueError: If shape metadata is missing or non-concrete, the bound shape
            has non-positive dimensions, or the dtype is unsupported.

    Example:
        An fp16 node of concrete shape ``(32, 128)`` returns ``8192``.
    """
    # Shape metadata comes from the annotation (``output_size``) or the traced
    # tensor (``val``); a single-element list wraps the real shape.
    shape = node.meta.get("output_size")
    if shape is None and node.meta.get("val") is not None:
        shape = tuple(node.meta["val"].shape)
    if (
        isinstance(shape, (list, tuple))
        and len(shape) == 1
        and isinstance(shape[0], (list, tuple))
    ):
        shape = shape[0]
    if not isinstance(shape, (list, tuple)):
        raise TypeError(f"FX value {node.name!r} has no tensor shape metadata")
    # Bind symbolic dimensions to concrete extents before multiplying.
    dimensions = [_bind_fx_expression(dim, bindings) for dim in shape]
    if any(not isinstance(dim, int) or dim < 1 for dim in dimensions):
        raise ValueError(
            f"FX value {node.name!r} has invalid concrete shape {dimensions!r}"
        )
    tensor = node.meta.get("val")
    # Prefer the declared application dtype; fall back to the traced tensor's.
    dtype = node.meta.get("application_dtype", getattr(tensor, "dtype", None))
    # Concretized helper nodes (slices/joins) may omit dtype metadata; fall back
    # to the nearest producing tensor that has it.
    if dtype is None:
        seen: set[object] = {node}
        frontier = list(node.all_input_nodes)
        while dtype is None and frontier:
            item = frontier.pop()
            if item in seen:
                continue
            seen.add(item)
            dtype = item.meta.get("application_dtype") or getattr(
                item.meta.get("val"), "dtype", None
            )
            frontier.extend(item.all_input_nodes)
    # Map the dtype to its element width in bytes.
    dtype_name = str(dtype).removeprefix("torch.").lower()
    element_bytes = _DTYPE_BYTES.get(dtype_name)
    if element_bytes is None:
        raise ValueError(f"FX value {node.name!r} has unsupported dtype {dtype!r}")
    # Total bytes = element width times the product of the concrete dimensions.
    size = element_bytes
    for dimension in dimensions:
        size *= dimension
    return size
