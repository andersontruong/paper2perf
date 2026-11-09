"""Dataflow mapping results and the region-free FX mapping pipeline.

Owns ``MappingResult`` and the FX mapping entry point ``map_fx_graph``.
"""

from __future__ import annotations

import copy
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil, prod
from typing import TYPE_CHECKING

import networkx as nx
import torch

from dataflow_mapper.placement import (
    _fx_requirements,
    _fx_value_nbytes,
    _transparent_sources,
    _view_boundary,
    fx_task_graph,
    place_fx_graph,
)
from dataflow_mapper.routing import select_routes
from dataflow_mapper.scheduling.fx import schedule_fx_graph
from dataflow_mapper.scheduling.global_storage import StorageLedger
from dataflow_mapper.scheduling.model import (
    MemoryCapacityError,
    ScopeKey,
    TilingScope,
)
from hw_model.model import HardwareGraph
from parabolic.parallel_model import ParallelModel

if TYPE_CHECKING:
    from dataflow_mapper.viz.trace import MappingTrace


@dataclass(frozen=True)
class MappingResult:
    """A complete dataflow mapping: where data lives, how it moves, and when.

    ``graph`` is the placed and routed task graph (each node carries its
    ``resource``), ``latency`` is the hierarchical schedule's makespan, and
    ``resources`` is the set of hardware resources this scope reserved.
    """

    graph: nx.MultiDiGraph
    latency: float
    resources: frozenset[str] = frozenset()


def map_fx_graph(
    graph_module: torch.fx.GraphModule,
    hardware_graph: HardwareGraph,
    concrete_bindings: Mapping[str, int],
    *,
    validate_memory_capacity: bool = True,
    trace: MappingTrace | None = None,
    scope_path: ScopeKey = (),
    ledger: StorageLedger | None = None,
    _inherited_pins: Mapping[str, str] | None = None,
    promotion_policy: str = "largest_first",
) -> MappingResult:
    """Place, route, and schedule an annotated FX graph and its nested scopes.

    Args:
        graph_module: FX module whose tiled scopes have selected factors.
        hardware_graph: Validated hardware to map the tasks onto.
        concrete_bindings: Symbolic axis name to concrete extent for this scope.
        validate_memory_capacity: Forwarded to the scheduler.
        trace: Optional collector for placed/routed/scheduled artifacts per scope.
        scope_path: Ancestor tile-coordinate path for this scope (internal).
        ledger: Internal residency collector threaded through nested scopes;
            each scope composes its subtree before returning to its parent.
        _inherited_pins: Internal boundary pins inherited from an enclosing
            scope; root calls leave this unset.
        promotion_policy: Boundary-promotion policy. The default
            ``"largest_first"`` promotes the largest boundary values until the
            adjacent memory fits.

    Returns:
        The placed and routed task graph, its makespan latency, and the hardware
        resources the scope reserved.

    Example:
        >>> result = map_fx_graph(graph_module, hardware, {"M": 32, "K": 128, "N": 128})
        >>> result.latency  # seconds

    Use :func:`map_with_capacity_retiling` to choose factors for a factorless
    ``apply_tiling_plan`` template before mapping it.
    """
    hardware_graph.validate()
    if promotion_policy != "largest_first":
        raise ValueError(
            f"unknown promotion policy {promotion_policy!r}; "
            "expected 'largest_first'"
        )
    # The outermost call owns the root scope and its pinned external boundaries.
    is_root = ledger is None
    if ledger is None:
        ledger = StorageLedger()
    if is_root:
        # Transpose is a view, not modeled work. Remove it before concretization
        # and carry the source name across wrapper boundaries so root pins reach it.
        _prune_transpose_aliases(graph_module)
    # Concretize this scope's symbolic branches, then map every wrapped child
    # scope first (innermost-first); its aggregate makespan is the wrapper's cost.
    ParallelModel.concretize(graph_module, concrete_bindings)
    # This module's own tiling scope (axis/factor), set when it was fused.
    module_meta = graph_module.meta.get("tiling")
    module_axis = str(module_meta["axis"]) if module_meta is not None else None
    inherited_pins = dict(_inherited_pins or {})
    if is_root:
        # The outermost scope owns the model's external inputs and outputs. Infer
        # the root memory from topology alone: candidate memories have no direct
        # compute neighbor; if none qualify, use the memories farthest from
        # compute. Capacity and id break ties deterministically.
        hardware_nodes = hardware_graph.nodes_by_id()
        memory_ids = [
            node_id for node_id in hardware_nodes if hardware_graph.is_memory(node_id)
        ]
        compute_ids = {
            node_id for node_id in hardware_nodes if hardware_graph.is_compute(node_id)
        }
        if not memory_ids:
            raise ValueError("hardware graph has no memory resources")
        root_candidates = [
            memory_id
            for memory_id in memory_ids
            if not any(
                neighbor in compute_ids
                for neighbor in hardware_graph.graph.neighbors(memory_id)
            )
        ]
        if not root_candidates:
            # No memory sits above compute: use the one farthest from it.
            distances = nx.multi_source_dijkstra_path_length(
                hardware_graph.graph, compute_ids, weight=None
            )
            farthest = max(distances.get(memory_id, -1) for memory_id in memory_ids)
            root_candidates = [
                memory_id
                for memory_id in memory_ids
                if distances.get(memory_id, -1) == farthest
            ]
        root_memory = max(
            root_candidates,
            key=lambda memory_id: (
                hardware_graph.memory_capacity(memory_id) or 0,
                memory_id,
            ),
        )
        # Merge the inferred root pins under any inherited pins.
        root_pins = {
            task.split(":", 1)[1]: root_memory
            for task in fx_task_graph(
                graph_module,
                concrete_bindings,
                module_meta=module_meta,
                is_root=True,
            ).nodes
            if task.startswith(("source:", "sink:"))
        }
        inherited_pins = _with_boundary_pins(root_pins, inherited_pins)
    child_pins = _map_child_scopes(
        graph_module,
        hardware_graph,
        concrete_bindings,
        inherited_pins,
        validate_memory_capacity,
        module_meta,
        scope_path,
        trace,
        ledger,
        promotion_policy,
    )
    # Pin each wrapper input where its child placed it, so values stay in one memory.
    scope_memories = _with_boundary_pins(inherited_pins, child_pins)
    if is_root:
        # Root sources/sinks are the model's external boundaries. Child aliases
        # may expose the same value at a lower memory, but they must not move the
        # root buffer: the first crossing into that lower memory is a real edge.
        scope_memories = _with_boundary_pins(scope_memories, root_pins)
    # Resolve this scope's boundary footprint before placement. Child scopes
    # stay mapped; only values that would live in the adjacent memory compete
    # for its capacity. The default policy promotes the largest such values
    # until the remaining footprint fits.
    promotions: dict[str, str] = {}
    if not is_root:
        probe = fx_task_graph(
            graph_module,
            concrete_bindings,
            module_meta=module_meta,
            is_root=False,
        )
        fx_nodes = {node.name: node for node in graph_module.graph.nodes}
        boundary_sizes: dict[str, int] = {}
        for node_id, data in probe.nodes(data=True):
            is_sink = node_id.startswith("sink:")
            if node_id.startswith("source:"):
                value_id = node_id.removeprefix("source:")
            elif is_sink:
                value_id = node_id.removeprefix("sink:")
            else:
                continue
            fx_node = fx_nodes.get(value_id)
            if fx_node is None:
                continue
            # Summed outputs are already smaller than their reduction inputs;
            # promoting them would stage the result without reducing pressure.
            # Concatenated outputs are promoted at the parent scope, where the
            # joined value is the child wrapper's boundary and the branch
            # outputs stay in the lower memory.
            if is_sink:
                producer_node = fx_node
                if (
                    producer_node.op == "call_function"
                    and producer_node.target is operator.getitem
                    and producer_node.args
                    and isinstance(producer_node.args[0], torch.fx.Node)
                ):
                    producer_node = producer_node.args[0]
                if producer_node.op == "call_module":
                    child = graph_module.get_submodule(producer_node.target)
                    child_meta = child.meta.get("tiling")
                    if child_meta is not None and bool(
                        child_meta.get("contracting", False)
                    ):
                        continue
            size_bytes = _fx_value_nbytes(fx_node, concrete_bindings)
            boundary_sizes[value_id] = max(
                boundary_sizes.get(value_id, 0), size_bytes
            )
        child_memories = [
            memory
            for memory in child_pins.values()
            if hardware_graph.is_memory(memory)
        ]
        adjacent_memory: str | None = None
        if child_memories:
            memory_counts: dict[str, int] = {}
            for memory in child_memories:
                memory_counts[memory] = memory_counts.get(memory, 0) + 1
            adjacent_memory = min(
                memory_counts,
                key=lambda memory: (-memory_counts[memory], memory),
            )
        else:
            hardware_nodes = hardware_graph.nodes_by_id()
            memory_ids = [
                node_id
                for node_id in hardware_nodes
                if hardware_graph.is_memory(node_id)
            ]
            compute_ids = {
                node_id
                for node_id in hardware_nodes
                if hardware_graph.is_compute(node_id)
            }
            if memory_ids and compute_ids:
                distances = nx.multi_source_dijkstra_path_length(
                    hardware_graph.graph, compute_ids, weight=None
                )
                nearby = [
                    memory_id
                    for memory_id in memory_ids
                    if distances.get(memory_id, 0) > 0
                ]
                if nearby:
                    adjacent_memory = min(
                        nearby,
                        key=lambda memory_id: (
                            distances.get(memory_id, 0),
                            memory_id,
                        ),
                    )
        # A concatenation output stays aliased to its branch buffer in the
        # adjacent memory. It only becomes a real buffer when promoted.
        for node_id, data in probe.nodes(data=True):
            if not node_id.startswith("sink:") or not data.get("alias"):
                continue
            if data.get("alias_scope") is not None:
                continue
            value_id = node_id.removeprefix("sink:")
            if value_id in promotions:
                continue
            alias_value = data.get("alias_value")
            producer = fx_nodes.get(alias_value)
            memory = (
                producer.meta.get("parent_scope_boundaries", {}).get(alias_value)
                if producer is not None
                else None
            )
            if memory is not None:
                scope_memories[value_id] = memory
            elif adjacent_memory is not None:
                scope_memories[value_id] = adjacent_memory
        overfull: list[tuple[int, str, str]] = []
        for value_id, size_bytes in boundary_sizes.items():
            memory = scope_memories.get(value_id, adjacent_memory)
            if memory is None:
                continue
            capacity = hardware_graph.memory_capacity(memory)
            if capacity is None or size_bytes <= capacity:
                continue
            overfull.append((size_bytes, value_id, memory))
        overfull.sort(key=lambda item: (-item[0], item[1]))
        for size_bytes, value_id, memory in overfull:
            target = _promotion_target(memory, size_bytes, hardware_graph)
            if target is None or target == memory:
                continue
            promotions[value_id] = target
    # Capacity of every finite memory, shared by the remap loop and subtree
    # composition below.
    memory_capacities = {
        resource_id: hardware_graph.memory_capacity(resource_id)
        for resource_id in hardware_graph.nodes_by_id()
        if hardware_graph.is_memory(resource_id)
        and hardware_graph.memory_capacity(resource_id) is not None
    }
    # Remapping is bounded by the number of values that could be promoted. Each
    # attempt rebuilds the mapping, so intra-iteration ordering is recomputed and
    # subtree composition sees the new placement.
    remap_limit = max(1, len(graph_module.graph.nodes))
    remaps = 0
    # Retry this scope with any remaining promoted buffers. Child scopes stay
    # mapped; a promoted parent value writes into the child's boundary.
    while True:
        attempt_memories = _with_boundary_pins(scope_memories, promotions)
        # Choose a resource for every task using the aggregate child costs.
        graph = place_fx_graph(
            graph_module,
            hardware_graph,
            concrete_bindings,
            boundary_memory_resources=attempt_memories,
            module_meta=module_meta,
            is_root=is_root,
        )
        # A promoted alias becomes a real buffer. Drop its alias annotation so
        # routing inserts the transfer into/out of the child boundary, and a
        # sink alias regains its real transfer size.
        fx_nodes = {node.name: node for node in graph_module.graph.nodes}
        for node_id, data in graph.nodes(data=True):
            if node_id.startswith("source:"):
                value_id = node_id.removeprefix("source:")
            elif node_id.startswith("sink:"):
                value_id = node_id.removeprefix("sink:")
            else:
                continue
            if node_id.startswith("sink:") and data.get("alias") and not data.get(
                "alias_scope"
            ):
                data["alias_scope"] = scope_path
            if value_id not in promotions:
                continue
            if node_id.startswith("sink:"):
                if not data.get("alias"):
                    continue
                data.pop("alias", None)
                data.pop("alias_value", None)
                data.pop("alias_scope", None)
                fx_node = fx_nodes.get(value_id)
                if fx_node is None:
                    continue
                size_bytes = _fx_value_nbytes(fx_node, concrete_bindings)
                for source_task in graph.predecessors(node_id):
                    for key in graph[source_task][node_id]:
                        graph.edges[source_task, node_id, key][
                            "size_bytes"
                        ] = size_bytes
            else:
                data.pop("alias_value", None)
        if trace is not None:
            # Snapshot placement before routing annotates the same edge objects.
            trace.record_placed(scope_path, module_axis, graph.copy())
        # Connect producers to consumers through the hardware network.
        select_routes(graph, hardware_graph)
        if trace is not None:
            trace.record_routed(scope_path, module_axis, graph)
        # Build the physical event graph and time it hierarchically. Any local
        # tile capacity error is reported as one root-to-leaf path key.
        try:
            event_graph, hierarchical_schedule = schedule_fx_graph(
                graph,
                hardware_graph,
                validate_memory_capacity=validate_memory_capacity,
                trace=trace,
                scope_path=scope_path,
                module_axis=module_axis,
                ledger=ledger,
                scope_key=scope_path,
            )
            # Compose this mapping subtree before accepting the attempt. The
            # parent later composes its own windows with this child's resolved
            # windows, so cross-scope capacity is a mapping constraint rather
            # than a separate root-only phase. Windows stay keyed by owning
            # scope and aliases collapse to the physical buffer they describe.
            resolution = ledger.resolve(scope_path, memory_capacities)
            if validate_memory_capacity:
                for memory_id, peak, capacity in resolution.violations:
                    sizes: dict[str, int] = {}
                    for windows in resolution.windows_by_scope.values():
                        for value_id, window_memory, size_bytes, *_ in windows:
                            if window_memory == memory_id:
                                sizes[value_id] = max(
                                    sizes.get(value_id, 0), size_bytes
                                )
                    contributors = tuple(
                        sorted(
                            sizes.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    )
                    scope = resolution.scope_by_memory.get(memory_id, scope_path)
                    raise MemoryCapacityError(
                        f"memory {memory_id!r} capacity exceeded: {peak} > {capacity}",
                        memory_id=memory_id,
                        peak_bytes=peak,
                        capacity_bytes=capacity,
                        required_scale=peak / capacity,
                        scope_path=scope,
                        contributors=contributors,
                        kind="cross_scope",
                    )
            break
        except MemoryCapacityError as error:
            candidate = None
            if (
                not is_root
                and error.memory_id is not None
                and error.contributors
            ):
                capacity = hardware_graph.memory_capacity(error.memory_id)
                for value_id, size_bytes in error.contributors:
                    if capacity is not None and size_bytes <= capacity:
                        continue
                    target = _promotion_target(
                        error.memory_id, size_bytes, hardware_graph
                    )
                    if target is not None and promotions.get(value_id) != target:
                        candidate = (value_id, target)
                        break
            if candidate is None or remaps >= remap_limit:
                # Composition already reports a full path key. Local scheduling
                # errors are relative to this scope, so prefix the path here.
                if error.kind != "cross_scope":
                    error.scope_path = (*scope_path, *error.scope_path)
                raise
            promotions[candidate[0]] = candidate[1]
            remaps += 1
    # Report the placed graph, its makespan, and the reserved resources.
    return MappingResult(
        graph=graph,
        latency=hierarchical_schedule.makespan_seconds,
        resources=frozenset(
            data["resource_id"] for _, data in event_graph.nodes(data=True)
        ),
    )


def map_with_capacity_retiling(
    tiled_model: ParallelModel,
    hardware_graph: HardwareGraph,
    concrete_bindings: Mapping[str, int],
    *,
    min_tile_size: Mapping[str, int] | None = None,
    max_attempts: int = 8,
    trace: MappingTrace | None = None,
    promotion_policy: str = "largest_first",
) -> tuple[ParallelModel, dict[str, int], MappingResult]:
    """Map a symbolically tiled model, choosing and retuning its parfactors.

    Starts every tiled split at factor ``2``, subject to the per-axis minimum
    tile size. A split must leave room for at least two replicas whose final
    tile meets that minimum. Whenever a :class:`MemoryCapacityError` survives
    mapping remap and modulo II search, increases the failing split's factor by
    its ``required_scale`` (smaller chunks use less storage) and maps a fresh
    copy of ``tiled_model``. Stops after ``max_attempts`` or once no larger
    factor fits, re-raising as genuinely infeasible.

    Factors are shared across sibling tiling sites: every scope with the same
    ``(axis, split)`` key selects the same factor. Retuning one split grows all
    of its sibling scopes together.

    Args:
        tiled_model: Factorless symbolic tiling template from
            ``apply_tiling_plan``; each attempt maps a fresh copy, so it is
            never mutated.
        hardware_graph: Validated hardware the model is mapped onto.
        concrete_bindings: Symbolic axis name -> concrete extent.
        min_tile_size: Optional per-axis minimum tile size in extent units;
            missing axes default to ``1``.
        max_attempts: Maximum map attempts before failing.
        trace: Optional collector forwarded to every attempt's ``map_fx_graph``;
            scopes with the same key are overwritten in place.
        promotion_policy: Boundary-promotion policy. The default
            ``"largest_first"`` promotes the largest scope boundaries until the
            adjacent memory fits.

    Returns:
        The concretized model that fit, its selected ``{axis@split: factor}``
        map, and its :class:`MappingResult`.

    Raises:
        ValueError: If a tiled axis has no concrete binding, a minimum tile
            size is not positive, or a split cannot satisfy the minimum.
        MemoryCapacityError: If a violation has no tile-local scope, the failing
            split cannot grow, or ``max_attempts`` is exhausted.

    Example:
        >>> tiled = model.apply_tiling_plan(("M", "N"))
        >>> concrete, parfactors, result = map_with_capacity_retiling(
        ...     tiled, hardware, {"M": 32, "K": 128, "N": 128}
        ... )
    """
    # Normalize the hardware-provided per-axis minimum tile sizes.
    minimums = {
        str(axis): int(size) for axis, size in (min_tile_size or {}).items()
    }
    if any(size < 1 for size in minimums.values()):
        raise ValueError("minimum tile sizes must be positive")

    # Collect the split levels per axis. Sibling scopes agree on the split key,
    # so this also deduplicates modules that share a factor.
    split_levels: dict[str, set[int]] = {}
    for _, module in tiled_model.gm.named_modules():
        meta = module.meta.get("tiling")
        if meta is None:
            continue
        axis = str(meta["axis"])
        split = int(meta.get("split", 0))
        if axis not in concrete_bindings:
            raise ValueError(f"missing concrete binding for tiled axis {axis!r}")
        split_levels.setdefault(axis, set()).add(split)

    # Choose an initial factor for every split level. The template is
    # factorless; a small default exercises tiling, while capacity feedback
    # grows it. Sequential division guarantees the final tile keeps the
    # minimum size.
    current: dict[str, int] = {}
    for axis, splits in split_levels.items():
        extent = int(concrete_bindings[axis])
        minimum = minimums.get(axis, 1)
        remaining = extent
        for split in sorted(splits):
            max_factor = remaining // minimum
            if max_factor < 2:
                raise ValueError(
                    f"tiled axis {axis!r} split {split} cannot satisfy "
                    f"minimum tile size {minimum} at extent {extent}"
                )
            factor = min(2, max_factor)
            current[f"{axis}@{split}"] = factor
            remaining //= factor

    last: MemoryCapacityError | None = None
    for _ in range(max_attempts):
        # Map a fresh copy so the caller keeps its symbolic template and each
        # retry starts from the same structure with the current factors.
        attempt = copy.copy(tiled_model)
        attempt.gm = copy.deepcopy(tiled_model.gm)
        # Apply every selected factor to its split's sibling scopes; the
        # template itself stays factorless.
        for _, module in attempt.gm.named_modules():
            meta = module.meta.get("tiling")
            if meta is not None:
                axis = str(meta["axis"])
                split = int(meta.get("split", 0))
                meta["factor"] = current[f"{axis}@{split}"]
        try:
            result = map_fx_graph(
                attempt.gm,
                hardware_graph,
                concrete_bindings,
                trace=trace,
                promotion_policy=promotion_policy,
            )
        except MemoryCapacityError as error:
            last = error
            # Any error reaching this driver has exhausted its local repair:
            # structural buffers cannot shrink, intra/cross-scope remap failed,
            # and inter-iteration II growth failed. Retiling is the fallback.
            # The full path key holds the innermost tile coordinate; its split
            # identifies which shared factor must grow.
            coordinate = next(
                (
                    item
                    for item in reversed(error.scope_path)
                    if isinstance(item, TilingScope)
                ),
                None,
            )
            if error.required_scale is None or coordinate is None:
                raise
            axis = coordinate.axis
            split = coordinate.split
            key = f"{axis}@{split}"
            if axis not in split_levels or key not in current:
                raise
            splits = sorted(split_levels[axis])
            if split not in splits:
                raise
            position = splits.index(split)
            previous_product = prod(
                current[f"{axis}@{item}"] for item in splits[:position]
            )
            downstream_product = prod(
                current[f"{axis}@{item}"] for item in splits[position + 1 :]
            )
            extent = int(concrete_bindings[axis])
            minimum = minimums.get(axis, 1)
            # Preserve the minimum final tile size for this axis.
            max_factor = extent // (
                previous_product * downstream_product * minimum
            )
            if max_factor < 2:
                raise
            grown = min(max_factor, ceil(current[key] * error.required_scale))
            if grown <= current[key]:
                raise
            current[key] = grown
            continue
        return attempt, dict(sorted(current.items())), result
    assert last is not None
    raise MemoryCapacityError(
        f"re-tiling did not resolve capacity within {max_attempts} attempts: {last}",
        scope_path=last.scope_path,
        memory_id=last.memory_id,
        peak_bytes=last.peak_bytes,
        capacity_bytes=last.capacity_bytes,
        required_scale=last.required_scale,
        contributors=last.contributors,
        kind=last.kind,
    )


def _prune_transpose_aliases(graph_module: torch.fx.GraphModule) -> None:
    """Remove unsupported transpose views and canonicalize scope boundaries.

    A transpose is metadata-only for the current hardware model, so it should
    not become a task or a boundary value. Removing it is not enough on its own:
    a wrapper's child sees a new placeholder name, which would lose the parent's
    memory pin. This walks the tiled module hierarchy, removes every
    ``aten.t.default``, and renames matching child placeholders to the canonical
    parent value name. Only direct external-source arguments are renamed; view
    values keep their own boundary identity.

    Args:
        graph_module: Root of the tiled FX module hierarchy to rewrite in place.

    Raises:
        ValueError: If a transpose is malformed, a boundary rename would collide
            with an existing FX name, or a shared child would need conflicting
            canonical names.
    """
    pending: list[tuple[torch.fx.GraphModule, dict[str, str]]] = [
        (graph_module, {})
    ]
    visited: dict[int, dict[str, str]] = {}
    while pending:
        module, renames = pending.pop()
        previous = visited.get(id(module))
        if previous is not None:
            for current, canonical in renames.items():
                if previous.get(current, current) != canonical:
                    raise ValueError(
                        "shared module needs conflicting boundary names: "
                        f"{current!r} was {previous.get(current, current)!r}, "
                        f"now {canonical!r}"
                    )
            continue
        graph = module.graph
        all_names = {node.name: node for node in graph.nodes}
        for placeholder in list(graph.nodes):
            if placeholder.op != "placeholder":
                continue
            canonical = renames.get(placeholder.name)
            if canonical is None or canonical == placeholder.name:
                continue
            occupant = all_names.get(canonical)
            if occupant is not None and occupant is not placeholder:
                raise ValueError(
                    f"cannot rename {placeholder.name!r} to {canonical!r}: "
                    f"name already in use in {module!r}"
                )
            placeholder.name = canonical
            placeholder.target = canonical
            all_names[canonical] = placeholder
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target != torch.ops.aten.t.default:
                continue
            if len(node.args) != 1 or node.kwargs:
                raise ValueError("transpose node must have one positional input")
            source = node.args[0]
            if not isinstance(source, torch.fx.Node):
                raise ValueError("transpose input must be an FX node")
            node.replace_all_uses_with(source)
            graph.erase_node(node)
        visited[id(module)] = {
            node.name: node.name
            for node in graph.nodes
            if node.op == "placeholder"
        }
        call_targets = [
            node.target for node in graph.nodes if node.op == "call_module"
        ]
        for node in graph.nodes:
            if node.op != "call_module":
                continue
            child = module.get_submodule(node.target)
            child_renames: dict[str, str] = {}
            # A shared child called with different slice/value names cannot be
            # renamed to any one call's parent names. Keep its own placeholder
            # identities and let boundary aliases handle the pin.
            shared_child = call_targets.count(node.target) > 1
            if not shared_child:
                placeholders = [
                    item for item in child.graph.nodes if item.op == "placeholder"
                ]
                for index, argument in enumerate(node.args):
                    if index >= len(placeholders):
                        break
                    if (
                        isinstance(argument, torch.fx.Node)
                        and argument.op in {"placeholder", "get_attr"}
                    ):
                        child_renames[placeholders[index].name] = argument.name
                for keyword, argument in node.kwargs.items():
                    if (
                        not isinstance(argument, torch.fx.Node)
                        or argument.op not in {"placeholder", "get_attr"}
                    ):
                        continue
                    placeholder = next(
                        (
                            item
                            for item in placeholders
                            if item.target == keyword
                        ),
                        None,
                    )
                    if placeholder is not None:
                        child_renames[placeholder.name] = argument.name
            pending.append((child, child_renames))


def _map_child_scopes(
    graph_module: torch.fx.GraphModule,
    hardware_graph: HardwareGraph,
    concrete_bindings: Mapping[str, int],
    inherited_pins: Mapping[str, str],
    validate_memory_capacity: bool,
    module_meta: Mapping[str, object] | None,
    scope_path: ScopeKey,
    trace: MappingTrace | None,
    ledger: StorageLedger,
    promotion_policy: str = "largest_first",
) -> dict[str, str]:
    """Map every wrapped child scope, recording its cost and boundary pins.

    Walks the ``call_module`` wrappers of ``graph_module`` (the child scopes
    produced by tiling), maps each one recursively, and annotates the wrapper
    node with what the parent's placement and routing passes need:

    * ``node.meta["scope"]`` is ``(latency, resources)``: the child's makespan and
      the hardware resources it reserved, used to cost the wrapper as a single
      aggregate task.
    * ``node.meta["scope_boundaries"]`` maps each child boundary value name to
      the memory the child placed it on.

    Args:
        graph_module: Parent FX module whose ``call_module`` nodes wrap child
            scopes; its symbolic branches must already carry tiling markers.
        hardware_graph: Validated hardware the children are mapped onto.
        concrete_bindings: Symbolic axis name to concrete extent for the parent.
        inherited_pins: Boundary value -> memory pins inherited from the
            enclosing scope. The wrapper's own output pin, if any, is forwarded
            onto the child's outputs.
        validate_memory_capacity: Forwarded to each child's ``map_fx_graph``.
        scope_path: Ancestor tile-coordinate path of this scope.
        trace: Optional artifact collector forwarded to each child scope.
        ledger: Global storage ledger used to link each child to its wrapper.

    Returns:
        The union of the children's boundary placements as a ``value name ->
        memory name`` map. The parent uses these to pin the wrapper inputs so a
        value never straddles two memories.

    Example:
        Called once per scope by ``map_fx_graph`` before the parent is placed::

            child_pins = _map_child_scopes(
                graph_module, hardware_graph, bindings, inherited_pins, True
            )
            scope_pins = _with_boundary_pins(inherited_pins, child_pins)
    """
    # Boundary placements collected across all children, returned to the parent.
    child_pins: dict[str, str] = {}
    parent_key = scope_path
    for node in graph_module.graph.nodes:
        # Only call_module wrappers carry a nested child scope.
        if node.op != "call_module":
            continue
        submodule = graph_module.get_submodule(node.target)
        # The child scope carries its own tiling facts on the submodule.
        child_meta = submodule.meta.get("tiling")
        # Bind this split's unique parfactor so the child's symbolic shapes
        # (extent / <parfactor>) resolve to concrete sizes.
        child_bindings = dict(concrete_bindings)
        if child_meta is not None:
            # Bind this split's unique parfactor, e.g. "M_par_0" = 2.
            child_bindings[str(child_meta["parfactor"])] = int(
                child_meta["factor"]
            )
        # The child's placeholders share the parent's value names, so parent pins
        # apply directly. Its outputs are named differently, so carry boundary
        # pins from the wrapper and from any parent getitem that reads one output.
        pins: dict[str, str] = {}
        output = next(n for n in submodule.graph.nodes if n.op == "output")
        output_names: list[str] = []
        # Flatten the returned values (through tuples/lists/dicts) to names.
        torch.fx.node.map_arg(
            output.args,
            lambda item, output_names=output_names: (
                output_names.append(item.name)
                if isinstance(item, torch.fx.Node)
                else item
            ),
        )
        # Expose the child's output order to visualization and diagnostics.
        node.meta["scope_outputs"] = tuple(output_names)
        output_pin = inherited_pins.get(node.name)
        if output_pin is not None:
            # A single wrapper pin applies to every output it returns.
            pins = {name: output_pin for name in output_names}
        # Multi-output wrappers are unpacked by parent getitems; each getitem can
        # carry its own boundary pin and names the child output it reads.
        for candidate in graph_module.graph.nodes:
            if (
                candidate.op != "call_function"
                or candidate.target is not operator.getitem
                or not candidate.args
                or candidate.args[0] is not node
                or len(candidate.args) < 2
            ):
                continue
            output_index = candidate.args[1]
            if (
                isinstance(output_index, bool)
                or not isinstance(output_index, int)
                or not 0 <= output_index < len(output_names)
            ):
                continue
            candidate.meta["scope_output"] = output_names[output_index]
            pin = inherited_pins.get(candidate.name)
            if pin is not None:
                pins[output_names[output_index]] = pin
        # The logical axis scope is one coordinate regardless of which branch
        # call enters it. Branch indices stay on task metadata for diagnostics.
        child_scope_path = scope_path
        if module_meta is not None:
            enclosing = TilingScope(
                str(module_meta["axis"]),
                int(module_meta["factor"]),
                0,
                bool(module_meta.get("contracting", False)),
                int(module_meta.get("split", 0)),
            )
            child_scope_path = scope_path + (enclosing,)
        # Use the child module name, not the call-node name: multiple calls to
        # the same logical child scope must share one scope path.
        child_scope_path = (*child_scope_path, node.target)
        # Record the child scope path for output-storage aliases.
        node.meta["scope_child_path"] = child_scope_path
        # Link this wrapper to the child scope so the ledger can offset its window.
        ledger.record_child(parent_key, node.name, child_scope_path)
        # Map the child innermost-first; its makespan becomes this scope's cost.
        try:
            child = map_fx_graph(
                submodule,
                hardware_graph,
                child_bindings,
                _inherited_pins=_with_boundary_pins(inherited_pins, pins),
                validate_memory_capacity=validate_memory_capacity,
                trace=trace,
                scope_path=child_scope_path,
                ledger=ledger,
                promotion_policy=promotion_policy,
            )
        except MemoryCapacityError:
            # The child already reports its own full path key.
            raise
        # Expose the aggregate cost and resource set to the parent's placement.
        node.meta["scope"] = (child.latency, child.resources)
        # Where the child placed each of its own source/sink boundary values.
        boundaries = {
            boundary.split(":", 1)[1]: data["resource"]
            for boundary, data in child.graph.nodes(data=True)
            if boundary.startswith(("source:", "sink:"))
        }
        # A child slice is its own boundary value, but the parent still names the
        # unsliced base. Expose that base name at the view's memory so wrapper
        # inputs route to the child buffer instead of the aggregate compute core.
        child_compute_names = {
            item.name
            for item in submodule.graph.nodes
            if (
                item.op == "call_function"
                and _fx_requirements(item, child_bindings)
            )
            or item.op == "call_module"
        }
        child_source_cache: dict[str, tuple[torch.fx.Node, ...]] = {}
        for fx_node in submodule.graph.nodes:
            if fx_node.op != "call_function":
                continue
            view = _view_boundary(fx_node, child_compute_names)
            if view is not fx_node or view.name not in boundaries:
                continue
            for source in _transparent_sources(
                view, child_compute_names, child_source_cache
            ):
                boundaries.setdefault(source.name, boundaries[view.name])
        # A single-output wrapper reuses its child's output buffer. Expose that
        # memory under the wrapper's own value name so routing starts at the
        # child-boundary memory instead of the aggregate compute resource.
        if len(output_names) == 1 and output_names[0] in boundaries:
            boundaries[node.name] = boundaries[output_names[0]]
        # The parent routes the wrapper's inputs to those same memories.
        node.meta["scope_boundaries"] = boundaries
        # The parent's call args bind to the child's placeholders positionally, so
        # map the parent's own value names onto the same boundary memories. A
        # parent value may be a transparent view of a graph input, so trace it
        # back to the surviving source the parent's task graph will name.
        placeholders = [
            item.name for item in submodule.graph.nodes if item.op == "placeholder"
        ]
        parent_compute_names = {
            item.name
            for item in graph_module.graph.nodes
            if (
                item.op == "call_function" and _fx_requirements(item, concrete_bindings)
            )
            or item.op == "call_module"
        }
        source_cache: dict[str, tuple[torch.fx.Node, ...]] = {}
        parent_boundaries: dict[str, str] = {}
        for argument, placeholder in zip(node.args, placeholders):
            if not isinstance(argument, torch.fx.Node) or placeholder not in boundaries:
                continue
            view = _view_boundary(argument, parent_compute_names)
            if view is not None:
                parent_boundaries[view.name] = boundaries[placeholder]
                ledger.record_view_alias(
                    child_scope_path, placeholder, parent_key, view.name
                )
                continue
            sources = _transparent_sources(
                argument, parent_compute_names, source_cache
            )
            if len(sources) == 1:
                ledger.record_view_alias(
                    child_scope_path, placeholder, parent_key, sources[0].name
                )
            for source in sources:
                parent_boundaries[source.name] = boundaries[placeholder]
        # A single-output wrapper's own value lives in the child boundary memory;
        # the parent task graph must route from that memory, not the aggregate core.
        if len(output_names) == 1 and output_names[0] in boundaries:
            parent_boundaries[node.name] = boundaries[output_names[0]]
        # Multi-output wrappers are unpacked by parent getitems. Each getitem
        # names one child exit, so route it to that exit's boundary memory too.
        for candidate in graph_module.graph.nodes:
            if (
                candidate.op != "call_function"
                or candidate.target is not operator.getitem
                or not candidate.args
                or candidate.args[0] is not node
            ):
                continue
            output_name = candidate.meta.get("scope_output")
            if output_name in boundaries:
                parent_boundaries[candidate.name] = boundaries[output_name]
        node.meta["parent_scope_boundaries"] = parent_boundaries
        # Merge into the scope-wide pin map returned to the caller.
        child_pins.update(boundaries)
    return child_pins


def _promotion_target(
    current_memory: str,
    required_bytes: int,
    hardware_graph: HardwareGraph,
) -> str | None:
    """Return the next outer memory that can hold a promoted buffer.

    Memory levels are discovered from topology, never from ids: candidates are
    memories farther from the compute resources than ``current_memory``. The
    nearest sufficient candidate wins, with capacity and id as deterministic
    tie-breakers.

    Args:
        current_memory: Memory currently holding the overflowing value.
        required_bytes: Size of the value that must fit in the target memory.
        hardware_graph: Validated hardware topology.

    Returns:
        A memory id, or ``None`` when no outer memory can hold the value.

    Example:
        Promote an oversized L2-resident tensor to the next enclosing memory::

            target = _promotion_target("l2", tensor_bytes, hardware)
    """
    hardware_nodes = hardware_graph.nodes_by_id()
    memory_ids = [
        node_id for node_id in hardware_nodes if hardware_graph.is_memory(node_id)
    ]
    compute_ids = {
        node_id for node_id in hardware_nodes if hardware_graph.is_compute(node_id)
    }
    if not memory_ids or not compute_ids:
        return None
    distances = nx.multi_source_dijkstra_path_length(
        hardware_graph.graph, compute_ids, weight=None
    )
    current_distance = distances.get(current_memory, -1)
    candidates = [
        memory_id
        for memory_id in memory_ids
        if memory_id != current_memory
        and distances.get(memory_id, -1) > current_distance
        and (
            hardware_graph.memory_capacity(memory_id) is None
            or hardware_graph.memory_capacity(memory_id) >= required_bytes
        )
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda memory_id: (
            distances.get(memory_id, -1),
            hardware_graph.memory_capacity(memory_id) or 0,
            memory_id,
        ),
    )


def _with_boundary_pins(
    inherited_pins: Mapping[str, str] | None,
    pins: Mapping[str, str],
) -> dict[str, str]:
    """Return the boundary memory pins with ``pins`` merged over them.

    ``pins`` take precedence over any existing assignment for the same value
    name, letting a child's placement override an inherited parent pin.

    Args:
        inherited_pins: Base boundary value name -> memory pins.
        pins: Boundary ``value name -> memory name`` assignments to merge in.

    Returns:
        A new dict with the merged pins (empty when both inputs are empty).

    Example:
        Pin each child boundary where its scope placed it::

            scope_pins = _with_boundary_pins(inherited_pins, child_pins)
    """
    if not pins:
        return dict(inherited_pins or {})
    return {**(inherited_pins or {}), **pins}
