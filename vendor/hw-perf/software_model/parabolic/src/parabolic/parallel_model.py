"""Symbolic tiling of a ``ParallelModel``'s FX graph.

Each split is encapsulated in one ``call_module`` axis scope holding one
representative branch; :meth:`ParallelModel.concretize` then expands those
scopes into legal, fully-unrolled ATen graphs. A tiling plan is a global axis
sequence: repeated axes recurse, and sibling module sites are tiled
independently.

Example:
    >>> model = create_parabolic_parallel_model(LinearModel, symbols, shapes)
    >>> tiled = model.apply_tiling_plan(("M", "M", "N"))
    >>> ParallelModel.concretize(tiled.gm, {"M": 32, "K": 128, "N": 128})
"""

import copy
import operator
from collections.abc import Mapping, Sequence
from typing import Any

import sympy as sp
import torch
from torch.fx.passes.tools_common import stable_topological_sort
from torch.fx.passes.utils.fuser_utils import (
    erase_nodes,
    fuse_as_graphmodule,
    topo_sort,
)

from parabolic.flop_algebra import flop_requirement_vectors
from parabolic.symbolic_model import SymbolicModel

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_shape(n: torch.fx.Node) -> list[Any] | None:
    """Extract an FX node's output shape from its metadata.

    Args:
        n: The FX node to read.

    Returns:
        The output shape as a list of dimension expressions, or ``None`` when
        neither ``output_size`` nor an example ``val`` is available. A nested
        single-element ``output_size`` (multi-output op) is unwrapped.

    Example:
        >>> _get_shape(addmm_node)  # e.g. [M, N]
    """
    if "output_size" in n.meta and n.meta["output_size"] is not None:
        shape_data = n.meta["output_size"]
        if (
            isinstance(shape_data, list)
            and len(shape_data) > 0
            and isinstance(shape_data[0], (list, tuple))
        ):
            return list(shape_data[0])
        if isinstance(shape_data, (list, tuple)):
            return list(shape_data)
    if "val" in n.meta:
        return list(n.meta["val"].shape)
    return None


def _detach_submodule(module: torch.fx.GraphModule, name: str) -> None:
    """Detach a child submodule by name, across supported torch versions.

    ``GraphModule.delete_submodule`` is the public API where available; older
    versions fall back to removing the ``_modules`` entry directly. Detaching
    (rather than registering ``None``) is deliberate: a ``None`` entry still
    occupies the name, while an orphan from a previous concretization must
    disappear from ``named_modules()``/``state_dict()`` traversal entirely.

    Args:
        module: Parent module whose child should be detached.
        name: Direct child submodule name (no dotted path).

    Example:
        >>> _detach_submodule(gm, "Tiled_t_N")
    """
    delete = getattr(module, "delete_submodule", None)
    if callable(delete) and delete(name):
        return
    module._modules.pop(name, None)


class ParallelModel(SymbolicModel):
    """An annotated FX graph that supports symbolic axis tiling.

    :meth:`apply_tiling_plan` applies a global axis sequence to every
    independent module site. Repeated axes nest recursively; sibling sites are
    never merged. :meth:`concretize` (static) expands the resulting symbolic
    branches into concrete ATen graphs ready for the mapper.
    """

    def prune_sym_size_chains(self) -> torch.fx.GraphModule:
        """Remove dead shape-arithmetic chains from ``self.gm``.

        Deletes chains that start at ``aten.sym_size.int`` and continue through
        arithmetic ops, so they do not survive into the concrete graph.

        Returns:
            ``self.gm``, mutated in place.

        Example:
            >>> model.prune_sym_size_chains()
        """
        sym_size_op = torch.ops.aten.sym_size.int

        # Set to hold all nodes that belong to the symbolic size chain
        chain_nodes = set()

        # 1. Forward Pass: Identify the chains
        for node in self.gm.graph.nodes:
            if node.op == "call_function":
                if node.target == sym_size_op:
                    chain_nodes.add(node)
                else:
                    # Check if this is a built-in operator (e.g., _operator.mul)
                    mod_name = getattr(node.target, "__module__", "")
                    is_operator = mod_name in ("_operator", "operator", "builtins")

                    if is_operator:
                        # Check if any input to this operator is already part of the chain
                        depends_on_chain = False

                        def check_arg(a):
                            nonlocal depends_on_chain
                            if isinstance(a, torch.fx.Node) and a in chain_nodes:
                                depends_on_chain = True

                        # Map over args and kwargs to catch nested node references
                        torch.fx.node.map_arg(node.args, check_arg)
                        torch.fx.node.map_arg(node.kwargs, check_arg)

                        if depends_on_chain:
                            chain_nodes.add(node)

        # 2. Reverse Pass: Safely disconnect and erase the nodes
        for node in reversed(list(self.gm.graph.nodes)):
            if node in chain_nodes:
                # Check what is using this node before deleting
                users = list(node.users.keys())
                for user in users:
                    if user.op == "output":
                        # If the shape calculation leaked into the outputs, remove it
                        old_output_tuple = user.args[0]
                        if isinstance(old_output_tuple, (tuple, list)):
                            new_output_tuple = tuple(arg for arg in old_output_tuple if arg != node)
                            user.args = (new_output_tuple,)
                    else:
                        # WARNING: If a real ATen op (like aten.view) depends on this shape
                        # calculation, pruning it will replace the dimension with None.
                        # If this happens intentionally, you might need to handle it.
                        pass

                # Replace remaining uses with None so FX allows deletion
                node.replace_all_uses_with(None)
                self.gm.graph.erase_node(node)

        # 3. Clean up any remaining dead code triggered by the severed connections
        self.gm.graph.eliminate_dead_code()
        return self.gm


    def propagate_attr_metadata_pass(self) -> torch.fx.GraphModule:
        """Fill in ``output_size`` for ``get_attr`` nodes from their consumers.

        A ``get_attr`` node usually has no shape metadata of its own; its first
        user's ``input_size`` (indexed by argument position) supplies it.

        Returns:
            ``self.gm``, mutated in place.

        Example:
            >>> model.propagate_attr_metadata_pass()
        """
        for node in self.gm.graph.nodes:
            # We only care about attribute lookups
            if node.op == "get_attr":

                # Look at all nodes that use this attribute
                for user in node.users:
                    # Check if the user has input_size metadata
                    if "input_size" in user.meta and user.meta["input_size"] is not None:

                        # user.meta['input_size'] is usually a list of shapes,
                        # one for each node.arg. We need to find which index this node is.
                        try:
                            # Find which argument index our get_attr node occupies
                            arg_index = -1
                            for i, arg in enumerate(user.args):
                                if arg == node:
                                    arg_index = i
                                    break

                            if arg_index != -1:
                                input_sizes = user.meta["input_size"]

                                # Safety check: ensure the list is long enough
                                if isinstance(input_sizes, list) and arg_index < len(input_sizes):
                                    attr_shape = input_sizes[arg_index]

                                    # Set the metadata on the get_attr node
                                    # We wrap it in a list to match your [[B, S, D]] convention
                                    node.meta["output_size"] = [list(attr_shape)]

                                    # Once we've found a valid shape from one user,
                                    # we can move to the next get_attr node.
                                    break

                        except Exception:  # noqa: BLE001, S112 - best-effort shape probe
                            continue

        return self.gm

    # -------------------------------------------------------------------
    # Region finding
    # -------------------------------------------------------------------

    @staticmethod
    def region_for_axis(
        gm: torch.fx.GraphModule,
        axis: str,
        scope: Sequence[str] | None = None,
    ) -> tuple[tuple[torch.fx.Node, ...], tuple[torch.fx.Node, ...], tuple[torch.fx.Node, ...]]:
        """Axis-carrying region within ``scope`` and its boundary.

        The region is the scoped nodes whose output carries ``axis`` plus the
        scoped matmuls that reduce it. Entries are the axis-carrying producers
        feeding the region from outside it (the parent's slices when tiling
        inside a wrapper), or the axis roots when there is no scope. Exits are
        region nodes whose axis-carrying consumers lie outside the region.

        Args:
            gm: The graph module to inspect.
            axis: Symbolic axis name to tile (e.g. ``"M"``).
            scope: Optional enclosing branch-body node names; when given, only
                nodes in that scope are considered.

        Returns:
            ``(entries, exits, region)``, each a tuple of FX nodes. ``region``
            is in graph order; ``entries``/``exits`` are its boundary.

        Example:
            >>> entries, exits, region = ParallelModel.region_for_axis(gm, "M")
        """
        # Matrix multiplies are (currently) the only ops that *reduce* an axis rather than
        # propagate it, so they are handled specially below.
        # TODO(reductions): only matmul contractions are recognized. To support
        # sum/mean/prod/amax examples, register those reducing ops here and give
        # each the right combine; and exclude atomic reductions (softmax,
        # native_layer_norm) from tiling their reduced axis. See the `sum` join
        # in insert_symbolic_branch, which currently assumes additive partials.
        matmul_ops = {torch.ops.aten.mm, torch.ops.aten.bmm, torch.ops.aten.addmm}
        # When ``scope`` is given we only look at nodes inside that enclosing
        # branch body (the parent wrapper's nodes); otherwise the whole graph.
        scope_set = set(scope) if scope is not None else None

        def carries(node: torch.fx.Node) -> bool:
            """Whether any dimension of the node's output mentions the axis."""
            shape = _get_shape(node) or ()
            return any(
                axis in {symbol.name for symbol in getattr(extent, "free_symbols", ())}
                for extent in shape
            )

        def in_scope(node: torch.fx.Node) -> bool:
            """Whether the node belongs to the current enclosing scope."""
            return scope_set is None or node.name in scope_set

        # The region is every in-scope node whose output carries the axis, plus
        # the in-scope matmuls that reduce it (they do not carry the axis on
        # their output, but one of their inputs does).
        region = [
            node
            for node in gm.graph.nodes
            if in_scope(node)
            and (
                carries(node)
                or (
                    getattr(node.target, "_overloadpacket", None) in matmul_ops
                    and any(carries(predecessor) for predecessor in node.all_input_nodes)
                )
            )
        ]
        # Membership only; `region` keeps graph (topological) order for cloning.
        region_set = set(region)
        # Entries are the values the region consumes from outside it: the
        # axis-carrying producers not already in the region. When unscoped, axis
        # roots (no axis-carrying input) are also entries, so an axis introduced
        # by a placeholder/get_attr still has something to slice.
        entries: list[torch.fx.Node] = []
        for node in gm.graph.nodes:
            if node in region_set:
                for predecessor in node.all_input_nodes:
                    if (
                        carries(predecessor)
                        and predecessor not in region_set
                        and predecessor not in entries
                    ):
                        entries.append(predecessor)
            if (
                scope is None
                and carries(node)
                and node in region_set
                and not any(carries(predecessor) for predecessor in node.all_input_nodes)
                and node not in entries
            ):
                entries.append(node)
        # A contraction axis joins at the matmul that reduces it; otherwise the
        # join is at the region nodes whose axis-carrying consumers leave the region.
        contractions = [
            node for node in gm.graph.nodes if node in region_set and not carries(node)
        ]
        if contractions:
            # Contracting matmuls both consume the region and produce its result.
            exits = contractions
        else:
            # No reduction: exits are the region nodes whose axis-carrying
            # consumers sit outside the region.
            exits = [
                node
                for node in gm.graph.nodes
                if node in region_set
                and not any(carries(user) and user in region_set for user in node.users)
            ]
        return tuple(entries), tuple(exits), tuple(region)

    def apply_tiling_plan(
        self,
        axis_order: Sequence[str],
    ) -> "ParallelModel":
        """Return a copied model with the global tiling sequence applied.

        The sequence may repeat an axis; each occurrence is one recursive split.
        Every eligible plain module is an independent tiling site. A site is
        eligible when an axis-carrying region ends at that module boundary and
        does not cross a nested ``call_module``. Sibling sites are never merged.

        For each occurrence, every active site that carries the axis receives a
        single axis scope containing one representative slice branch and the
        join. The site then continues from that scope's plain child body, so a
        repeated axis nests inside the previous split. Sites that do not carry
        the axis are skipped. Forwarding and intermediate child bodies are then
        inlined; the innermost leaf body remains as its own leaf scope.

        Args:
            axis_order: Global axis sequence. Repeated axes recurse; the index of
                each occurrence identifies its split.

        Steps:

        1. Work on a deep copy of the model.
        2. Find the independent plain-module tiling sites.
        3. For each global axis occurrence, tile every active site that carries
           the axis.
        4. Record ``axis``/``split``/``parfactor``/``contracting``/``symbolic``
           on each tiled scope.
        5. Inline forwarding/intermediate bodies; keep the innermost leaf body
           as a leaf scope.
        6. Leave each enclosing graph shape intact: the region becomes one
           ``call_module`` to the axis scope.

        The axis scope is a one-replica template; a factor is selected before
        concretization.

        Returns:
            A new ``ParallelModel`` with one scope module per axis occurrence;
            ``self`` is left unchanged.

        Example:
            >>> tiled = model.apply_tiling_plan(("M", "M", "N"))
        """
        # Preserve the caller's sequence; repeated axes are intentional splits.
        order = tuple(axis_order)

        # Rewrite a copy of the model so the source graph is left untouched.
        transformed = copy.copy(self)
        # Deep-copy the FX graph so every edit below stays local to the copy.
        transformed.gm = copy.deepcopy(self.gm)
        if not order:
            return transformed

        # Discover independent sites before any rewriting. A module is a site
        # when at least one requested axis forms a region entirely inside it.
        # Modules whose region crosses a call_module stay out; their children
        # are considered as separate sites instead.
        axes = tuple(dict.fromkeys(order))
        # Each site tracks the module being tiled, its enclosing tiled scope,
        # and the body call that owns it. The latter two are None for a site
        # that has not been tiled yet.
        active_sites: list[
            tuple[
                torch.fx.GraphModule,
                torch.fx.GraphModule | None,
                torch.fx.Node | None,
            ]
        ] = []
        for _, module in transformed.gm.named_modules():
            if "tiling" in module.meta:
                continue
            for axis in axes:
                entries, exits, closure = transformed.region_for_axis(module, axis)
                if (
                    entries
                    and exits
                    and not any(node.op == "call_module" for node in closure)
                ):
                    active_sites.append((module, None, None))
                    break

        # Apply the global sequence to each site independently. Repeated axes
        # continue from the body created by the previous occurrence.
        for split, axis in enumerate(order):
            parfactor = sp.Symbol(
                f"{axis}_par_{split}", positive=True, integer=True
            )
            tiled_any = False
            for index, (module, parent_tiled, parent_body_call) in enumerate(
                active_sites
            ):
                entries, exits, closure = transformed.region_for_axis(module, axis)
                # This site does not carry the axis; leave it for later axes.
                if not entries:
                    continue
                # A region crossing a module boundary is tiled in its children,
                # never by merging those children into this site.
                if any(node.op == "call_module" for node in closure):
                    continue
                # An axis with no boundary value has nothing to join.
                if not exits:
                    raise ValueError(
                        f"subgraph has no boundary carrying axis {axis!r}"
                    )

                tiled = ParallelModel.insert_symbolic_branch(
                    module, axis, split, parfactor
                )
                # The tiled scope has exactly one representative body call.
                body_call = next(
                    node
                    for node in tiled.graph.nodes
                    if node.op == "call_module"
                )
                body = tiled.get_submodule(body_call.target)

                active_sites[index] = (body, tiled, body_call)
                tiled_any = True
            # No site carried this occurrence: the plan does not apply.
            if not tiled_any:
                raise ValueError(f"no module carries axis {axis!r}")

        # Inline every remaining plain child module into its parent scope.
        # This removes both forwarding bodies left by earlier axes and the final
        # leaf body, leaving exactly one scope module per axis occurrence.
        changed = True
        while changed:
            changed = False
            for parent in list(transformed.gm.modules()):
                for child_call in list(parent.graph.nodes):
                    if child_call.op != "call_module":
                        continue
                    target_name = child_call.target
                    if not isinstance(target_name, str):
                        continue
                    child = parent.get_submodule(target_name)
                    if child.meta.get("tiling") is not None:
                        continue
                    # Preserve the innermost leaf as its own scope. Intermediate
                    # plain bodies contain the next axis scope and are inlined.
                    if parent.meta.get("tiling") is not None and not any(
                        node.op == "call_module"
                        and child.get_submodule(node.target).meta.get("tiling")
                        is not None
                        for node in child.graph.nodes
                    ):
                        continue
                    child_output = next(
                        node for node in child.graph.nodes if node.op == "output"
                    )
                    child_values = child_output.args[0]
                    if not isinstance(child_values, (tuple, list)):
                        child_values = (child_values,)
                    # Move nested child scopes and attributes up before copying.
                    # Two sibling sites can own same-named scopes; keep moved
                    # submodules collision-free and rewrite their call targets.
                    target_renames: dict[str, str] = {}
                    for name, submodule in list(child.named_children()):
                        target = name
                        suffix = 1
                        while (
                            target in parent._modules
                            and parent.get_submodule(target) is not submodule
                        ):
                            target = f"{name}_{suffix}"
                            suffix += 1
                        target_renames[name] = target
                        if target not in parent._modules:
                            parent.add_submodule(target, submodule)
                    for child_node in child.graph.nodes:
                        if child_node.op != "get_attr":
                            continue
                        name = child_node.target
                        if isinstance(name, str) and hasattr(child, name):
                            setattr(parent, name, getattr(child, name))
                    value_map: dict[torch.fx.Node, torch.fx.Node] = {}
                    placeholders = [
                        node for node in child.graph.nodes if node.op == "placeholder"
                    ]
                    for placeholder, argument in zip(placeholders, child_call.args):
                        if isinstance(argument, torch.fx.Node):
                            value_map[placeholder] = argument
                    insertion = child_call
                    for child_node in child.graph.nodes:
                        if child_node.op in {"placeholder", "output"}:
                            continue
                        with parent.graph.inserting_after(insertion):
                            if child_node.op == "call_module":
                                copied = parent.graph.call_module(
                                    target_renames.get(
                                        child_node.target, child_node.target
                                    ),
                                    args=torch.fx.node.map_arg(
                                        child_node.args,
                                        lambda value: value_map.get(value, value),
                                    ),
                                    kwargs=torch.fx.node.map_arg(
                                        child_node.kwargs,
                                        lambda value: value_map.get(value, value),
                                    ),
                                )
                            else:
                                copied = parent.graph.node_copy(
                                    child_node,
                                    lambda value: value_map.get(value, value),
                                )
                        copied.meta = copy.copy(child_node.meta)
                        value_map[child_node] = copied
                        insertion = copied
                    if len(child_values) == 1:
                        replacement = value_map[child_values[0]]
                        child_call.replace_all_uses_with(replacement)
                    else:
                        for user in list(child_call.users):
                            if (
                                user.op != "call_function"
                                or user.target is not operator.getitem
                                or len(user.args) < 2
                                or user.args[0] is not child_call
                            ):
                                raise ValueError(
                                    "multi-output leaf body is used outside getitem"
                                )
                            index = user.args[1]
                            if isinstance(index, bool) or not isinstance(index, int):
                                raise ValueError(
                                    "multi-output leaf body index is invalid"
                                )
                            replacement = value_map[child_values[index]]
                            user.replace_all_uses_with(replacement)
                            parent.graph.erase_node(user)
                    parent.graph.erase_node(child_call)
                    _detach_submodule(parent, target_name)
                    stable_topological_sort(parent)
                    parent.graph.eliminate_dead_code()
                    parent.graph.lint()
                    changed = True
                    break
                if changed:
                    break

        # Return the transformed copy; the original model is unchanged.
        return transformed

    @staticmethod
    def insert_symbolic_branch(
        gm: torch.fx.GraphModule,
        axis: str,
        split: int,
        parfactor: sp.Symbol,
    ) -> torch.fx.GraphModule:
        """Insert one symbolic axis scope into ``gm``.

        The scope owns one representative slice branch and the join for the
        axis. Its child is a plain fused body; later plan entries tile that body
        and collapse forwarding wrappers. The final leaf body is inlined by
        :meth:`apply_tiling_plan`, so the innermost axis scope owns its leaf
        computation directly.

        Args:
            gm: Graph module to rewrite in place.
            axis: Symbolic axis name to split.
            split: Global occurrence index of this split.
            parfactor: Unique symbolic factor for the chunk size.

        Returns:
            The axis scope module, for inspection.

        Example:
            >>> scope = ParallelModel.insert_symbolic_branch(gm, "M", 0, sym)
        """
        # Boundary and interior of the region to tile (see region_for_axis).
        entries, exits, closure = ParallelModel.region_for_axis(gm, axis)
        if not exits:
            raise ValueError(f"subgraph has no boundary carrying axis {axis!r}")
        entries_set = set(entries)
        closure_set = set(closure)
        # The body is the region minus its entry values. It is the only part
        # that moves into the shared plain body module.
        body_nodes = topo_sort(
            [node for node in closure if node not in entries_set]
        )
        if not body_nodes:
            raise ValueError(f"subgraph has no body for axis {axis!r}")
        # A region node whose output no longer carries the axis is a contraction.
        contracting = tuple(
            node.name
            for node in closure
            if not any(
                axis in {symbol.name for symbol in getattr(extent, "free_symbols", ())}
                for extent in (_get_shape(node) or ())
            )
        )
        kind = "sum" if contracting else "cat"

        # Stable, collision-free name for this axis scope in its parent graph.
        base_name = f"Tiled_{entries[0].name}_{axis}"
        module_name = base_name
        suffix = 1
        while module_name in gm._modules:
            module_name = f"{base_name}_{suffix}"
            suffix += 1
        body_name = f"{module_name}_body"

        # Fuse only the body. Inputs remain external values; the axis scope
        # slices the entries and forwards everything else unchanged.
        body_gm, body_inputs, body_outputs = fuse_as_graphmodule(
            gm, body_nodes, body_name
        )
        body_gm.meta.pop("tiling", None)
        if len(body_outputs) != len(exits):
            raise ValueError("body fusion exposed an unexpected output count")

        # The body describes one chunk, so replace the tiled axis extent
        # (``K -> K / K_par``) in every shape and FLOP requirement. Ancestor
        # axes were substituted when their own body was built; replacing only
        # ``axis`` here preserves that nesting.
        for node in body_gm.graph.nodes:
            for key in ("output_size", "input_size", "flop_req"):
                value = node.meta.get(key)
                if not isinstance(value, list):
                    continue
                # Fusion copies metadata shallowly; detach this value before
                # rewriting so ancestor/original nodes keep their full shapes.
                value = copy.deepcopy(value)
                node.meta[key] = value
                for index, item in enumerate(value):
                    if isinstance(item, (list, tuple)):
                        if isinstance(item, tuple):
                            item = list(item)
                            value[index] = item
                        for dim, extent in enumerate(item):
                            if not isinstance(extent, sp.Basic):
                                continue
                            item[dim] = extent.xreplace(
                                {
                                    symbol: symbol / parfactor
                                    for symbol in extent.free_symbols
                                    if getattr(symbol, "name", None) == axis
                                }
                            )
                    elif isinstance(item, sp.Basic):
                        value[index] = item.xreplace(
                            {
                                symbol: symbol / parfactor
                                for symbol in item.free_symbols
                                if getattr(symbol, "name", None) == axis
                            }
                        )

        body_output = next(
            node for node in body_gm.graph.nodes if node.op == "output"
        )
        body_output_values = body_output.args[0]
        if not isinstance(body_output_values, (tuple, list)):
            body_output_values = (body_output_values,)
        if len(body_output_values) != len(body_outputs):
            raise ValueError("body fusion exposed an unexpected output count")

        # ---- axis scope: representative slices + body call + join ----------
        scope_graph = torch.fx.Graph()
        scope_gm = torch.fx.GraphModule({}, scope_graph)
        scope_inputs: dict[torch.fx.Node, torch.fx.Node] = {}
        for original in body_inputs:
            placeholder = scope_graph.placeholder(original.name)
            placeholder.meta = copy.copy(original.meta)
            scope_inputs[original] = placeholder

        body_args = []
        for original in body_inputs:
            if original not in entries_set:
                body_args.append(scope_inputs[original])
                continue
            shape = list(_get_shape(original) or [])
            entry_dim = next(
                (
                    d
                    for d, extent in enumerate(shape)
                    if axis in {s.name for s in getattr(extent, "free_symbols", ())}
                ),
                -1,
            )
            if entry_dim < 0:
                body_args.append(scope_inputs[original])
                continue
            extent = shape[entry_dim]
            if isinstance(extent, sp.Basic):
                chunk = extent / parfactor
            elif isinstance(extent, int):
                chunk = sp.Integer(extent) / parfactor
            else:
                chunk = extent
            slice_node = scope_graph.call_function(
                torch.ops.aten.slice.Tensor,
                args=(scope_inputs[original], entry_dim, 0, chunk, 1),
            )
            slice_node.meta["output_size"] = [
                *shape[:entry_dim],
                chunk,
                *shape[entry_dim + 1 :],
            ]
            slice_node.meta["application_dtype"] = original.meta.get(
                "application_dtype"
            )
            body_args.append(slice_node)

        scope_gm.add_submodule(body_name, body_gm)
        body_call = scope_graph.call_module(body_name, args=tuple(body_args))
        body_call.meta["application_dtype"] = next(
            (
                node.meta.get("application_dtype")
                for node in body_outputs
                if node.meta.get("application_dtype")
            ),
            None,
        )

        if len(body_outputs) == 1:
            body_results = (body_call,)
            chunked_output = body_output_values[0].meta.get("output_size")
            if chunked_output is not None:
                body_call.meta["output_size"] = copy.copy(chunked_output)
        else:
            result_nodes = []
            for index, chunked in enumerate(body_output_values):
                getitem = scope_graph.call_function(
                    operator.getitem, (body_call, index)
                )
                getitem.meta = copy.copy(chunked.meta)
                getitem.meta["application_dtype"] = chunked.meta.get(
                    "application_dtype"
                )
                getitem.meta.pop("flop_req", None)
                result_nodes.append(getitem)
            body_results = tuple(result_nodes)

        joins = []
        for index, original in enumerate(body_outputs):
            join_inputs = [body_results[index]]
            shape = list(_get_shape(original) or [])
            if kind == "cat":
                join_dim = next(
                    (
                        d
                        for d, extent in enumerate(shape)
                        if axis
                        in {s.name for s in getattr(extent, "free_symbols", ())}
                    ),
                    -1,
                )
                if join_dim < 0:
                    raise ValueError(
                        f"cannot locate axis {axis!r} on join output"
                    )
                join = scope_graph.call_function(
                    torch.ops.aten.cat.default, args=(join_inputs, join_dim)
                )
            else:
                stacked = scope_graph.call_function(
                    torch.ops.aten.stack.default, args=(join_inputs, 0)
                )
                stacked.meta["output_size"] = [[1, *shape]]
                stacked.meta["application_dtype"] = original.meta.get(
                    "application_dtype"
                )
                join = scope_graph.call_function(
                    torch.ops.aten.sum.dim_IntList, args=(stacked, [0])
                )
                join.meta["flop_req"] = flop_requirement_vectors[
                    torch.ops.aten.sum
                ]([1, *shape], [0])
            join.meta["output_size"] = [list(shape)]
            join.meta["application_dtype"] = original.meta.get(
                "application_dtype"
            )
            joins.append(join)

        # One representative join per exit; concretize replicates the branch.
        scope_graph.output(joins[0] if len(joins) == 1 else tuple(joins))
        scope_graph.lint()
        scope_gm.meta["tiling"] = {
            "axis": axis,
            "split": split,
            "parfactor": str(parfactor),
            "contracting": bool(contracting),
            "symbolic": True,
        }

        # Replace the region in the parent with one call to the axis scope.
        gm.add_submodule(module_name, scope_gm)
        with gm.graph.inserting_after(closure[-1]):
            outer_call = gm.graph.call_module(
                module_name, args=tuple(body_inputs)
            )
        outer_call.meta["application_dtype"] = body_call.meta[
            "application_dtype"
        ]
        # The parent reads the axis call as the region's replacement value, so
        # it needs the joined full-extent shape. Multi-output regions expose the
        # per-exit getitems created below, which copy the join metadata.
        if len(joins) == 1:
            outer_call.meta["output_size"] = copy.copy(
                joins[0].meta.get("output_size")
            )

        exit_positions = {
            node: index for index, node in enumerate(body_outputs)
        }
        if len(joins) == 1:
            for user in list(exits[0].users):
                if user not in closure_set:
                    user.replace_input_with(exits[0], outer_call)
        else:
            with gm.graph.inserting_after(outer_call):
                outer_results = []
                for index, join in enumerate(joins):
                    getitem = gm.graph.call_function(
                        operator.getitem, (outer_call, index)
                    )
                    getitem.meta = copy.copy(join.meta)
                    outer_results.append(getitem)
            for exit_node in exits:
                output_index = exit_positions[exit_node]
                for user in list(exit_node.users):
                    if user not in closure_set:
                        user.replace_input_with(
                            exit_node, outer_results[output_index]
                        )

        # Drop the original body; the axis scope owns the copies now.
        erase_nodes(gm, body_nodes)
        gm.graph.eliminate_dead_code()
        stable_topological_sort(gm)
        gm.graph.lint()
        return scope_gm


    @staticmethod
    def concretize(
        gm: torch.fx.GraphModule,
        bindings: Mapping[str, int],
        *,
        concretized: dict[tuple, str] | None = None,
        visited: set[int] | None = None,
    ) -> torch.fx.GraphModule:
        """Expand symbolic axis scopes in place, bottom-up.

        Nested scopes are concrete before their enclosing scope replicates its
        branch. Expanding one scope clones its representative slice branch and
        rebuilds that scope's join arity; child modules are shared across branch
        calls and are never cloned.

        Args:
            gm: Graph module whose tiled scopes to expand.
            bindings: Symbolic dimension name -> concrete value.
            concretized: Kept for API compatibility; tiled scopes expand in
                place and do not use a template-to-concrete memo.
            visited: Optional shared set of module ids already walked.

        Returns:
            ``gm`` with its tiled scopes expanded (mutated in place).

        Example:
            >>> ParallelModel.concretize(tiled.gm, {"M": 32, "K": 128, "N": 128})
        """
        # Guard against re-walking a shared body module through sibling wrappers.
        if visited is None:
            visited = set()
        if id(gm) in visited:
            return gm
        visited.add(id(gm))

        # Bottom-up: a body and any nested scopes inside it must be concrete
        # before the enclosing tiled scope starts replicating body calls.
        for node in list(gm.graph.nodes):
            if node.op == "call_module":
                ParallelModel.concretize(
                    gm.get_submodule(node.target),
                    bindings,
                    concretized=concretized,
                    visited=visited,
                )

        # Expand this scope if it is a symbolic tiled scope.
        meta = gm.meta.get("tiling")
        if meta is not None and meta.get("symbolic", False):
            factor = meta.get("factor")
            if factor is None:
                raise ValueError("symbolic tiled scope has no selected factor")
            ParallelModel._expand_branch(gm, int(factor))
            meta["symbolic"] = False

        return gm


    @staticmethod
    def _expand_branch(
        gm: torch.fx.GraphModule,
        parfactor: int,
    ) -> None:
        """Rewrite one symbolic axis scope into ``parfactor`` concrete branches.

        The scope holds one representative branch (its slices and the leaf call
        or operation chain), plus the join for each exit. Expansion clones the
        branch in place and rebuilds the join arity. Child scope modules are
        shared across branch calls and are never cloned.

        Args:
            gm: Axis scope to rewrite in place.
            parfactor: Number of concrete branches to produce.

        Returns:
            ``None``; ``gm`` is rewritten in place.

        Example:
            >>> ParallelModel._expand_branch(axis_scope, 2)
        """
        nodes = list(gm.graph.nodes)
        output = next(node for node in nodes if node.op == "output")
        output_values = output.args[0]
        if not isinstance(output_values, (tuple, list)):
            output_values = (output_values,)
        joins = [value for value in output_values if isinstance(value, torch.fx.Node)]
        if not joins:
            raise ValueError("tiled scope has no join")

        # A joining scope ends in cat, or in stack -> sum for a contraction.
        # The representative branch root is the single value feeding each join.
        internal_join_nodes: set[torch.fx.Node] = set(joins)
        branch_roots: list[torch.fx.Node] = []
        for join in joins:
            if join.target in {
                torch.ops.aten.cat.default,
                torch.ops.aten.stack.default,
            }:
                representative = join.args[0]
            elif join.target is torch.ops.aten.sum.dim_IntList:
                stacked = join.args[0]
                if not isinstance(stacked, torch.fx.Node):
                    raise ValueError("sum join has no stack input")
                internal_join_nodes.add(stacked)
                representative = stacked.args[0]
            else:
                raise ValueError(f"unsupported tiling join {join.target!r}")
            if not isinstance(representative, (tuple, list)) or not representative:
                raise ValueError("join has no representative input")
            root = representative[0]
            if not isinstance(root, torch.fx.Node):
                raise ValueError("join representative is not a node")
            branch_roots.append(root)

        # Collect only the representative branch, stopping at placeholders and
        # at the join nodes that will be rebuilt below.
        seen: set[torch.fx.Node] = set()
        pending = list(branch_roots)
        while pending:
            node = pending.pop()
            if node in seen or node in internal_join_nodes or node.op == "placeholder":
                continue
            seen.add(node)
            pending.extend(node.all_input_nodes)
        branch_nodes = [node for node in nodes if node in seen]
        template_slices = [
            node
            for node in branch_nodes
            if node.op == "call_function"
            and node.target is torch.ops.aten.slice.Tensor
        ]
        other_branch_nodes = [
            node for node in branch_nodes if node not in set(template_slices)
        ]

        # Insert new nodes after the last placeholder, before the template
        # branch that will be removed.
        insertion = next(
            (node for node in reversed(nodes) if node.op == "placeholder"),
            None,
        )
        if insertion is None:
            raise ValueError("tiled scope has no placeholders")

        branch_results: list[list[torch.fx.Node]] = [[] for _ in joins]
        for index in range(parfactor):
            val_map: dict[torch.fx.Node, torch.fx.Node] = {}
            for template_slice in template_slices:
                source = template_slice.args[0]
                dim = int(template_slice.args[1])
                shape = list(_get_shape(source) or [])
                if dim < 0 or dim >= len(shape):
                    raise ValueError("tiled slice dimension is out of range")
                extent = shape[dim]
                symbolic = isinstance(extent, torch.SymInt) or (
                    isinstance(extent, sp.Basic) and bool(extent.free_symbols)
                )
                if symbolic:
                    with gm.graph.inserting_after(insertion):
                        runtime_extent = gm.graph.call_function(
                            torch.ops.aten.sym_size.int, args=(source, dim)
                        )
                    runtime_extent.meta["branch_index"] = index
                    insertion = runtime_extent
                    with gm.graph.inserting_after(insertion):
                        chunk_size = gm.graph.call_function(
                            operator.floordiv, args=(runtime_extent, parfactor)
                        )
                    chunk_size.meta["branch_index"] = index
                    insertion = chunk_size
                else:
                    runtime_extent = int(extent)
                    chunk_size = runtime_extent // parfactor
                if index == 0:
                    start = 0
                elif symbolic:
                    with gm.graph.inserting_after(insertion):
                        start = gm.graph.call_function(
                            operator.mul, args=(chunk_size, index)
                        )
                    start.meta["branch_index"] = index
                    insertion = start
                else:
                    start = index * chunk_size
                if index == parfactor - 1:
                    end = runtime_extent
                elif symbolic:
                    with gm.graph.inserting_after(insertion):
                        end = gm.graph.call_function(
                            operator.mul, args=(chunk_size, index + 1)
                        )
                    end.meta["branch_index"] = index
                    insertion = end
                else:
                    end = (index + 1) * chunk_size
                with gm.graph.inserting_after(insertion):
                    sl = gm.graph.call_function(
                        torch.ops.aten.slice.Tensor,
                        args=(source, dim, start, end, 1),
                    )
                sl.meta = copy.copy(template_slice.meta)
                sl.meta["branch_index"] = index
                insertion = sl
                val_map[template_slice] = sl

            # Clone the rest of the representative branch (leaf ops or a call
            # to the next concrete axis scope) in graph order.
            for template_node in other_branch_nodes:
                with gm.graph.inserting_after(insertion):
                    copied = gm.graph.node_copy(
                        template_node,
                        lambda value, val_map=val_map: val_map.get(value, value),
                    )
                copied.meta = copy.copy(template_node.meta)
                copied.meta["branch_index"] = index
                insertion = copied
                val_map[template_node] = copied

            for exit_index, root in enumerate(branch_roots):
                result = val_map.get(root)
                if result is None:
                    raise ValueError("branch result was not cloned")
                branch_results[exit_index].append(result)

        # Rebuild every join with one input per concrete branch.
        for exit_index, join in enumerate(joins):
            inputs = branch_results[exit_index]
            if join.target is torch.ops.aten.cat.default:
                join.args = (inputs, join.args[1])
            elif join.target is torch.ops.aten.stack.default:
                join.args = (inputs, 0)
            else:
                stacked = join.args[0]
                if not isinstance(stacked, torch.fx.Node):
                    raise ValueError("sum join has no stack input")
                stacked.args = (inputs, 0)
                partial_shape = list(_get_shape(inputs[0]) or [])
                stacked.meta["output_size"] = [[parfactor, *partial_shape]]
                join.meta["flop_req"] = flop_requirement_vectors[
                    torch.ops.aten.sum
                ]([parfactor, *partial_shape], [0])

        # Drop the representative branch after the replacements are in place.
        for node in reversed(branch_nodes):
            gm.graph.erase_node(node)
        gm.graph.eliminate_dead_code()
        stable_topological_sort(gm)
        gm.graph.lint()
