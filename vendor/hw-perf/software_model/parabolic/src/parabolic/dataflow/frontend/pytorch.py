"""Symbolic capture of a PyTorch model as an annotated Parabolic ParallelModel.

Model definitions live with the examples and benchmarks that use them. This
module owns only the capture entry point: given a module class, its named
symbolic parameters, and its symbolic input shapes, it produces the annotated
``ParallelModel`` the mapper consumes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from parabolic.parallel_model import ParallelModel


def create_parabolic_parallel_model(
    model: type[torch.nn.Module],
    symbols: Mapping[str, object],
    input_shapes: tuple[tuple[str, ...], ...],
    *,
    dtype: str = "fp16",
    args: tuple[object, ...] = (),
    kwargs: Mapping[str, object] | None = None,
) -> Any:
    """Capture a symbolic PyTorch model as an annotated Parabolic ParallelModel.

    ``model`` is a module class or factory: the symbolic trace instantiates it
    with SymInt-backed sizes. ``input_shapes`` lists the symbolic dimensions of
    every positional input tensor, in order.
    """
    if not callable(model):
        raise TypeError("Parabolic capture requires a PyTorch module class or factory")
    if not symbols or any(not isinstance(name, str) or not name for name in symbols):
        raise ValueError("Parabolic capture requires named symbolic expressions")
    if not input_shapes or any(
        not shape or len(set(shape)) != len(shape) for shape in input_shapes
    ):
        raise ValueError("Parabolic capture requires non-empty symbolic input shapes")
    unknown = {name for shape in input_shapes for name in shape}.difference(symbols)
    if unknown:
        raise ValueError(
            f"PyTorch input shape references unknown symbols: {', '.join(sorted(unknown))}"
        )
    if not dtype:
        raise ValueError("Parabolic capture requires a dtype")

    parallel_model = ParallelModel(model, dict(symbols), {}, *args, **dict(kwargs or {}))
    parallel_model.annotate(*input_shapes)
    parallel_model.propagate_attr_metadata_pass()
    parallel_model.prune_sym_size_chains()
    _normalize_tensor_output_shapes(parallel_model, torch)
    _set_input_output_shapes(parallel_model, symbols, input_shapes)
    for node in parallel_model.gm.graph.nodes:
        if node.meta.get("output_size") is not None:
            node.meta["application_dtype"] = dtype
    return parallel_model


def _set_input_output_shapes(
    model: Any,
    symbols: Mapping[str, object],
    input_shapes: tuple[tuple[str, ...], ...],
) -> None:
    placeholders = tuple(node for node in model.gm.graph.nodes if node.op == "placeholder")
    if len(placeholders) != len(input_shapes):
        raise ValueError("annotated PyTorch graph input count does not match the declared shapes")
    for node, shape in zip(placeholders, input_shapes, strict=True):
        node.meta["output_size"] = [symbols[name] for name in shape]


def _normalize_tensor_output_shapes(model: Any, torch: Any) -> None:
    """Keep legacy IO accounting metadata out of semantic tensor shapes."""
    for node in model.gm.graph.nodes:
        value = node.meta.get("val")
        output_size = node.meta.get("output_size")
        semantic_size = (
            output_size[0]
            if (
                isinstance(output_size, (tuple, list))
                and len(output_size) == 1
                and isinstance(output_size[0], (tuple, list))
            )
            else output_size
        )
        if (
            isinstance(value, torch.Tensor)
            and isinstance(semantic_size, (tuple, list))
            and len(semantic_size) != len(value.shape)
        ):
            node.meta["output_size"] = list(value.shape)
