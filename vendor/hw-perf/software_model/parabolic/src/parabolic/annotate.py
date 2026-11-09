from collections import defaultdict
from typing import Any

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._ops import OpOverload
from torch._subclasses import FakeTensorMode
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from parabolic.counter import SymbolicAnnotator
from parabolic.symbol import create_symint

"""
FLOP specialization

Certain ops may have unique formulas for their final flops/IO
We need to annotate them to allow for postprocessing

Ex: Online softmax can be machine specific
  - Flops: (# Elements) * (Machine Flops * 3 + 7)
  - IO: (# Elements) * 3
  - For a MxN input, we'll just store M*N elementwise "exp" flops
  - Later on, we can postprocess for the true flop/IO formulas

Default can just be None
"""
flop_type = defaultdict(lambda: 'default')
flop_type[torch.ops.aten._softmax] = 'exp'
# torch.ops.aten.gelu


def _get_counts(model: torch.nn.Module, sample_input: torch.Tensor, symbolic_dims: tuple[int] = ()):
    """Symbolically traces model to count FLOPs and IO bytes
    Subtitutes the dimensions of `sample_input` with symbolic sizes.

    Args:
        model (torch.nn.Module): Model to trace
        sample_input (torch.Tensor): Input tensor
        symbolic_dims (Tuple[int]): Dimension indices to make symbolic (default: tuple())

    Returns:
        dict: ``flops`` and ``io`` counts per Core ATen Op, plus ``flop_req``
              hardware requirement vectors in ``[alu, exp, mac]`` order.

    """
    shape_env = ShapeEnv()
    counts = {}

    with (
        FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True) as fake_mode,
        enable_python_dispatcher(),
        SymbolicAnnotator() as context,
    ):
        symbolic_shape = list(sample_input.shape)
        for dim in symbolic_dims:
            symbolic_shape[dim] = create_symint(fake_mode.shape_env, 2)

        fake_input = torch.rand(symbolic_shape)
        model.forward(fake_input)
        counts['flops'] = context.flop_counts
        counts['io']    = { 'inputs': context.input_shapes, 'outputs': context.output_shapes }
        counts['flop_req'] = context.flop_requirements

    return counts


def _map_counts(
    model: torch.nn.Module,
    sample_input: torch.Tensor | tuple[torch.Tensor, ...],
    counts: dict[str, dict[str, Any]],
) -> GraphModule:
    """Maps pre-computed FLOP and IO counts to nodes in an FX graph.

    ATen Graphs are topologically ordered. We pop from each type of Op based
    on this ordering to assign counts to the correct nodes.

    Args:
        model (torch.nn.Module): Model to extract the graph from
        sample_input (torch.Tensor): Input tensor for tracing
        counts (Dict[str, Dict[str, Any]]): FLOP and IO counts from _get_counts.
            ``flop_req`` is optional: without it the requirement vectors are
            simply left unannotated.

    Returns:
        GraphModule: FX graph with metadata annotated on each node including
                     'flops', 'input_size', and 'output_size'
    """
    sample_inputs = sample_input if isinstance(sample_input, tuple) else (sample_input,)
    gm = make_fx(model)(*sample_inputs)
    requirement_counts = counts.get('flop_req', {})
    for node in list(gm.graph.nodes):
        if node.op == 'call_function' and isinstance(node.target, OpOverload):
            overloadpacket = node.target._overloadpacket
            node.meta['flop_type'] = flop_type[overloadpacket]
            if overloadpacket in counts['flops']:
                count = counts['flops'][overloadpacket].pop(0)
                node.meta['flops'] = count
            if overloadpacket in counts['io']['inputs']:
                inputs = counts['io']['inputs'][overloadpacket].pop(0)
                node.meta['input_size'] = inputs
            if overloadpacket in counts['io']['outputs']:
                outputs = counts['io']['outputs'][overloadpacket].pop(0)
                node.meta['output_size'] = outputs
            if overloadpacket in requirement_counts:
                req = requirement_counts[overloadpacket].pop(0)
                node.meta['flop_req'] = req
    for node in list(gm.graph.nodes):
        if node.op in ['get_attr', 'placeholder']:
            node.meta['flop_type'] = flop_type[node.op]
            if 'output_size' not in node.meta:
                for user in node.users:
                    found = False
                    for i, arg in enumerate(user.args):
                        if arg is node:
                            if 'input_size' not in user.meta:
                                continue
                            node.meta['output_size'] = user.meta['input_size'][i]
                            found = True
                            break
                    if found:
                        break


    return gm


def extract_dataflow(model: torch.nn.Module, sample_input: torch.Tensor, symbolic_dims: tuple[int, ...] = (0,)) -> GraphModule:
    """Extracts dataflow information (FLOPs and IO) from a model.

    Symbolically traces the model to count FLOPs and IO bytes, then maps these
    counts to the corresponding nodes in the FX graph.

    Args:
        model (torch.nn.Module): Model to extract dataflow from
        sample_input (torch.Tensor): Input tensor for tracing
        symbolic_dims (Tuple[int, ...]): Dimension indices to make symbolic (default: (0,))

    Returns:
        GraphModule: FX graph with metadata annotated on each node including
                     'flops', 'input_size', and 'output_size'
    """
    counts = _get_counts(model, sample_input, symbolic_dims=symbolic_dims)
    gm = _map_counts(model, sample_input, counts)
    return gm
