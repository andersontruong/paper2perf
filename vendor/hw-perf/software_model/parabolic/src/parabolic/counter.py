from collections import defaultdict
from typing import Any

import sympy as sp
import torch
from torch._ops import OpOverloadPacket
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry, shape_wrapper

from parabolic.flop_algebra import flop_requirement_vectors


class SymbolicAnnotator:
    """Symbolic tracer for PyTorch operations that counts FLOPs and IO.

    This class provides a context manager interface for symbolically tracing
    PyTorch operations to count floating point operations (FLOPs) and track
    input/output shapes. It uses PyTorch's predefined flop_registry for
    operation-specific FLOP counting formulas and can be extended with
    custom mappings.

    Attributes:
        flop_registry (dict): Registry mapping operation packets to FLOP
            counting functions. Built from PyTorch's flop_registry plus any
            custom mappings provided.
        flop_counts (defaultdict): Stores computed FLOP counts for each
            operation packet encountered during tracing.
        input_shapes (defaultdict): Stores input shapes for each operation
            packet encountered during tracing.
        output_shapes (defaultdict): Stores output shapes for each operation
            packet encountered during tracing.

    Args:
        custom_mapping (Optional[dict[Any, Any]]): Optional custom mapping of
            operation packets to FLOP counting functions. Can extend or override
            PyTorch's built-in formulas.

    Example:
        >>> with SymbolicAnnotator() as annotator:
        ...     # Run your model here
        ...     model(input)
        ... # Access counts after context
        ... flop_counts = annotator.flop_counts
        ... input_shapes = annotator.input_shapes

    The class is designed to work with _SymbolicAnnotator (TorchDispatchMode)
    which handles the actual dispatch interception during tracing.
    """
    def __init__(
            self,
            custom_mapping: dict[Any, Any] | None = None
    ):
        custom_mapping = {} if custom_mapping is None else custom_mapping
        self.flop_registry = {
            **flop_registry,
            **{
                k: v if getattr(v, "_get_raw", False) else shape_wrapper(v)
                for k, v in custom_mapping.items()
            },
        }
        # self.flop_requirement_vectors = flop_requirement_vectors
        self.flop_requirement_vectors = {
            **{
                k: v if getattr(v, "_get_raw", False) else shape_wrapper(v)
                for k, v in flop_requirement_vectors.items()
            }
        }
        self.flop_counts = defaultdict(list)
        self.flop_requirements = defaultdict(list)
        self.input_shapes  = defaultdict(list)
        self.output_shapes = defaultdict(list)

    def __enter__(self):
        self.mode = _SymbolicAnnotator(self)
        self.mode.__enter__()
        return self

    def __exit__(self, *args):
        assert self.mode is not None
        b = self.mode.__exit__(*args)
        self.mode = None  # break cycles
        return b

    @classmethod
    def _remove_dup(cls, symint: Any | torch.SymInt):
        """Remove duplicate symbols from symbolic expressions.

        PyTorch sometimes likes to concretize factors in a symbolic expression.
        We remedy this by inserting a `*_dup` symbol as an extra factor.
        Here, we remove this extra symbol by substituting it with 1.

        Args:
            symint (Union[Any, torch.SymInt]): Symbolic integer or any value

        Returns:
            Union[Any, sympy.Expr]: Cleaned sympy expression or original value
        """
        if not isinstance(symint, torch.SymInt):
            return symint

        sympy_expr = symint._sympy_()
        free_symbols = list(sympy_expr.free_symbols)
        for symbol in free_symbols:
            if symbol.name.endswith('dup'):
                sympy_expr = sympy_expr.subs(symbol, 1)

        return sympy_expr

    def _count_flops(self, func_packet: "OpOverloadPacket", out, args, kwargs):
        """Count FLOPs for a given operation packet.

        Looks up the flop counting formula from the flop_registry (predefined in
        PyTorch) based on the operation packet, evaluates the formula with the
        given arguments, removes duplicate symbols from the result, and stores
        the computed flop count.

        Args:
            func_packet (OpOverloadPacket): Operation packet to count FLOPs for
            out: Output tensor/value from the operation
            args: Positional arguments to the operation
            kwargs: Keyword arguments to the operation

        Returns:
            None: Side effect - appends flop count to flop_counts list.
        """
        flop_count_func = self.flop_registry[func_packet]
        flop_count = flop_count_func(*args, **kwargs, out_val=out)  # type: ignore[operator]
        flop_count = self._remove_dup(flop_count)
        self.flop_counts[func_packet].append(flop_count)

    def _count_requirements(self, func_packet: "OpOverloadPacket", out, args, kwargs):
        """Count flop requirement vectors for a given operation packet.

        Looks up the requirement vector formula from flop_requirement_vectors
        (in parabolic.flop_algebra), evaluates it with the given arguments,
        and stores the computed requirement vector.

        Args:
            func_packet (OpOverloadPacket): Operation packet to count requirements for
            out: Output tensor/value from the operation
            args: Positional arguments to the operation
            kwargs: Keyword arguments to the operation

        Returns:
            None: Side effect - appends requirement vector to flop_requirements list.
        """
        if func_packet in self.flop_requirement_vectors:
            req_func = self.flop_requirement_vectors[func_packet]
            req_vec = req_func(*args, **kwargs, out_val=out)
            self.flop_requirements[func_packet].append(req_vec)

    @classmethod
    def _flatten_shapes(cls, arg: list | torch.Tensor | Any) -> list | Any:
        """Recursively flatten and clean tensor shapes by removing duplicate symbols.

        Args:
            arg: Tensor, list, or any value to process

        Returns:
            Cleaned shapes with duplicate symbols removed
        """
        if isinstance(arg, torch.Tensor):
            return [cls._remove_dup(dim) for dim in arg.shape]
        elif isinstance(arg, list):
            return [cls._flatten_shapes(subarg) for subarg in arg]
        else:
            return arg

    def _count_io(self, func_packet: "OpOverloadPacket", out, args, kwargs):
        """Count IO sizes for inputs and outputs of an operation.

        Processes input and output shapes, removes duplicate symbols, and stores
        the computed IO sizes in input_shapes and output_shapes.

        Args:
            func_packet (OpOverloadPacket): Operation packet to count IO for
            out: Output tensor/value from the operation
            args: Positional arguments to the operation
            kwargs: Keyword arguments to the operation

        Returns:
            None: Side effect - appends IO sizes to input_shapes/output_shapes
        """
        # Process input shapes
        args_shape = [self._normalize_arg_shape(arg) for arg in args]
        args_shape = [self._remove_dup_from_shapes(arg) for arg in args_shape]
        self.input_shapes[func_packet].append(args_shape)

        # Process output shapes
        out_shape = self._normalize_arg_shape(out)
        out_shape = self._remove_dup_from_shapes(out_shape)
        # Handle multiple outputs
        # In this case (for example native_layer_norm), we'll just output the first tensor for now
        if isinstance(out_shape, list) and isinstance(out_shape[0], list) and len(out_shape) > 1:
            out_shape = out_shape[0]
        self.output_shapes[func_packet].append(out_shape)


    def _normalize_arg_shape(self, arg):
        """Normalize argument to shape list.

        Converts tensors/lists to shape lists, and other values to [1].
        """
        if isinstance(arg, (list, torch.Tensor)):
            return self._flatten_shapes(arg)
        if isinstance(arg, (tuple, torch.Tensor)):
            return self._flatten_shapes(list(arg))
        else:
            return [1]

    def _remove_dup_from_shapes(self, shapes):
        """Remove duplicate symbols from shape list or nested shape list.

        Args:
            shapes: List of shapes (possibly nested) to clean

        Returns:
            Cleaned shapes with duplicate symbols divided out
        """
        if isinstance(shapes, list):
            return [self._remove_dup_from_shapes(item) for item in shapes]
        elif isinstance(shapes, torch.SymInt) and isinstance(shapes._sympy_(), sp.Expr):
            sympy_expr = shapes._sympy_()
            new_sympy_expr = sympy_expr
            for symbol in sympy_expr.free_symbols:
                if symbol.name.endswith('dup'):
                    new_sympy_expr //= symbol
            return new_sympy_expr
        else:
            return shapes

class _SymbolicAnnotator(TorchDispatchMode):
    """TorchDispatchMode wrapper that intercepts and counts PyTorch operations.

    This class implements the PyTorch TorchDispatchMode interface to intercept
    PyTorch operations during symbolic tracing. It works in conjunction with
    SymbolicAnnotator to count floating point operations
    (FLOPs) and track input/output shapes for ATen operations.

    Example:
        >>> # Used internally by SymbolicAnnotator context manager
        >>> annotator = SymbolicAnnotator()
        >>> with _SymbolicAnnotator(annotator):
        ...     # Operations are intercepted and counted here
        ...     result = torch.add(a, b)
        ... # Access counts via annotator
        >>> flop_counts = annotator.flop_counts
        >>> input_shapes = annotator.input_shapes

    The class automatically handles operation decomposition and ensures that
    FLOPs and IO are counted accurately
    """

    def __init__(self, counter: SymbolicAnnotator):
        self.counter = counter

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        # Skip ops from non-standard dispatch_sizes_strides_policy such as NJT
        if func in {
                    # torch.ops.aten.sym_is_contiguous.default,
                    torch.ops.aten.is_contiguous.default,
                    torch.ops.aten.is_contiguous.memory_format,
                    torch.ops.aten.is_strides_like_format.default,
                    torch.ops.aten.is_non_overlapping_and_dense.default,
                    torch.ops.aten.size.default,
                    torch.ops.aten.sym_size.default,
                    torch.ops.aten.stride.default,
                    torch.ops.aten.sym_stride.default,
                    torch.ops.aten.storage_offset.default,
                    torch.ops.aten.sym_storage_offset.default,
                    torch.ops.aten.numel.default,
                    torch.ops.aten.sym_numel.default,
                    torch.ops.aten.dim.default,
                    torch.ops.prim.layout.default}:

            return NotImplemented

        if isinstance(func, torch._ops.HigherOrderOperator):
            return 0

        # If we don't have func in flop_registry, see if it can decompose
        if func not in self.counter.flop_registry and func is not torch.ops.prim.device.default:
            with self:
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r

        # no further decomposition; execute & count flops
        # Count IO based on shape
        out = func(*args, **kwargs)
        if isinstance(func, torch._ops.OpOverload) and func.namespace == 'aten':
            self.counter._count_io(func._overloadpacket, out, args, kwargs)
        if func._overloadpacket in self.counter.flop_registry:
            self.counter._count_flops(func._overloadpacket, out, args, kwargs)
        if func._overloadpacket in self.counter.flop_requirement_vectors:
            self.counter._count_requirements(func._overloadpacket, out, args, kwargs)
        return out
