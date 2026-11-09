import contextlib
import copy
import inspect

import networkx as nx
import sympy as sp
import torch
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv, ValueRangesSLoc

from parabolic.annotate import _map_counts
from parabolic.concretize import solve_minimal_unique
from parabolic.counter import SymbolicAnnotator
from parabolic.flop_algebra import req_to_vec, requirement_dict
from parabolic.symbol import create_symint


@contextlib.contextmanager
def _suppress_symint_guard():
    """Temporarily prevent SymInt specialization via guard_int.

    During symbolic model creation, math.sqrt() and similar stdlib calls
    force SymInt.__index__() which triggers guard_int() -> evaluate_expr(),
    permanently specializing the symbol to its hint value.

    This context manager patches guard_int to return the hint directly,
    skipping evaluate() so no Eq(symbol, value) guard is created.
    """
    from torch.fx.experimental.sym_node import SymNode
    orig_guard_int = SymNode.guard_int
    SymNode.guard_int = lambda self, file="", line=0: self.hint
    try:
        yield
    finally:
        SymNode.guard_int = orig_guard_int

# Parabolic
class SymbolicModel:
    def __init__(self, model, symbols=None, batch_shape=None, *args, **kwargs):
        symbols = {} if symbols is None else symbols
        batch_shape = {} if batch_shape is None else batch_shape
        self.args = args
        self.kwargs = kwargs
        self.model = model
        # Transform symbols to be product with _dup variants
        # input_symbols = {}
        # for name, expr in symbols.items():
        #     # Create the _factor and _dup symbols
        #     factor_name = f"{name}_factor"
        #     factor_symbol = sp.Symbol(factor_name)
        #     dup_name = f"{name}_dup"
        #     dup_symbol = sp.Symbol(dup_name)
        #     # Original becomes product of factor and duplicate
        #     input_symbols[name] = factor_symbol * dup_symbol
        #     # Also add the factor and duplicate symbols
        #     input_symbols[factor_name] = factor_symbol
        #     input_symbols[dup_name] = dup_symbol
        # self.input_symbols = input_symbols
        self.input_symbols = dict(symbols)
        sig = inspect.signature(self.model)
        self.positional_params = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
        self._reset()

        # print(self.input_symbols)

        # Unrolled `with` statement
        # See PEP343: https://peps.python.org/pep-0343/
        # self.fake_tensor_mode.__enter__()
        # batch_size = sp.symbols('batch_size', positive=True, integer=True)
        for sym, shape in batch_shape.items():
            self._add_input_symbol(sym, shape)
        self._register_all_symbols()
        self._init_pytorch()

        # self.fake_tensor_mode.__exit__(None, None, None)

    def _reset(self):
        """Clears all symbols and resets ShapeEnv/FakeTensorMode
        """
        self.registered_symbols = {}
        self.symbol_hints = {}
        self.shape_env = ShapeEnv()
        self.fake_tensor_mode = FakeTensorMode(
            shape_env=self.shape_env,
            allow_non_fake_inputs=True
        )
        self.symbolic_kwargs = self.kwargs.copy()
        self.concrete_kwargs = self.kwargs.copy()

    def _register_symbol(self, symbol_name, sympy_expr, hint):
        if symbol_name in self.registered_symbols:
            raise ValueError(f'Symbol {symbol_name} already registered.')

        ######
        # Pretty bad PyTorch surgery, sorry...
        ######
        # print('Registering', symbol_name, 'as', sympy_expr)
        symbol = create_symint(self.fake_tensor_mode.shape_env, hint)
        symbol.node._expr = sympy_expr
        self.shape_env.add_backed_var_to_val(sympy_expr, hint)
        self.fake_tensor_mode.shape_env.backed_var_to_val[sympy_expr] = sp.Integer(hint)
        self.fake_tensor_mode.shape_env.var_to_range[sympy_expr] = (
            self.fake_tensor_mode.shape_env._default_value_range(True)
        )
        sloc = self.fake_tensor_mode.shape_env._get_sloc()
        self.fake_tensor_mode.shape_env.var_to_range_sloc[sympy_expr] = ValueRangesSLoc(
            sloc,
            sloc,
        )
        self.registered_symbols[symbol_name] = symbol

        if symbol_name in self.positional_params:
            self.symbolic_kwargs[symbol_name] = symbol
            self.concrete_kwargs[symbol_name] = hint

    def _register_all_symbols(self):
        concrete_hints = solve_minimal_unique(self.input_symbols.values(), upper_bound=200)
        self.concrete_hints = {}
        with self.fake_tensor_mode:
            for name, sympy_expr in self.input_symbols.items():
                hint = 1
                if isinstance(sympy_expr, sp.Symbol):
                    hint = concrete_hints[sympy_expr]
                else:
                    hint = int(sympy_expr.subs(concrete_hints))
                self.concrete_hints[name] = hint
                # print('Name:', name)
                # print('Expr:', sympy_expr)
                # print('Hint:', hint)

                self._register_symbol(name, sympy_expr, hint)
        # print('hints', self.concrete_hints)

    # TODO: fix
    # def _replace_input_symbol(self, symbol_name, sympy_expr, hint):
    #     if not symbol_name in self.input_symbols:
    #         raise Exception(f'Symbol {symbol_name} is not a valid input symbol.')
    #     self._add_input_symbol(symbol_name, sympy_expr, hint)
    #     self._reset()
    #     self._register_all_symbols()

    def _add_input_symbol(self, symbol_name, sympy_expr):
        self.input_symbols[symbol_name] = sympy_expr

    # TODO: fix
    # def __getitem__(self, symbol_name):
    #     if not symbol_name in self.input_symbols:
    #         raise Exception(f'Symbol {symbol_name} does not exist.')
    #     return self.input_symbols[symbol_name]
    #
    # def __setitem__(self, symbol_name, expr_hint):
    #     if not symbol_name in self.input_symbols:
    #         raise Exception(f'Symbol {symbol_name} does not exist.')
    #     if not instance(expr_hint, tuple) or len(expr_hint) != 2:
    #         raise Exception(f'Symbol {symbol_name} is not a 2-tuple.')
    #     sympy_expr, hint = expr_hint
    #     if not instance(sympy_expr, sp.Basic):
    #         raise Exception(f'Expression {symbol_name} is not a SymPy expression.')
    #     self._add_input_symbol(symbol_name, sympy_expr, hint)
    #
    # def __delitem__(self, symbol_name):
    #     if not symbol_name in self.input_symbols:
    #         raise Exception(f'Symbol {symbol_name} does not exist.')
    #     del self.input_symbols[symbol_name]

    def _init_pytorch(self):
        with self.fake_tensor_mode, _suppress_symint_guard():
            self.symbolic_model = self.model(*self.args, **self.symbolic_kwargs)
        self.concrete_model = self.model(*self.args, **self.concrete_kwargs)

    def get_all_symbols(self):
        return self.symbols

    def get_symbol(self, symbol_name: str):
        if symbol_name in self.registered_symbols:
            return self.registered_symbols[symbol_name]

        raise KeyError(f'Symbol with name {symbol_name} does not exist.')

    def annotate(self, input_symbols: tuple, *additional_input_symbols: tuple):
        input_shapes = (input_symbols, *additional_input_symbols)
        counts = {}
        with (self.fake_tensor_mode,
             _suppress_symint_guard(),
             SymbolicAnnotator() as context,
             torch._dynamo.config.patch(fake_tensor_cache_enabled=False)):

            symbolic_inputs = tuple(
                torch.rand(
                    tuple(self.get_symbol(symbol_name) for symbol_name in input_shape)
                )
                for input_shape in input_shapes
            )
            self.symbolic_model.forward(*symbolic_inputs)
            counts['flops'] = context.flop_counts
            counts['io']    = { 'inputs': context.input_shapes, 'outputs': context.output_shapes }
            counts['flop_req'] = context.flop_requirements
            counts = copy.deepcopy(counts)

            # if not hasattr(self, 'gm'):
            # self.gm = _map_counts(self.concrete_model, concrete_input, counts)
            self.gm = _map_counts(self.concrete_model, symbolic_inputs, counts)
            # from torch._inductor.fx_passes.post_grad import remove_noop_ops
            # remove_noop_ops(self.gm.graph)
            self.gm.graph.eliminate_dead_code()

            # from torch.fx.experimental.proxy_tensor import make_fx
            # fx_graph = make_fx(self.symbolic_model)(symbolic_input)

            # compiled_model = torch.compile(self.symbolic_model)
            # print('Compiled', compiled_model)

            # Probably better..
            # from torch.fx.experimental.proxy_tensor import make_fx
            # fx = make_fx(self.concrete_model)(concrete_input)
            # print(fx.graph)

            # from torch.fx.passes.shape_prop import ShapeProp
            # out = ShapeProp(self.gm, fake_mode=self.fake_tensor_mode).propagate(symbolic_input)
            # print('Prop', out)

            # else:
            #     from torch.fx.passes.shape_prop import ShapeProp
                # ShapeProp(self.gm).propagate(concrete_input)
                # self.gm = _map_counts_gm(self.gm, counts)

        return self.gm

    def _substitute_expr(self, expr: sp.Basic, concrete_table=None):
        evaluated_expr = expr
        for symbol in expr.free_symbols:
            hint = self.concrete_hints[symbol.name]
            if isinstance(concrete_table, dict) and symbol.name in concrete_table:
                hint = concrete_table[symbol.name]
            evaluated_expr = evaluated_expr.subs(symbol, hint)
        return int(evaluated_expr)

    def to_nx(self, concrete_table=None):
        # Add ATen ops here to not prune them
        supported_ops = {
            torch.ops.aten.addmm,
            torch.ops.aten.bmm,
            torch.ops.aten._softmax,
            # torch.ops.aten.slice,
            # torch.ops.aten.add,
            torch.ops.aten.native_layer_norm,
            torch.ops.aten.gelu,
            torch.ops.aten.sum,
            'get_attr',
            'placeholder',
            'output'
        }

        ignore_completely = {
            torch.ops.aten.sym_size,
        }

        # Add nodes
        dfg = nx.DiGraph()
        existing_nodes = set()
        for node in self.gm.graph.nodes:
            shape = node.meta.get('output_size')
            if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
                shape = shape[0]
            shape = tuple(shape) if isinstance(shape, (list, tuple)) else ()
            if (node.op == 'call_function'
                and isinstance(node.target, torch._ops.OpOverload)):

                flop = node.meta.get('flops', 0)
                if isinstance(flop, sp.Basic):
                    flop = self._substitute_expr(flop, concrete_table=concrete_table)
                if 'flop_req' in node.meta:
                    flop_req = node.meta['flop_req']
                elif 'flops' in node.meta:
                    flop_req = [0, 0, flop]
                else:
                    flop_req = 0
                dfg.add_node(
                    node.name,
                    func=node.target._overloadpacket,
                    type='compute',
                    flop=flop,
                    flop_req=flop_req,
                    collective_latency=node.meta.get('collective_latency', 0),
                    shape=shape,
                )
                existing_nodes.add(node.name)
            elif node.name.startswith('getitem'):
                dfg.add_node(
                    node.name,
                    func=node.target,
                    type='mem',
                    flop=0,
                    flop_req=req_to_vec(requirement_dict),
                    shape=shape,
                )
                existing_nodes.add(node.name)
            elif node.name.startswith('_param') or node.name.startswith('arg'):
                dfg.add_node(
                    node.name,
                    func=node.op,
                    type='mem',
                    flop=0,
                    flop_req=req_to_vec(requirement_dict),
                    shape=shape,
                )
                existing_nodes.add(node.name)
            elif node.name.startswith('output'):
                dfg.add_node(
                    node.name,
                    func=node.target,
                    type='mem',
                    flop=0,
                    flop_req=req_to_vec(requirement_dict),
                    shape=shape,
                )
                existing_nodes.add(node.name)

        # Add edges
        for node in self.gm.graph.nodes:
            if node.op in ['call_function', 'get_attr', 'placeholder', 'output']:
                edge_shape = node.meta.get('output_size')
                if (isinstance(edge_shape, (list, tuple)) and edge_shape
                        and isinstance(edge_shape[0], (list, tuple))):
                    edge_shape = edge_shape[0]
                edge_shape = tuple(edge_shape) if isinstance(edge_shape, (list, tuple)) else ()
                for user in list(node.users.keys()):
                    user = str(user)
                    if user in existing_nodes and node.name in existing_nodes:
                        data = 0
                        if edge_shape:
                            import math
                            data = math.prod(edge_shape)
                            if isinstance(data, sp.Basic):
                                data = self._substitute_expr(data, concrete_table=concrete_table)
                        dfg.add_edge(node.name, user, data=data, shape=edge_shape)

        # Incrementally edges from unsupported ops
        # Reconnect edges to neighbors
        for node in list(dfg.nodes()):
            if (
                dfg.nodes[node]['func'] not in supported_ops
                and dfg.nodes[node]['func'] not in ignore_completely
            ):
                in_edges = list(dfg.in_edges(node))
                out_edges = list(dfg.out_edges(node))
                for src, _ in in_edges:
                    for _, dst in out_edges:
                        # Pick the smaller of incoming and outgoing data.
                        # Slice nodes reduce data (outgoing < incoming);
                        # cat nodes expand data (incoming < outgoing).
                        in_data = dfg[src][node].get('data', 0) if 'data' in dfg[src][node] else 0
                        out_data = dfg[node][dst].get('data', 0) if 'data' in dfg[node][dst] else 0
                        dfg.add_edge(src, dst, data=min(in_data, out_data), shape=dfg.nodes[src].get('shape', ()))

        # Delete nodes for unsupported ops
        for node in list(dfg.nodes()):
            if dfg.nodes[node]['func'] not in supported_ops:
                dfg.remove_node(node)

        return dfg


