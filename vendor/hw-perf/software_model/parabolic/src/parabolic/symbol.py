from torch.fx.experimental.symbolic_shapes import ShapeEnv


def create_symtype(cls, pytype, shape_env: ShapeEnv, val):
    from torch._dynamo.source import ConstantSource
    from torch.fx.experimental.sym_node import SymNode

    symbol = shape_env.create_symbol(
        val,
        source = ConstantSource(f"__testing_only{len(shape_env.backed_var_to_val)}"),
        constraint_dim=None,
        do_not_specialize_zero_one=True
    )

    return cls(
        SymNode(
            symbol,
            shape_env,
            pytype,
            hint=val,
        )
    )


def create_symint(shape_env: ShapeEnv, val: int):
    from torch import SymInt
    return create_symtype(SymInt, int, shape_env, val)

