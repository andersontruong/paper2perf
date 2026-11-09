"""Capability-aware normalized FLOP cost algebra.

Raw operation counts describe what an operator requires.  A hardware capability
weights each requirement into the common normalized-FLOP unit used by sparse-roof
compute peaks.  Expressions remain symbolic until a mapping needs concrete cost.
"""

from __future__ import annotations

from collections.abc import Mapping

import sympy as sp


class CapabilityCostError(ValueError):
    """Raised when a requirement cannot execute on a capability vector."""


def requirements_supported(
    requirements: Mapping[str, object],
    capability: Mapping[str, float],
) -> bool:
    """Return whether every nonzero requirement has a positive capability."""
    for name, value in requirements.items():
        expression = _to_expression(value)
        if expression == 0:
            continue
        factor = capability.get(name, 0.0)
        if not isinstance(factor, int | float) or isinstance(factor, bool) or factor <= 0:
            return False
    return True


def normalized_flop_expression(
    requirements: Mapping[str, object],
    capability: Mapping[str, float],
) -> sp.Expr:
    """Return sparse-roof normalized FLOPs for a named requirement mapping."""
    if not requirements:
        raise CapabilityCostError("flop requirements must not be empty")
    if not requirements_supported(requirements, capability):
        missing = sorted(
            name for name, value in requirements.items()
            if _to_expression(value) != 0 and capability.get(name, 0.0) <= 0
        )
        raise CapabilityCostError(
            "hardware lacks positive capability for requirements: " + ", ".join(missing)
        )
    return sp.simplify(sum(
        _to_expression(value) * sp.sympify(capability[name])
        for name, value in requirements.items()
        if _to_expression(value) != 0
    ))


def _to_expression(value: object) -> sp.Expr:
    if isinstance(value, bool):
        raise CapabilityCostError("boolean values are not FLOP requirements")
    if isinstance(value, int):
        return sp.Integer(value)
    if isinstance(value, sp.Expr):
        return value
    sympy_method = getattr(value, "_sympy_", None)
    if callable(sympy_method):
        result = sympy_method()
        if isinstance(result, sp.Expr):
            return result
    raise CapabilityCostError(f"unsupported FLOP requirement expression {value!r}")
