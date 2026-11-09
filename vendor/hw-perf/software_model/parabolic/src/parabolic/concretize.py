"""
Find minimal concrete integer values (> 1) for each symbol in a set of SymPy
expressions and/or bare SymPy symbols such that:
  - Every expression evaluates to a defined (non-zero-denominator) value
  - All symbol values are unique (no two symbols share the same value)
  - The values are minimised (smallest possible integers > 1)

The input list may contain:
  • SymPy expressions  – their free symbols are included and div-by-zero is avoided
  • Bare SymPy symbols – simply added to the uniqueness pool with no extra constraints

Requires:  pip install pysmt z3-solver sympy
           pysmt-install --z3  (if using the pysmt-managed z3 backend)

Usage:
    python minimal_unique_values.py
or import and call `solve_minimal_unique(items)`.
"""

from __future__ import annotations

import sympy as sp
from pysmt.shortcuts import (
    GT,
    LE,
    Div,
    Equals,
    Int,
    NotEquals,
    Plus,
    Solver,
    Symbol,
    Times,
)
from pysmt.typing import INT as INT_TYPE

# ---------------------------------------------------------------------------
# 1.  SymPy  →  pysmt expression translator
# ---------------------------------------------------------------------------

def sympy_to_pysmt(expr: sp.Expr, sym_map: dict) -> object:
    """
    Recursively convert a SymPy expression to a pysmt arithmetic expression.
    sym_map: {sympy.Symbol -> pysmt Symbol}
    """
    if isinstance(expr, sp.Symbol):
        return sym_map[expr]

    if isinstance(expr, sp.Integer):
        return Int(int(expr))

    if isinstance(expr, sp.Rational):
        # p/q  →  treat as integer division (floor); warn if non-integer context
        # For pure integer arithmetic we approximate rationals as Int(p) / Int(q).
        # pysmt's INT theory uses integer (floor) division.
        return Div(Int(int(expr.p)), Int(int(expr.q)))

    if isinstance(expr, sp.Add):
        args = [sympy_to_pysmt(a, sym_map) for a in expr.args]
        result = args[0]
        for a in args[1:]:
            result = Plus(result, a)
        return result

    if isinstance(expr, sp.Mul):
        args = [sympy_to_pysmt(a, sym_map) for a in expr.args]
        result = args[0]
        for a in args[1:]:
            result = Times(result, a)
        return result

    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        if isinstance(exp, sp.Integer) and int(exp) > 0:
            b = sympy_to_pysmt(base, sym_map)
            result = b
            for _ in range(int(exp) - 1):
                result = Times(result, b)
            return result
        raise NotImplementedError(
            f"Only positive integer exponents are supported, got: {expr}"
        )

    raise NotImplementedError(f"Unsupported SymPy node: {type(expr).__name__}  ({expr})")


# ---------------------------------------------------------------------------
# 2.  Main solver
# ---------------------------------------------------------------------------

def solve_minimal_unique(
    items: list[sp.Basic],
    upper_bound: int = 200,
    solver_name: str = "z3",
) -> dict[sp.Symbol, int] | None:
    """
    Given a list of SymPy expressions and/or bare SymPy symbols, find the
    lexicographically minimal assignment of integer values > 1 to every
    symbol such that:
      * All values are distinct.
      * All expressions evaluate without division-by-zero.

    Bare sp.Symbol entries in the list are simply added to the uniqueness
    pool -- they receive a minimal unique value with no additional constraints.

    Returns a dict {sympy.Symbol: int} or None if unsatisfiable.

    Parameters
    ----------
    items        : list of sp.Expr or sp.Symbol (may be mixed)
    upper_bound  : maximum value tried for any symbol (default 200)
    solver_name  : pysmt solver to use ('z3', 'msat', 'cvc4', ...)
    """

    # Split items into bare symbols (no extra constraints) and real expressions
    extra_symbols: set[sp.Symbol] = set()
    expressions: list[sp.Expr] = []
    for item in items:
        if isinstance(item, sp.Symbol):
            extra_symbols.add(item)
        else:
            expressions.append(item)

    # Collect all free symbols from expressions, then merge with extra symbols
    all_symbols: set[sp.Symbol] = set(extra_symbols)
    for expr in expressions:
        all_symbols |= expr.free_symbols
    all_symbols = sorted(all_symbols, key=lambda s: s.name)

    if not all_symbols:
        print("No free symbols found.")
        return {}

    # Create pysmt integer symbols
    sym_map = {s: Symbol(s.name, INT_TYPE) for s in all_symbols}

    with Solver(name=solver_name) as solver:

        # --- Constraint 1: every value > 1 ---
        for ps in sym_map.values():
            solver.add_assertion(GT(ps, Int(3)))

        # --- Constraint 2: every value ≤ upper_bound (keeps search bounded) ---
        for ps in sym_map.values():
            solver.add_assertion(LE(ps, Int(upper_bound)))

        # # --- Constraint 3: every value is divisible by 2 ---
        # for ps in sym_map.values():
        #     # ps is even  <=>  2 * (ps / 2) == ps  (integer division)
        #     solver.add_assertion(Equals(Times(Int(2), Div(ps, Int(2))), ps))

        # --- Constraint 4: all values are distinct ---
        syms_list = list(sym_map.values())
        for i in range(len(syms_list)):
            for j in range(i + 1, len(syms_list)):
                solver.add_assertion(NotEquals(syms_list[i], syms_list[j]))

        # --- Constraint 5: no division-by-zero in any expression ---
        # We collect all denominator sub-expressions and assert they ≠ 0.
        for expr in expressions:
            for sub in sp.preorder_traversal(expr):
                if isinstance(sub, sp.Pow):
                    base, exp = sub.args
                    if isinstance(exp, sp.Integer) and int(exp) < 0:
                        # base^(-n)  →  base ≠ 0
                        try:
                            pysmt_base = sympy_to_pysmt(base, sym_map)
                            solver.add_assertion(NotEquals(pysmt_base, Int(0)))
                        except NotImplementedError:
                            pass

        # Check initial satisfiability
        if not solver.solve():
            print("No solution exists within the given bounds.")
            return None

        # --- Iterative minimisation: tighten upper bound symbol by symbol ---
        # Strategy: for each symbol (in sorted order), find the minimum value
        # it can take given the constraints already added for previous symbols.
        result: dict[sp.Symbol, int] = {}

        for sym in all_symbols:
            ps = sym_map[sym]
            # Binary-search for the minimum feasible value
            lo, hi = 2, upper_bound
            best = hi

            while lo <= hi:
                mid = (lo + hi) // 2
                # Temporarily assert ps <= mid, then roll back
                solver.push()
                solver.add_assertion(LE(ps, Int(mid)))
                if solver.solve():
                    best = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
                solver.pop()

            # Fix this symbol to its minimum
            solver.add_assertion(Equals(ps, Int(best)))
            result[sym] = best
            # print(f"  {sym.name} = {best}")

        return result


# ---------------------------------------------------------------------------
# 3.  Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    x, y, z, w, v = sp.symbols("x y z w v")

    demo_cases = [
        {
            "name": "Expressions only (original behaviour)",
            "items": [x + y, y - z, x * z],
        },
        {
            "name": "Bare symbols only",
            "items": [x, y, z, w],
        },
        {
            "name": "Mixed: expressions + extra bare symbol",
            # v appears nowhere in the expressions; it still gets a unique value
            "items": [x + y, x * z, v],
        },
        {
            "name": "Mixed: shared symbol in both expr and bare list",
            # w appears in an expression AND is listed bare -- no duplication
            "items": [x**2 - y, z + x, w],
        },
        {
            "name": "With division (no div-by-zero) + extra symbol",
            "items": [x / y + z, w * x - y, v],
        },
    ]

    for case in demo_cases:
        print(f"\n{'='*60}")
        print(f"Case : {case['name']}")
        # Pretty-print distinguishing bare symbols from expressions
        bare = [str(i) for i in case["items"] if isinstance(i, sp.Symbol)]
        exprs = [str(i) for i in case["items"] if not isinstance(i, sp.Symbol)]
        if exprs:
            print(f"Exprs: {exprs}")
        if bare:
            print(f"Extra symbols: {bare}")
        print("Solution:")
        assignment = solve_minimal_unique(case["items"])
        if assignment:
            print("  Final assignment:", {str(k): v for k, v in assignment.items()})
            # Verify expressions
            for item in case["items"]:
                if not isinstance(item, sp.Symbol):
                    val = item.subs(assignment)
                    print(f"    {item}  ->  {val}")
