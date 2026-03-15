"""
Author: Reza Iraji
Date:   March 2026

CVXPY + SymPy skeleton for SOS modeling.

1. Create Gram-matrix SOS polynomials,
2. Match coefficients with SymPy,
3. Export CVXPY constraints.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import cvxpy as cp
import sympy as sp
from .poly_basis import monomials


@dataclass
class SOSPoly:
    expr: sp.Expr
    gram: cp.Variable
    z: List[sp.Expr]


def gram_sos(vars_: Sequence[sp.Symbol], degree: int, prefix: str) -> SOSPoly:
    if degree % 2 != 0:
        raise ValueError("SOS degree must be even")
    z = monomials(vars_, degree // 2)
    Q = cp.Variable((len(z), len(z)), PSD=True, name=f"Q_{prefix}")
    expr = 0
    for i, zi in enumerate(z):
        for j, zj in enumerate(z):
            expr += sp.Symbol(f"__cvx_{prefix}_{i}_{j}") * zi * zj
    return SOSPoly(expr=sp.expand(expr), gram=Q, z=z)


def cvx_symbol_map(prefix: str, Q: cp.Variable) -> Dict[sp.Symbol, cp.Expression]:
    mp = {}
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            mp[sp.Symbol(f"__cvx_{prefix}_{i}_{j}")] = Q[i, j]
    return mp


def poly_coeff_map(expr: sp.Expr, vars_: Sequence[sp.Symbol]) -> Dict[Tuple[int, ...], sp.Expr]:
    poly = sp.Poly(sp.expand(expr), *vars_)
    out = {}
    for mon, coeff in poly.terms():
        out[mon] = coeff
    return out


def coefficient_matching_constraints(
    target: sp.Expr,
    sos_poly: SOSPoly,
    vars_: Sequence[sp.Symbol],
    affine_symbol_map: Dict[sp.Symbol, cp.Expression],
    prefix: str,
) -> List[cp.Constraint]:
    target_map = poly_coeff_map(target, vars_)
    sos_map = poly_coeff_map(sos_poly.expr, vars_)
    gram_map = cvx_symbol_map(prefix, sos_poly.gram)
    constraints: List[cp.Constraint] = []
    all_keys = set(target_map) | set(sos_map)
    for key in all_keys:
        lhs = target_map.get(key, 0)
        rhs = sos_map.get(key, 0)
        rhs_cvx = 0
        rhs_free = sp.expand(rhs)
        for sym, expr in gram_map.items():
            coeff = rhs_free.coeff(sym)
            if coeff != 0:
                rhs_cvx += float(coeff) * expr
                rhs_free -= coeff * sym
        lhs_cvx = 0
        lhs_free = sp.expand(lhs)
        for sym, expr in affine_symbol_map.items():
            coeff = lhs_free.coeff(sym)
            if coeff != 0:
                lhs_cvx += float(coeff) * expr
                lhs_free -= coeff * sym
        if sp.expand(lhs_free) != 0 or sp.expand(rhs_free) != 0:
            raise ValueError(
                "Non-affine symbolic remainder encountered. "
                "Use decision symbols only linearly in target polynomial."
            )
        constraints.append(lhs_cvx == rhs_cvx)
    return constraints
