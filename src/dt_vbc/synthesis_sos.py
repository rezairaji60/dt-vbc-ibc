"""
Author: Reza Iraji
Date:   March 2026

SOS synthesis skeleton for DT-VBC / IBC.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import cvxpy as cp
import sympy as sp
from .poly_basis import PolyTemplate
from .sos_utils import gram_sos, coefficient_matching_constraints


@dataclass
class SemialgebraicSet:
    polys: Sequence[sp.Expr]  # interpreted as g_i(x) >= 0


@dataclass
class SOSResult:
    status: str
    objective: float | None
    metadata: Dict


class ForwardDTVBCSOS:
    def __init__(self, f_exprs, vars_, X, X0, Xu, m=2, degree=2):
        self.f_exprs = tuple(f_exprs)
        self.vars_ = tuple(vars_)
        self.X = X
        self.X0 = X0
        self.Xu = Xu
        self.m = m
        self.degree = degree

    def build_problem(self):
        x = self.vars_
        B_exprs = []
        coeff_symbols = []
        coeff_vars = {}
        for i in range(self.m):
            poly, syms, _ = PolyTemplate(x, self.degree, f"b{i}_").build()
            B_exprs.append(poly)
            for s in syms:
                coeff_symbols.append(s)
                coeff_vars[s] = cp.Variable(name=str(s))

        eps = cp.Variable(nonneg=True, name="eps")
        A = cp.Variable((self.m, self.m), nonneg=True, name="A")

        constraints: List[cp.Constraint] = []
        affine_map = dict(coeff_vars)
        affine_map[sp.Symbol("eps")] = eps

        # Forward initial-set conditions:
        # -B_i(x) - sigma_0i(x)^T g0(x) is SOS
        for i in range(self.m):
            sigma_terms = []
            for j, g in enumerate(self.X0.polys):
                s = gram_sos(x, degree=2, prefix=f"sig0_{i}_{j}")
                sigma_terms.append((s, g, f"sig0_{i}_{j}"))
            target = -B_exprs[i]
            for s, g, _ in sigma_terms:
                target -= s.expr * g
            target = sp.expand(target)
            s_main = gram_sos(x, degree=max(2, self.degree + 2), prefix=f"init_{i}")
            for s, _, prefix in sigma_terms:
                constraints += coefficient_matching_constraints(sp.Integer(0), s, x, {}, prefix)
                constraints += [s.gram >> 0]
            constraints += coefficient_matching_constraints(target, s_main, x, affine_map, f"init_{i}")
            constraints += [s_main.gram >> 0]

        # Unsafe separation (stronger sufficient condition on one chosen index q):
        # B_q(x) - eps - sigma_u(x)^T g_u(x) is SOS
        q = 0
        sigma_terms = []
        for j, g in enumerate(self.Xu.polys):
            s = gram_sos(x, degree=2, prefix=f"sigu_{q}_{j}")
            sigma_terms.append((s, g, f"sigu_{q}_{j}"))
        target = B_exprs[q] - sp.Symbol("eps")
        for s, g, _ in sigma_terms:
            target -= s.expr * g
        s_main = gram_sos(x, degree=max(2, self.degree + 2), prefix="unsafe")
        for s, _, prefix in sigma_terms:
            constraints += coefficient_matching_constraints(sp.Integer(0), s, x, {}, prefix)
            constraints += [s.gram >> 0]
        constraints += coefficient_matching_constraints(sp.expand(target), s_main, x, affine_map, "unsafe")
        constraints += [s_main.gram >> 0]

        # Propagation:
        # (A B(x) - B(f(x)))_i - sigma_i(x)^T g(x) is SOS
        Bf = [sp.expand(b.subs(dict(zip(x, self.f_exprs)))) for b in B_exprs]
        for i in range(self.m):
            affine_target = -Bf[i]
            for j in range(self.m):
                affine_target += sp.Symbol(f"A_{i}_{j}") * B_exprs[j]
            local_affine = dict(affine_map)
            for ii in range(self.m):
                for jj in range(self.m):
                    local_affine[sp.Symbol(f"A_{ii}_{jj}")] = A[ii, jj]
            sigma_terms = []
            for j, g in enumerate(self.X.polys):
                s = gram_sos(x, degree=2, prefix=f"sigx_{i}_{j}")
                sigma_terms.append((s, g, f"sigx_{i}_{j}"))
            for s, g, _ in sigma_terms:
                affine_target -= s.expr * g
            s_main = gram_sos(x, degree=max(2, self.degree + 3), prefix=f"prop_{i}")
            for s, _, prefix in sigma_terms:
                constraints += coefficient_matching_constraints(sp.Integer(0), s, x, {}, prefix)
                constraints += [s.gram >> 0]
            constraints += coefficient_matching_constraints(sp.expand(affine_target), s_main, x, local_affine, f"prop_{i}")
            constraints += [s_main.gram >> 0]

        prob = cp.Problem(cp.Maximize(eps), constraints)
        return prob


# TODO:
# - BackwardDTVBCSOS
# - ForwardIBCSOS
# - BackwardIBCSOS
# Each follows the same pattern:
#   * polynomial templates for certificate components / frames,
#   * positive scalars lambda_i where needed,
#   * SOS multipliers per semialgebraic constraint,
#   * coefficient matching via SymPy.
