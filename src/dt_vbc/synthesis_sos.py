"""
Author: Reza Iraji
Date:   March 2026

SOS synthesis for DT-VBC / IBC with fixed comparison parameters.

Important modeling choice:
- To keep each synthesis problem convex, the comparison matrix A and the IBC
  scaling coefficients lambda_i are treated as fixed outer-loop parameters.
- The SDP then searches only for polynomial certificate coefficients,
  SOS multipliers, and the margin epsilon.
- A finite outer search over candidate A / lambda values is implemented in
  experiments/run_all_sos.py.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Any
import cvxpy as cp
import numpy as np
import sympy as sp
from .poly_basis import PolyTemplate
from .sos_utils import gram_sos, coefficient_matching_constraints, cvx_symbol_map

@dataclass
class SemialgebraicSet:
    polys: Sequence[sp.Expr]

@dataclass
class BuiltSOSProblem:
    problem: cp.Problem
    coeff_vars: Dict[str, cp.Variable]
    eps: cp.Variable
    metadata: Dict[str, Any]


def _build_poly_templates(vars_: Tuple[sp.Symbol, ...], count: int, degree: int, prefix: str):
    exprs = []
    coeff_vars: Dict[str, cp.Variable] = {}
    affine_map: Dict[sp.Symbol, cp.Expression] = {}
    for i in range(count):
        poly, syms, _ = PolyTemplate(vars_, degree, f"{prefix}{i}_").build()
        exprs.append(poly)
        for s in syms:
            v = cp.Variable(name=str(s))
            coeff_vars[str(s)] = v
            affine_map[s] = v
    return exprs, coeff_vars, affine_map


def _sigma_terms(vars_, set_polys, prefix: str):
    sigmas = []
    for j, g in enumerate(set_polys):
        s = gram_sos(vars_, degree=2, prefix=f"{prefix}_{j}")
        sigmas.append((s, g, f"{prefix}_{j}"))
    return sigmas


def _constraints_for_target(target, vars_, affine_map, prefix: str, degree_hint: int, sigma_terms):
    s_main = gram_sos(vars_, degree=degree_hint if degree_hint % 2 == 0 else degree_hint + 1, prefix=prefix)
    local_map = dict(affine_map)
    for s, _, sprefix in sigma_terms:
        local_map.update(cvx_symbol_map(sprefix, s.gram))
    cons = coefficient_matching_constraints(sp.expand(target), s_main, vars_, local_map, prefix)
    cons += [s_main.gram >> 0]
    for s, _, _ in sigma_terms:
        cons += [s.gram >> 0]
    return cons


def build_forward_dt_vbc_problem(f_exprs, vars_, X, X0, Xu, degree, A_value, m=2, unsafe_component=0, solver='SCS'):
    B_exprs, coeff_vars, affine_map = _build_poly_templates(vars_, m, degree, 'fwdB')
    eps = cp.Variable(nonneg=True, name='eps')
    affine_map[sp.Symbol('eps')] = eps
    constraints: List[cp.Constraint] = []

    for i in range(m):
        sigmas = _sigma_terms(vars_, X0.polys, f'sig0_{i}')
        target = -B_exprs[i] - sum(s.expr * g for s, g, _ in sigmas)
        constraints += _constraints_for_target(target, vars_, affine_map, f'init_{i}', max(2, degree + 2), sigmas)

    q = unsafe_component
    sigmas = _sigma_terms(vars_, Xu.polys, f'sigu_{q}')
    target = B_exprs[q] - sp.Symbol('eps') - sum(s.expr * g for s, g, _ in sigmas)
    constraints += _constraints_for_target(target, vars_, affine_map, 'unsafe', max(2, degree + 2), sigmas)

    Bf = [sp.expand(b.subs(dict(zip(vars_, f_exprs)))) for b in B_exprs]
    for i in range(m):
        sigmas = _sigma_terms(vars_, X.polys, f'sigx_{i}')
        target = -Bf[i] + sum(float(A_value[i, j]) * B_exprs[j] for j in range(m)) - sum(s.expr * g for s, g, _ in sigmas)
        constraints += _constraints_for_target(target, vars_, affine_map, f'prop_{i}', max(2, degree + 4), sigmas)

    return BuiltSOSProblem(cp.Problem(cp.Maximize(eps), constraints), coeff_vars, eps,
                           {'formulation': 'Forward DT-VBC', 'degree': degree, 'm': m, 'A': np.asarray(A_value).tolist(), 'solver': solver})


def build_backward_dt_vbc_problem(f_exprs, vars_, X, X0, Xu, degree, A_value, m=2, init_component=0, solver='SCS'):
    B_exprs, coeff_vars, affine_map = _build_poly_templates(vars_, m, degree, 'bwdB')
    eps = cp.Variable(nonneg=True, name='eps')
    affine_map[sp.Symbol('eps')] = eps
    constraints: List[cp.Constraint] = []

    for i in range(m):
        sigmas = _sigma_terms(vars_, Xu.polys, f'sigu_{i}')
        target = -B_exprs[i] - sum(s.expr * g for s, g, _ in sigmas)
        constraints += _constraints_for_target(target, vars_, affine_map, f'unsafe_{i}', max(2, degree + 2), sigmas)

    q = init_component
    sigmas = _sigma_terms(vars_, X0.polys, f'sig0_{q}')
    target = B_exprs[q] - sp.Symbol('eps') - sum(s.expr * g for s, g, _ in sigmas)
    constraints += _constraints_for_target(target, vars_, affine_map, 'init_sep', max(2, degree + 2), sigmas)

    Bf = [sp.expand(b.subs(dict(zip(vars_, f_exprs)))) for b in B_exprs]
    for i in range(m):
        sigmas = _sigma_terms(vars_, X.polys, f'sigx_{i}')
        target = -B_exprs[i] + sum(float(A_value[i, j]) * Bf[j] for j in range(m)) - sum(s.expr * g for s, g, _ in sigmas)
        constraints += _constraints_for_target(target, vars_, affine_map, f'prop_{i}', max(2, degree + 4), sigmas)

    return BuiltSOSProblem(cp.Problem(cp.Maximize(eps), constraints), coeff_vars, eps,
                           {'formulation': 'Backward DT-VBC', 'degree': degree, 'm': m, 'A': np.asarray(A_value).tolist(), 'solver': solver})


def build_forward_ibc_problem(f_exprs, vars_, X, X0, Xu, degree, lambdas, k=2, solver='SCS'):
    frames, coeff_vars, affine_map = _build_poly_templates(vars_, k + 1, degree, 'fibc')
    eps = cp.Variable(nonneg=True, name='eps')
    affine_map[sp.Symbol('eps')] = eps
    constraints: List[cp.Constraint] = []

    sigmas = _sigma_terms(vars_, X0.polys, 'sig0')
    target = -frames[0] - sum(s.expr * g for s, g, _ in sigmas)
    constraints += _constraints_for_target(target, vars_, affine_map, 'init', max(2, degree + 2), sigmas)

    for i in range(k + 1):
        sigmas = _sigma_terms(vars_, Xu.polys, f'sigu_{i}')
        target = frames[i] - sp.Symbol('eps') - sum(s.expr * g for s, g, _ in sigmas)
        constraints += _constraints_for_target(target, vars_, affine_map, f'unsafe_{i}', max(2, degree + 2), sigmas)

    for i in range(k):
        sigmas = _sigma_terms(vars_, X.polys, f'sigx_{i}')
        target = float(lambdas[i]) * sp.expand(frames[i + 1].subs(dict(zip(vars_, f_exprs)))) - frames[i] - sum(s.expr * g for s, g, _ in sigmas)
        constraints += _constraints_for_target(target, vars_, affine_map, f'prop_{i}', max(2, degree + 4), sigmas)
    sigmas = _sigma_terms(vars_, X.polys, f'sigx_{k}')
    target = float(lambdas[k]) * sp.expand(frames[k].subs(dict(zip(vars_, f_exprs)))) - frames[k] - sum(s.expr * g for s, g, _ in sigmas)
    constraints += _constraints_for_target(target, vars_, affine_map, f'prop_{k}', max(2, degree + 4), sigmas)

    return BuiltSOSProblem(cp.Problem(cp.Maximize(eps), constraints), coeff_vars, eps,
                           {'formulation': 'Forward IBC', 'degree': degree, 'k': k, 'lambdas': list(map(float, lambdas)), 'solver': solver})


def build_backward_ibc_problem(f_exprs, vars_, X, X0, Xu, degree, lambdas, k=2, solver='SCS'):
    frames, coeff_vars, affine_map = _build_poly_templates(vars_, k + 1, degree, 'bibc')
    eps = cp.Variable(nonneg=True, name='eps')
    affine_map[sp.Symbol('eps')] = eps
    constraints: List[cp.Constraint] = []

    sigmas = _sigma_terms(vars_, Xu.polys, 'sigu0')
    target = -frames[0] - sum(s.expr * g for s, g, _ in sigmas)
    constraints += _constraints_for_target(target, vars_, affine_map, 'unsafe0', max(2, degree + 2), sigmas)

    for i in range(k + 1):
        sigmas = _sigma_terms(vars_, X0.polys, f'sig0_{i}')
        target = frames[i] - sp.Symbol('eps') - sum(s.expr * g for s, g, _ in sigmas)
        constraints += _constraints_for_target(target, vars_, affine_map, f'init_{i}', max(2, degree + 2), sigmas)

    for i in range(k):
        sigmas = _sigma_terms(vars_, X.polys, f'sigx_{i}')
        target = frames[i + 1] - float(lambdas[i]) * sp.expand(frames[i].subs(dict(zip(vars_, f_exprs)))) - sum(s.expr * g for s, g, _ in sigmas)
        constraints += _constraints_for_target(target, vars_, affine_map, f'prop_{i}', max(2, degree + 4), sigmas)
    sigmas = _sigma_terms(vars_, X.polys, f'sigx_{k}')
    target = frames[k] - float(lambdas[k]) * sp.expand(frames[k].subs(dict(zip(vars_, f_exprs)))) - sum(s.expr * g for s, g, _ in sigmas)
    constraints += _constraints_for_target(target, vars_, affine_map, f'prop_{k}', max(2, degree + 4), sigmas)

    return BuiltSOSProblem(cp.Problem(cp.Maximize(eps), constraints), coeff_vars, eps,
                           {'formulation': 'Backward IBC', 'degree': degree, 'k': k, 'lambdas': list(map(float, lambdas)), 'solver': solver})


def solve_built_problem(built: BuiltSOSProblem, solver='SCS', verbose=False, max_iters=30000):
    kwargs: Dict[str, Any] = {'verbose': verbose}
    if solver.upper() == 'SCS':
        kwargs.update({'eps': 1e-5, 'max_iters': max_iters})
    built.problem.solve(solver=solver, **kwargs)
    coeffs = {k: float(v.value) for k, v in built.coeff_vars.items() if v.value is not None}
    return {
        **built.metadata,
        'status': built.problem.status,
        'objective': None if built.problem.value is None else float(built.problem.value),
        'epsilon': None if built.eps.value is None else float(built.eps.value),
        'coefficients': coeffs,
    }