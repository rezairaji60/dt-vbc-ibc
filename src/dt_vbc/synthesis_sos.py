"""
Stable collocation-based surrogate synthesis for DT-VBC / IBC experiments.

Author: Reza Iraji
Date:   March 2026
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np

from dt_vbc.polynomials import (
    eval_poly_2d,
    evaluate_monomials_2d,
    monomial_exponents_2d,
)


Array = np.ndarray


@dataclass
class BuiltProblem:
    problem: cp.Problem
    variables: Dict[str, cp.Variable]
    metadata: Dict[str, object]


def _box_boundary_points(box: Tuple[Tuple[float, float], Tuple[float, float]], n: int) -> Array:
    (xlo, xhi), (ylo, yhi) = box
    xs = np.linspace(xlo, xhi, n)
    ys = np.linspace(ylo, yhi, n)

    pts = []
    for x in xs:
        pts.append([x, ylo])
        pts.append([x, yhi])
    for y in ys[1:-1]:
        pts.append([xlo, y])
        pts.append([xhi, y])
    return np.asarray(pts, dtype=float)


def _interior_grid(box: Tuple[Tuple[float, float], Tuple[float, float]], n: int) -> Array:
    (xlo, xhi), (ylo, yhi) = box
    xs = np.linspace(xlo, xhi, n)
    ys = np.linspace(ylo, yhi, n)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()])


def _sample_sets(
    domain: Tuple[Tuple[float, float], Tuple[float, float]],
    x0_box: Tuple[Tuple[float, float], Tuple[float, float]],
    xu_box: Tuple[Tuple[float, float], Tuple[float, float]],
    grid_n_domain: int,
    grid_n_boundary: int,
) -> Dict[str, Array]:
    return {
        "domain": _interior_grid(domain, grid_n_domain),
        "x0": _box_boundary_points(x0_box, grid_n_boundary),
        "xu": _box_boundary_points(xu_box, grid_n_boundary),
    }


def _normalization_constraints(coeff: cp.Variable, exponents: Sequence[Tuple[int, int]]) -> List[cp.Constraint]:
    """
    Normalize quadratic templates to avoid arbitrary scaling.

    Preferred normalization:
      coeff(x1^2) + coeff(x2^2) == 1
    and constant term <= 0.5 for numerical stability.
    """
    idx_x2 = None
    idx_y2 = None
    idx_const = None
    for i, e in enumerate(exponents):
        if e == (0, 0):
            idx_const = i
        elif e == (2, 0):
            idx_x2 = i
        elif e == (0, 2):
            idx_y2 = i

    cons: List[cp.Constraint] = []
    if idx_x2 is not None and idx_y2 is not None:
        cons.append(coeff[idx_x2] + coeff[idx_y2] == 1.0)
        cons.append(coeff[idx_x2] >= 0.05)
        cons.append(coeff[idx_y2] >= 0.05)
    if idx_const is not None:
        cons.append(coeff[idx_const] <= 0.5)
        cons.append(coeff[idx_const] >= -2.0)
    return cons


def _evaluate_affine_poly(Phi: Array, coeff: cp.Variable) -> cp.Expression:
    return Phi @ coeff


def _bisection_search(
    build_fn,
    eps_lo: float,
    eps_hi: float,
    steps: int,
    solver: str,
    solver_kwargs: Optional[Dict] = None,
):
    solver_kwargs = dict(solver_kwargs or {})
    best = None
    lo = eps_lo
    hi = eps_hi

    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        built = build_fn(mid)
        try:
            built.problem.solve(solver=solver, **solver_kwargs)
        except Exception:
            hi = mid
            continue

        status = built.problem.status
        feasible = status in ("optimal", "optimal_inaccurate")
        if feasible:
            lo = mid
            best = built
        else:
            hi = mid

    return best, lo


def solve_forward_dt_vbc(
    dynamics,
    domain,
    x0_box,
    xu_box,
    degree: int,
    n_components: int,
    comparison_matrix: Array,
    grid_n_domain: int = 25,
    grid_n_boundary: int = 10,
    eps_hi: float = 0.1,
    bisection_steps: int = 10,
    solver: str = "SCS",
    solver_kwargs: Optional[Dict] = None,
) -> Dict[str, object]:
    exponents = monomial_exponents_2d(degree)
    samples = _sample_sets(domain, x0_box, xu_box, grid_n_domain, grid_n_boundary)

    Phi_domain = evaluate_monomials_2d(samples["domain"], exponents)
    Phi_x0 = evaluate_monomials_2d(samples["x0"], exponents)
    Phi_xu = evaluate_monomials_2d(samples["xu"], exponents)

    f_domain = dynamics(samples["domain"])
    Phi_f_domain = evaluate_monomials_2d(f_domain, exponents)

    A = np.asarray(comparison_matrix, dtype=float)
    assert A.shape == (n_components, n_components)

    def build_problem(eps_fixed: float) -> BuiltProblem:
        coeffs = [cp.Variable(len(exponents), name=f"fwdB{i}") for i in range(n_components)]
        constraints: List[cp.Constraint] = []

        for i in range(n_components):
            Bi_x0 = _evaluate_affine_poly(Phi_x0, coeffs[i])
            Bi_xu = _evaluate_affine_poly(Phi_xu, coeffs[i])
            Bi_dom = _evaluate_affine_poly(Phi_domain, coeffs[i])
            Bi_fdom = _evaluate_affine_poly(Phi_f_domain, coeffs[i])

            constraints += [Bi_x0 <= -eps_fixed]
            constraints += _normalization_constraints(coeffs[i], exponents)

            rhs = 0
            for j in range(n_components):
                rhs = rhs + A[i, j] * _evaluate_affine_poly(Phi_domain, coeffs[j])
            constraints += [Bi_fdom <= rhs - eps_fixed]

        # stronger sufficient unsafe separation: choose component 0
        constraints += [_evaluate_affine_poly(Phi_xu, coeffs[0]) >= eps_fixed]

        obj = cp.Minimize(0.0)
        problem = cp.Problem(obj, constraints)
        return BuiltProblem(
            problem=problem,
            variables={f"fwdB{i}": coeffs[i] for i in range(n_components)},
            metadata={"eps_fixed": eps_fixed, "A": A},
        )

    t0 = perf_counter()
    built, eps_star = _bisection_search(
        build_problem,
        eps_lo=0.0,
        eps_hi=eps_hi,
        steps=bisection_steps,
        solver=solver,
        solver_kwargs=solver_kwargs,
    )
    runtime = perf_counter() - t0

    if built is None:
        return {
            "status": "infeasible",
            "epsilon": 0.0,
            "coefficients": {},
            "runtime_sec": runtime,
            "A": A,
        }

    coeff_out = {
        name: np.asarray(var.value, dtype=float).reshape(-1)
        for name, var in built.variables.items()
    }
    return {
        "status": built.problem.status,
        "epsilon": float(eps_star),
        "coefficients": coeff_out,
        "runtime_sec": runtime,
        "A": A,
        "exponents": exponents,
    }


def solve_backward_dt_vbc(
    dynamics,
    domain,
    x0_box,
    xu_box,
    degree: int,
    n_components: int,
    comparison_matrix: Array,
    grid_n_domain: int = 25,
    grid_n_boundary: int = 10,
    eps_hi: float = 0.1,
    bisection_steps: int = 10,
    solver: str = "SCS",
    solver_kwargs: Optional[Dict] = None,
) -> Dict[str, object]:
    exponents = monomial_exponents_2d(degree)
    samples = _sample_sets(domain, x0_box, xu_box, grid_n_domain, grid_n_boundary)

    Phi_domain = evaluate_monomials_2d(samples["domain"], exponents)
    Phi_x0 = evaluate_monomials_2d(samples["x0"], exponents)
    Phi_xu = evaluate_monomials_2d(samples["xu"], exponents)

    f_domain = dynamics(samples["domain"])
    Phi_f_domain = evaluate_monomials_2d(f_domain, exponents)

    A = np.asarray(comparison_matrix, dtype=float)
    assert A.shape == (n_components, n_components)

    def build_problem(eps_fixed: float) -> BuiltProblem:
        coeffs = [cp.Variable(len(exponents), name=f"bwdB{i}") for i in range(n_components)]
        constraints: List[cp.Constraint] = []

        for i in range(n_components):
            Bi_xu = _evaluate_affine_poly(Phi_xu, coeffs[i])
            Bi_dom = _evaluate_affine_poly(Phi_domain, coeffs[i])
            Bi_fdom = _evaluate_affine_poly(Phi_f_domain, coeffs[i])

            constraints += [Bi_xu <= -eps_fixed]
            constraints += _normalization_constraints(coeffs[i], exponents)

            rhs = 0
            for j in range(n_components):
                rhs = rhs + A[i, j] * _evaluate_affine_poly(Phi_f_domain, coeffs[j])
            constraints += [Bi_dom <= rhs - eps_fixed]

        constraints += [_evaluate_affine_poly(Phi_x0, coeffs[0]) >= eps_fixed]

        obj = cp.Minimize(0.0)
        problem = cp.Problem(obj, constraints)
        return BuiltProblem(
            problem=problem,
            variables={f"bwdB{i}": coeffs[i] for i in range(n_components)},
            metadata={"eps_fixed": eps_fixed, "A": A},
        )

    t0 = perf_counter()
    built, eps_star = _bisection_search(
        build_problem,
        eps_lo=0.0,
        eps_hi=eps_hi,
        steps=bisection_steps,
        solver=solver,
        solver_kwargs=solver_kwargs,
    )
    runtime = perf_counter() - t0

    if built is None:
        return {
            "status": "infeasible",
            "epsilon": 0.0,
            "coefficients": {},
            "runtime_sec": runtime,
            "A": A,
        }

    coeff_out = {
        name: np.asarray(var.value, dtype=float).reshape(-1)
        for name, var in built.variables.items()
    }
    return {
        "status": built.problem.status,
        "epsilon": float(eps_star),
        "coefficients": coeff_out,
        "runtime_sec": runtime,
        "A": A,
        "exponents": exponents,
    }


def solve_forward_ibc(
    dynamics,
    domain,
    x0_box,
    xu_box,
    degree: int,
    n_frames: int,
    lambdas: Sequence[float],
    grid_n_domain: int = 25,
    grid_n_boundary: int = 10,
    eps_hi: float = 0.1,
    bisection_steps: int = 10,
    solver: str = "SCS",
    solver_kwargs: Optional[Dict] = None,
) -> Dict[str, object]:
    exponents = monomial_exponents_2d(degree)
    samples = _sample_sets(domain, x0_box, xu_box, grid_n_domain, grid_n_boundary)

    Phi_domain = evaluate_monomials_2d(samples["domain"], exponents)
    Phi_x0 = evaluate_monomials_2d(samples["x0"], exponents)
    Phi_xu = evaluate_monomials_2d(samples["xu"], exponents)

    f_domain = dynamics(samples["domain"])
    Phi_f_domain = evaluate_monomials_2d(f_domain, exponents)

    lambdas = list(lambdas)
    assert len(lambdas) == n_frames

    def build_problem(eps_fixed: float) -> BuiltProblem:
        coeffs = [cp.Variable(len(exponents), name=f"fwdI{i}") for i in range(n_frames)]
        constraints: List[cp.Constraint] = []

        constraints += [_evaluate_affine_poly(Phi_x0, coeffs[0]) <= -eps_fixed]

        for i in range(n_frames):
            constraints += _normalization_constraints(coeffs[i], exponents)
            constraints += [_evaluate_affine_poly(Phi_xu, coeffs[i]) >= eps_fixed]

        for i in range(n_frames - 1):
            lhs = lambdas[i] * _evaluate_affine_poly(Phi_f_domain, coeffs[i + 1]) - _evaluate_affine_poly(Phi_domain, coeffs[i])
            constraints += [lhs <= -eps_fixed]

        lhs_last = lambdas[-1] * _evaluate_affine_poly(Phi_f_domain, coeffs[-1]) - _evaluate_affine_poly(Phi_domain, coeffs[-1])
        constraints += [lhs_last <= -eps_fixed]

        problem = cp.Problem(cp.Minimize(0.0), constraints)
        return BuiltProblem(
            problem=problem,
            variables={f"fwdI{i}": coeffs[i] for i in range(n_frames)},
            metadata={"eps_fixed": eps_fixed, "lambdas": lambdas},
        )

    t0 = perf_counter()
    built, eps_star = _bisection_search(
        build_problem,
        eps_lo=0.0,
        eps_hi=eps_hi,
        steps=bisection_steps,
        solver=solver,
        solver_kwargs=solver_kwargs,
    )
    runtime = perf_counter() - t0

    if built is None:
        return {
            "status": "infeasible",
            "epsilon": 0.0,
            "coefficients": {},
            "runtime_sec": runtime,
            "lambdas": lambdas,
        }

    coeff_out = {
        name: np.asarray(var.value, dtype=float).reshape(-1)
        for name, var in built.variables.items()
    }
    return {
        "status": built.problem.status,
        "epsilon": float(eps_star),
        "coefficients": coeff_out,
        "runtime_sec": runtime,
        "lambdas": lambdas,
        "exponents": exponents,
    }


def solve_backward_ibc(
    dynamics,
    domain,
    x0_box,
    xu_box,
    degree: int,
    n_frames: int,
    lambdas: Sequence[float],
    grid_n_domain: int = 25,
    grid_n_boundary: int = 10,
    eps_hi: float = 0.1,
    bisection_steps: int = 10,
    solver: str = "SCS",
    solver_kwargs: Optional[Dict] = None,
) -> Dict[str, object]:
    exponents = monomial_exponents_2d(degree)
    samples = _sample_sets(domain, x0_box, xu_box, grid_n_domain, grid_n_boundary)

    Phi_domain = evaluate_monomials_2d(samples["domain"], exponents)
    Phi_x0 = evaluate_monomials_2d(samples["x0"], exponents)
    Phi_xu = evaluate_monomials_2d(samples["xu"], exponents)

    f_domain = dynamics(samples["domain"])
    Phi_f_domain = evaluate_monomials_2d(f_domain, exponents)

    lambdas = list(lambdas)
    assert len(lambdas) == n_frames

    def build_problem(eps_fixed: float) -> BuiltProblem:
        coeffs = [cp.Variable(len(exponents), name=f"bwdI{i}") for i in range(n_frames)]
        constraints: List[cp.Constraint] = []

        constraints += [_evaluate_affine_poly(Phi_xu, coeffs[0]) <= -eps_fixed]

        for i in range(n_frames):
            constraints += _normalization_constraints(coeffs[i], exponents)
            constraints += [_evaluate_affine_poly(Phi_x0, coeffs[i]) >= eps_fixed]

        for i in range(n_frames - 1):
            lhs = _evaluate_affine_poly(Phi_domain, coeffs[i + 1]) - lambdas[i] * _evaluate_affine_poly(Phi_f_domain, coeffs[i])
            constraints += [lhs <= -eps_fixed]

        lhs_last = _evaluate_affine_poly(Phi_domain, coeffs[-1]) - lambdas[-1] * _evaluate_affine_poly(Phi_f_domain, coeffs[-1])
        constraints += [lhs_last <= -eps_fixed]

        problem = cp.Problem(cp.Minimize(0.0), constraints)
        return BuiltProblem(
            problem=problem,
            variables={f"bwdI{i}": coeffs[i] for i in range(n_frames)},
            metadata={"eps_fixed": eps_fixed, "lambdas": lambdas},
        )

    t0 = perf_counter()
    built, eps_star = _bisection_search(
        build_problem,
        eps_lo=0.0,
        eps_hi=eps_hi,
        steps=bisection_steps,
        solver=solver,
        solver_kwargs=solver_kwargs,
    )
    runtime = perf_counter() - t0

    if built is None:
        return {
            "status": "infeasible",
            "epsilon": 0.0,
            "coefficients": {},
            "runtime_sec": runtime,
            "lambdas": lambdas,
        }

    coeff_out = {
        name: np.asarray(var.value, dtype=float).reshape(-1)
        for name, var in built.variables.items()
    }
    return {
        "status": built.problem.status,
        "epsilon": float(eps_star),
        "coefficients": coeff_out,
        "runtime_sec": runtime,
        "lambdas": lambdas,
        "exponents": exponents,
    }