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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np

from .polynomials import evaluate_monomials_2d, monomial_exponents_2d


@dataclass
class SynthesisResult:
    formulation: str
    status: str
    feasible: bool
    epsilon: float | None
    degree: int
    n_functions: int
    comparison_matrix: np.ndarray | None
    coefficients: Dict[str, np.ndarray]
    notes: str


def _build_design(
    domain_pts: np.ndarray,
    x0_pts: np.ndarray,
    xu_pts: np.ndarray,
    degree: int,
) -> Dict[str, np.ndarray]:
    exps = monomial_exponents_2d(degree)
    return {
        "exponents": np.array(exps, dtype=int),
        "Phi_domain": evaluate_monomials_2d(domain_pts, exps),
        "Phi_x0": evaluate_monomials_2d(x0_pts, exps),
        "Phi_xu": evaluate_monomials_2d(xu_pts, exps),
    }


def _make_poly_vars(n_functions: int, n_monomials: int, prefix: str) -> Dict[str, cp.Variable]:
    return {f"{prefix}{i}": cp.Variable(n_monomials) for i in range(n_functions)}


def solve_forward_dt_vbc_collocation(
    domain_pts: np.ndarray,
    image_pts: np.ndarray,
    x0_pts: np.ndarray,
    xu_pts: np.ndarray,
    degree: int,
    comparison_matrix: np.ndarray,
    epsilon_upper: float = 1.0,
) -> SynthesisResult:
    design = _build_design(domain_pts, x0_pts, xu_pts, degree)
    exps = design["exponents"]
    n_monomials = len(exps)
    m = comparison_matrix.shape[0]
    coeffs = _make_poly_vars(m, n_monomials, "fwdB")

    eps = cp.Variable(nonneg=True)
    constraints = [eps <= epsilon_upper]

    B_domain = []
    B_image = []
    B_x0 = []
    B_xu = []

    Phi_domain = design["Phi_domain"]
    Phi_x0 = design["Phi_x0"]
    Phi_xu = design["Phi_xu"]
    Phi_image = evaluate_monomials_2d(image_pts, [tuple(e) for e in exps])

    for i in range(m):
        c = coeffs[f"fwdB{i}"]
        B_domain.append(Phi_domain @ c)
        B_image.append(Phi_image @ c)
        B_x0.append(Phi_x0 @ c)
        B_xu.append(Phi_xu @ c)

        constraints.append(B_x0[i] <= -eps)
        constraints.append(B_xu[i] >= eps)

    for i in range(m):
        rhs = 0
        for j in range(m):
            rhs = rhs + comparison_matrix[i, j] * B_domain[j]
        constraints.append(B_image[i] <= rhs - eps)

    objective = cp.Maximize(eps)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    feasible = problem.status in {"optimal", "optimal_inaccurate"} and eps.value is not None
    out = {
        name: np.array(var.value).reshape(-1) if var.value is not None else np.zeros(n_monomials)
        for name, var in coeffs.items()
    }
    return SynthesisResult(
        formulation="Forward DT-VBC",
        status=problem.status,
        feasible=feasible,
        epsilon=float(eps.value) if eps.value is not None else None,
        degree=degree,
        n_functions=m,
        comparison_matrix=comparison_matrix.copy(),
        coefficients=out,
        notes="Collocation-based surrogate search with fixed comparison matrix.",
    )


def solve_backward_dt_vbc_collocation(
    domain_pts: np.ndarray,
    image_pts: np.ndarray,
    x0_pts: np.ndarray,
    xu_pts: np.ndarray,
    degree: int,
    comparison_matrix: np.ndarray,
    epsilon_upper: float = 1.0,
) -> SynthesisResult:
    design = _build_design(domain_pts, x0_pts, xu_pts, degree)
    exps = design["exponents"]
    n_monomials = len(exps)
    m = comparison_matrix.shape[0]
    coeffs = _make_poly_vars(m, n_monomials, "bwdB")

    eps = cp.Variable(nonneg=True)
    constraints = [eps <= epsilon_upper]

    B_domain = []
    B_image = []
    B_x0 = []
    B_xu = []

    Phi_domain = design["Phi_domain"]
    Phi_x0 = design["Phi_x0"]
    Phi_xu = design["Phi_xu"]
    Phi_image = evaluate_monomials_2d(image_pts, [tuple(e) for e in exps])

    for i in range(m):
        c = coeffs[f"bwdB{i}"]
        B_domain.append(Phi_domain @ c)
        B_image.append(Phi_image @ c)
        B_x0.append(Phi_x0 @ c)
        B_xu.append(Phi_xu @ c)

        constraints.append(B_xu[i] <= -eps)
        constraints.append(B_x0[i] >= eps)

    for i in range(m):
        rhs = 0
        for j in range(m):
            rhs = rhs + comparison_matrix[i, j] * B_image[j]
        constraints.append(B_domain[i] <= rhs - eps)

    objective = cp.Maximize(eps)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    feasible = problem.status in {"optimal", "optimal_inaccurate"} and eps.value is not None
    out = {
        name: np.array(var.value).reshape(-1) if var.value is not None else np.zeros(n_monomials)
        for name, var in coeffs.items()
    }
    return SynthesisResult(
        formulation="Backward DT-VBC",
        status=problem.status,
        feasible=feasible,
        epsilon=float(eps.value) if eps.value is not None else None,
        degree=degree,
        n_functions=m,
        comparison_matrix=comparison_matrix.copy(),
        coefficients=out,
        notes="Collocation-based surrogate search with fixed comparison matrix.",
    )


def solve_forward_ibc_collocation(
    domain_pts: np.ndarray,
    image_pts: np.ndarray,
    x0_pts: np.ndarray,
    xu_pts: np.ndarray,
    degree: int,
    k: int,
    lambdas: Sequence[float],
    epsilon_upper: float = 1.0,
) -> SynthesisResult:
    design = _build_design(domain_pts, x0_pts, xu_pts, degree)
    exps = design["exponents"]
    n_monomials = len(exps)
    n_frames = k + 1
    coeffs = _make_poly_vars(n_frames, n_monomials, "fwdibc")

    eps = cp.Variable(nonneg=True)
    constraints = [eps <= epsilon_upper]

    Phi_domain = design["Phi_domain"]
    Phi_x0 = design["Phi_x0"]
    Phi_xu = design["Phi_xu"]
    Phi_image = evaluate_monomials_2d(image_pts, [tuple(e) for e in exps])

    B_domain = []
    B_image = []
    B_x0 = []
    B_xu = []

    for i in range(n_frames):
        c = coeffs[f"fwdibc{i}"]
        B_domain.append(Phi_domain @ c)
        B_image.append(Phi_image @ c)
        B_x0.append(Phi_x0 @ c)
        B_xu.append(Phi_xu @ c)
        constraints.append(B_xu[i] >= eps)

    constraints.append(B_x0[0] <= -eps)

    for i in range(k):
        constraints.append(lambdas[i] * B_image[i + 1] - B_domain[i] <= -eps)
    constraints.append(lambdas[k] * B_image[k] - B_domain[k] <= -eps)

    objective = cp.Maximize(eps)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    feasible = problem.status in {"optimal", "optimal_inaccurate"} and eps.value is not None
    out = {
        name: np.array(var.value).reshape(-1) if var.value is not None else np.zeros(n_monomials)
        for name, var in coeffs.items()
    }
    return SynthesisResult(
        formulation="Forward IBC",
        status=problem.status,
        feasible=feasible,
        epsilon=float(eps.value) if eps.value is not None else None,
        degree=degree,
        n_functions=n_frames,
        comparison_matrix=None,
        coefficients=out,
        notes="Collocation-based surrogate search with fixed positive scalings.",
    )


def solve_backward_ibc_collocation(
    domain_pts: np.ndarray,
    image_pts: np.ndarray,
    x0_pts: np.ndarray,
    xu_pts: np.ndarray,
    degree: int,
    k: int,
    lambdas: Sequence[float],
    epsilon_upper: float = 1.0,
) -> SynthesisResult:
    design = _build_design(domain_pts, x0_pts, xu_pts, degree)
    exps = design["exponents"]
    n_monomials = len(exps)
    n_frames = k + 1
    coeffs = _make_poly_vars(n_frames, n_monomials, "bwdibc")

    eps = cp.Variable(nonneg=True)
    constraints = [eps <= epsilon_upper]

    Phi_domain = design["Phi_domain"]
    Phi_x0 = design["Phi_x0"]
    Phi_xu = design["Phi_xu"]
    Phi_image = evaluate_monomials_2d(image_pts, [tuple(e) for e in exps])

    B_domain = []
    B_image = []
    B_x0 = []
    B_xu = []

    for i in range(n_frames):
        c = coeffs[f"bwdibc{i}"]
        B_domain.append(Phi_domain @ c)
        B_image.append(Phi_image @ c)
        B_x0.append(Phi_x0 @ c)
        B_xu.append(Phi_xu @ c)
        constraints.append(B_x0[i] >= eps)

    constraints.append(B_xu[0] <= -eps)

    for i in range(k):
        constraints.append(B_domain[i + 1] - lambdas[i] * B_image[i] <= -eps)
    constraints.append(B_domain[k] - lambdas[k] * B_image[k] <= -eps)

    objective = cp.Maximize(eps)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    feasible = problem.status in {"optimal", "optimal_inaccurate"} and eps.value is not None
    out = {
        name: np.array(var.value).reshape(-1) if var.value is not None else np.zeros(n_monomials)
        for name, var in coeffs.items()
    }
    return SynthesisResult(
        formulation="Backward IBC",
        status=problem.status,
        feasible=feasible,
        epsilon=float(eps.value) if eps.value is not None else None,
        degree=degree,
        n_functions=n_frames,
        comparison_matrix=None,
        coefficients=out,
        notes="Collocation-based surrogate search with fixed positive scalings.",
    )