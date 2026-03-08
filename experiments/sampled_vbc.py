from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def polynomial_basis(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    return np.column_stack([
        np.ones_like(x),
        x,
        y,
        x * x,
        x * y,
        y * y,
    ])


def fit_scalar_barrier_sampled(
    points: np.ndarray,
    next_points: np.ndarray,
    init_mask: np.ndarray,
    unsafe_mask: np.ndarray,
    alpha: float,
    eps: float = 0.05,
) -> np.ndarray:
    """Fit a sampled scalar certificate b such that b(f(x)) <= alpha b(x).

    This is a lightweight LP surrogate, not an exact SOS program.
    """
    Phi = polynomial_basis(points)
    Phi_next = polynomial_basis(next_points)
    n = Phi.shape[1]

    A_ub = []
    b_ub = []

    # init: b(x) <= -eps
    for row in Phi[init_mask]:
        A_ub.append(row)
        b_ub.append(-eps)

    # unsafe: b(x) >= eps  ->  -b(x) <= -eps
    for row in Phi[unsafe_mask]:
        A_ub.append(-row)
        b_ub.append(-eps)

    # propagation: b(f(x)) - alpha b(x) <= 0
    for row_n, row in zip(Phi_next, Phi):
        A_ub.append(row_n - alpha * row)
        b_ub.append(0.0)

    # normalize at origin-like center sample near first point to avoid trivial zero
    A_eq = [Phi[0]]
    b_eq = [-1.0]

    c = np.zeros(n)
    res = linprog(
        c,
        A_ub=np.asarray(A_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray(A_eq),
        b_eq=np.asarray(b_eq),
        bounds=[(None, None)] * n,
        method='highs',
    )
    if not res.success:
        raise RuntimeError(f'LP fit failed: {res.message}')
    return res.x


def eval_scalar(coeffs: np.ndarray, points: np.ndarray) -> np.ndarray:
    return polynomial_basis(points) @ coeffs
