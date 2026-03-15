"""
Author: Reza Iraji
Date:   March 2026
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


Box = Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass(frozen=True)
class PolynomialSystem2D:
    name: str
    vars: Tuple[str, str]
    domain: Box
    x0_box: Box
    xu_box: Box
    dynamics: Callable[[np.ndarray], np.ndarray]
    description: str


def system_s1_dynamics(x: np.ndarray) -> np.ndarray:
    x1 = x[..., 0]
    x2 = x[..., 1]
    y1 = 0.72 * x1 + 0.10 * x2 - 0.12 * (x1 ** 3)
    y2 = -0.08 * x1 + 0.68 * x2 - 0.08 * (x2 ** 3)
    return np.stack([y1, y2], axis=-1)


def system_s2_dynamics(x: np.ndarray) -> np.ndarray:
    x1 = x[..., 0]
    x2 = x[..., 1]
    y1 = 0.60 * x1 + 0.05 * x2
    y2 = -0.02 * (x1 ** 3) + 0.70 * x2
    return np.stack([y1, y2], axis=-1)


SYSTEMS: Dict[str, PolynomialSystem2D] = {
    "S1": PolynomialSystem2D(
        name="S1",
        vars=("x1", "x2"),
        domain=((-1.4, 1.4), (-1.4, 1.4)),
        x0_box=((-0.2, 0.2), (-0.2, 0.2)),
        xu_box=((0.95, 1.25), (0.95, 1.25)),
        dynamics=system_s1_dynamics,
        description="Baseline polynomial benchmark.",
    ),
    "S2": PolynomialSystem2D(
        name="S2",
        vars=("x1", "x2"),
        domain=((-1.2, 1.2), (-1.2, 1.2)),
        x0_box=((-0.15, 0.15), (-0.15, 0.15)),
        xu_box=((0.50, 1.10), (0.50, 1.10)),
        dynamics=system_s2_dynamics,
        description="Polynomial benchmark used for forward/backward comparison.",
    ),
}


def box_grid(box: Box, n: int) -> np.ndarray:
    (xlo, xhi), (ylo, yhi) = box
    xs = np.linspace(xlo, xhi, n)
    ys = np.linspace(ylo, yhi, n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    return pts


def sample_box(box: Box, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    (xlo, xhi), (ylo, yhi) = box
    x = rng.uniform(xlo, xhi, size=n)
    y = rng.uniform(ylo, yhi, size=n)
    return np.column_stack([x, y])


def simulate_trajectories(
    f: Callable[[np.ndarray], np.ndarray],
    x0_samples: np.ndarray,
    horizon: int,
) -> List[np.ndarray]:
    trajectories: List[np.ndarray] = []
    for x0 in x0_samples:
        traj = [np.asarray(x0, dtype=float)]
        x = np.asarray(x0, dtype=float)
        for _ in range(horizon):
            x = np.asarray(f(x[None, :])[0], dtype=float)
            traj.append(x.copy())
        trajectories.append(np.vstack(traj))
    return trajectories