from __future__ import annotations

import numpy as np
from typing import Callable

SystemMap = Callable[[np.ndarray], np.ndarray]


def contraction_map(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    xp = 0.62 * x - 0.10 * y
    yp = 0.08 * x + 0.58 * y
    return np.column_stack([xp, yp])


def expansion_map(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    xp = 1.18 * x + 0.06 * y
    yp = -0.04 * x + 1.12 * y
    return np.column_stack([xp, yp])


def box_mask(points: np.ndarray, x_lo: float, x_hi: float, y_lo: float, y_hi: float) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    return (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)


def annulus_mask(points: np.ndarray, r_lo: float, r_hi: float) -> np.ndarray:
    r2 = points[:, 0] ** 2 + points[:, 1] ** 2
    return (r2 >= r_lo ** 2) & (r2 <= r_hi ** 2)


def disk_mask(points: np.ndarray, r: float) -> np.ndarray:
    r2 = points[:, 0] ** 2 + points[:, 1] ** 2
    return r2 <= r ** 2


def iterate_map(f: SystemMap, x0: np.ndarray, steps: int) -> np.ndarray:
    traj = np.zeros((steps + 1, len(x0)))
    traj[0] = x0
    x = x0[None, :]
    for k in range(steps):
        x = f(x)
        traj[k + 1] = x[0]
    return traj
