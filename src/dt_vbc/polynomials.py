"""
Author: Reza Iraji
Date:   March 2026
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Exponent = Tuple[int, ...]


def monomial_exponents_2d(max_degree: int) -> List[Exponent]:
    exps: List[Exponent] = []
    for d in range(max_degree + 1):
        for a in range(d + 1):
            b = d - a
            exps.append((a, b))
    return exps


def evaluate_monomials_2d(points: np.ndarray, exponents: Sequence[Exponent]) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    cols = []
    for a, b in exponents:
        cols.append((x ** a) * (y ** b))
    return np.column_stack(cols)


def eval_poly_2d(points: np.ndarray, coeffs: np.ndarray, exponents: Sequence[Exponent]) -> np.ndarray:
    Phi = evaluate_monomials_2d(points, exponents)
    return Phi @ coeffs


def poly_label(prefix: str, idx: int) -> str:
    return f"{prefix}{idx}"