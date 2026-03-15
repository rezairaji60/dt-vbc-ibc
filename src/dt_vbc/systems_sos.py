"""
Author: Reza Iraji
Date:   March 2026
"""
from __future__ import annotations
import sympy as sp
from .synthesis_sos import SemialgebraicSet

x1, x2 = sp.symbols('x1 x2', real=True)

# System S1
f1 = (
    0.72 * x1 + 0.10 * x2 - 0.12 * x1**3,
    -0.08 * x1 + 0.68 * x2 - 0.08 * x2**3,
)
X1 = SemialgebraicSet(polys=[1.30**2 - x1**2, 1.30**2 - x2**2])
X01 = SemialgebraicSet(polys=[0.20**2 - x1**2, 0.20**2 - x2**2])
Xu1 = SemialgebraicSet(polys=[(x1 - 0.95) * (1.25 - x1), (x2 - 0.95) * (1.25 - x2)])

# System S2
f2 = (
    0.60 * x1 + 0.05 * x2,
    -0.02 * x1**3 + 0.70 * x2,
)
X2 = SemialgebraicSet(polys=[1.20**2 - x1**2, 1.20**2 - x2**2])
X02 = SemialgebraicSet(polys=[0.15**2 - x1**2, 0.15**2 - x2**2])
Xu2 = SemialgebraicSet(polys=[(x1 - 0.50) * (1.10 - x1), (x2 - 0.50) * (1.10 - x2)])

SYSTEMS = {
    'S1': {
        'vars': (x1, x2),
        'f': f1,
        'X': X1,
        'X0': X01,
        'Xu': Xu1,
        'domain': (-1.3, 1.3, -1.3, 1.3),
        'X0_box': (-0.2, 0.2, -0.2, 0.2),
        'Xu_box': (0.95, 1.25, 0.95, 1.25),
        'A_candidates_forward': [
            [[0.8, 0.0], [0.0, 0.8]],
            [[0.8, 0.05], [0.05, 0.8]],
            [[0.9, 0.0], [0.0, 0.9]],
        ],
        'A_candidates_backward': [
            [[1.1, 0.0], [0.0, 1.1]],
            [[1.4, 0.0], [0.0, 1.4]],
        ],
        'lambda_candidates': [
            [0.8, 0.8, 0.8],
            [1.0, 1.0, 1.0],
            [1.2, 1.2, 1.2],
        ],
    },
    'S2': {
        'vars': (x1, x2),
        'f': f2,
        'X': X2,
        'X0': X02,
        'Xu': Xu2,
        'domain': (-1.2, 1.2, -1.2, 1.2),
        'X0_box': (-0.15, 0.15, -0.15, 0.15),
        'Xu_box': (0.50, 1.10, 0.50, 1.10),
        'A_candidates_forward': [
            [[0.7, 0.0], [0.0, 0.7]],
            [[0.8, 0.0], [0.0, 0.8]],
        ],
        'A_candidates_backward': [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.4, 0.0], [0.0, 1.4]],
            [[1.8, 0.0], [0.0, 1.8]],
        ],
        'lambda_candidates': [
            [0.9, 0.9, 0.9],
            [1.0, 1.0, 1.0],
            [1.2, 1.2, 1.2],
            [1.5, 1.5, 1.5],
        ],
    },
}
