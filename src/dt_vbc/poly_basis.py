"""
Author: Reza Iraji
Date:   March 2026

Polynomial basis helpers for SOS skeleton.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import List, Sequence, Tuple
import sympy as sp


def monomial_exponents(dim: int, degree: int) -> List[Tuple[int, ...]]:
    exps = []
    for tup in product(range(degree + 1), repeat=dim):
        if sum(tup) <= degree:
            exps.append(tup)
    exps.sort(key=lambda t: (sum(t), t))
    return exps


def monomials(vars_: Sequence[sp.Symbol], degree: int) -> List[sp.Expr]:
    out = []
    for exp in monomial_exponents(len(vars_), degree):
        term = sp.Integer(1)
        for v, e in zip(vars_, exp):
            term *= v ** e
        out.append(sp.expand(term))
    return out


@dataclass
class PolyTemplate:
    vars_: Tuple[sp.Symbol, ...]
    degree: int
    coeff_prefix: str

    def build(self):
        mons = monomials(self.vars_, self.degree)
        coeffs = sp.symbols(f"{self.coeff_prefix}0:{len(mons)}", real=True)
        poly = sum(c * m for c, m in zip(coeffs, mons))
        return poly, list(coeffs), mons
