"""
Author: Reza Iraji
Date:   March 2026

High-level driver for SOS experiments.
"""
from __future__ import annotations
import sympy as sp
from dt_vbc.synthesis_sos import ForwardDTVBCSOS, SemialgebraicSet


def main():
    x1, x2 = sp.symbols("x1 x2", real=True)

    # Example: System S1 from the current repository.
    f1 = (
        0.72 * x1 + 0.10 * x2 - 0.12 * x1**3,
        -0.08 * x1 + 0.68 * x2 - 0.08 * x2**3,
    )

    # Domain and semialgebraic sets expressed as g(x) >= 0.
    X = SemialgebraicSet(polys=[1.30**2 - x1**2, 1.30**2 - x2**2])
    X0 = SemialgebraicSet(polys=[0.20**2 - x1**2, 0.20**2 - x2**2])
    Xu = SemialgebraicSet(polys=[(x1 - 0.95) * (1.25 - x1), (x2 - 0.95) * (1.25 - x2)])

    builder = ForwardDTVBCSOS(f1, (x1, x2), X, X0, Xu, m=2, degree=2)
    prob = builder.build_problem()

    print("Built SOS problem for Forward DT-VBC on S1.")
    print("CVXPY variables:", len(prob.variables()))
    print("CVXPY constraints:", len(prob.constraints))
    print("Next step: choose SDP solver (MOSEK / SCS) and solve.")


if __name__ == "__main__":
    main()
