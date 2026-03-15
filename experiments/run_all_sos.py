"""
Author: Reza Iraji
Date:   March 2026
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from dt_vbc.plotting_sos import plot_bar_comparison, plot_dt_vbc_components
from dt_vbc.synthesis_sos import (
    solve_backward_dt_vbc_collocation,
    solve_backward_ibc_collocation,
    solve_forward_dt_vbc_collocation,
    solve_forward_ibc_collocation,
)
from dt_vbc.systems_sos import SYSTEMS, box_grid


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TAB_DIR = os.path.join(RESULTS_DIR, "tables")


DEGREE = 2
GRID_N_DOMAIN = 33
GRID_N_BOUNDARY = 12


def ensure_dirs() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)


def build_sets(system_key: str) -> dict[str, np.ndarray]:
    sys = SYSTEMS[system_key]
    domain_pts = box_grid(sys.domain, GRID_N_DOMAIN)
    image_pts = sys.dynamics(domain_pts)
    x0_pts = box_grid(sys.x0_box, GRID_N_BOUNDARY)
    xu_pts = box_grid(sys.xu_box, GRID_N_BOUNDARY)
    return {
        "domain_pts": domain_pts,
        "image_pts": image_pts,
        "x0_pts": x0_pts,
        "xu_pts": xu_pts,
    }


def run_system_s1() -> list[dict]:
    sys = SYSTEMS["S1"]
    sets = build_sets("S1")

    A_fwd = np.array([[0.0, 0.8], [0.0, 0.9]], dtype=float)
    A_bwd = np.array([[0.9, 0.0], [0.7, 0.0]], dtype=float)
    lambdas = [1.0, 1.0, 1.0]

    results = []

    for label, solver in [
        ("Forward DT-VBC", lambda: solve_forward_dt_vbc_collocation(**sets, degree=DEGREE, comparison_matrix=A_fwd)),
        ("Backward DT-VBC", lambda: solve_backward_dt_vbc_collocation(**sets, degree=DEGREE, comparison_matrix=A_bwd)),
        ("Forward IBC", lambda: solve_forward_ibc_collocation(**sets, degree=DEGREE, k=2, lambdas=lambdas)),
        ("Backward IBC", lambda: solve_backward_ibc_collocation(**sets, degree=DEGREE, k=2, lambdas=lambdas)),
    ]:
        tic = time.time()
        res = solver()
        toc = time.time()
        results.append(
            {
                "system": "S1",
                "formulation": label,
                "degree": res.degree,
                "functions": res.n_functions,
                "status": res.status,
                "epsilon": res.epsilon,
                "runtime_sec": toc - tic,
                "coefficients": res.coefficients,
            }
        )

    # forward figure
    fwd = next(r for r in results if r["formulation"] == "Forward DT-VBC")
    plot_dt_vbc_components(
        os.path.join(FIG_DIR, "case1_forward_dt_vbc_synthesized.pdf"),
        "S1 forward DT-VBC",
        DEGREE,
        fwd["coefficients"],
        ["fwdB0", "fwdB1"],
        sys.dynamics,
        sys.domain,
        sys.x0_box,
        sys.xu_box,
        seed=1,
    )

    return results


def run_system_s2() -> list[dict]:
    sys = SYSTEMS["S2"]
    sets = build_sets("S2")

    A_fwd = np.array([[0.0, 0.8], [0.0, 0.9]], dtype=float)
    A_bwd = np.array([[0.9, 0.0], [0.7, 0.0]], dtype=float)
    lambdas = [1.0, 1.0, 1.0]

    results = []

    for label, solver in [
        ("Forward DT-VBC", lambda: solve_forward_dt_vbc_collocation(**sets, degree=DEGREE, comparison_matrix=A_fwd)),
        ("Backward DT-VBC", lambda: solve_backward_dt_vbc_collocation(**sets, degree=DEGREE, comparison_matrix=A_bwd)),
        ("Forward IBC", lambda: solve_forward_ibc_collocation(**sets, degree=DEGREE, k=2, lambdas=lambdas)),
        ("Backward IBC", lambda: solve_backward_ibc_collocation(**sets, degree=DEGREE, k=2, lambdas=lambdas)),
    ]:
        tic = time.time()
        res = solver()
        toc = time.time()
        results.append(
            {
                "system": "S2",
                "formulation": label,
                "degree": res.degree,
                "functions": res.n_functions,
                "status": res.status,
                "epsilon": res.epsilon,
                "runtime_sec": toc - tic,
                "coefficients": res.coefficients,
            }
        )

    # backward figure
    bwd = next(r for r in results if r["formulation"] == "Backward DT-VBC")
    plot_dt_vbc_components(
        os.path.join(FIG_DIR, "case2_backward_dt_vbc_synthesized.pdf"),
        "S2 backward DT-VBC",
        DEGREE,
        bwd["coefficients"],
        ["bwdB0", "bwdB1"],
        sys.dynamics,
        sys.domain,
        sys.x0_box,
        sys.xu_box,
        seed=2,
    )

    return results


def main() -> None:
    ensure_dirs()

    rows = []
    rows.extend(run_system_s1())
    rows.extend(run_system_s2())

    printable = []
    for r in rows:
        printable.append(
            {
                "system": r["system"],
                "formulation": r["formulation"],
                "degree": r["degree"],
                "functions": r["functions"],
                "status": r["status"],
                "epsilon": r["epsilon"],
                "runtime_sec": r["runtime_sec"],
            }
        )

    df = pd.DataFrame(printable)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(TAB_DIR, "sos_summary.csv"), index=False)

    plot_bar_comparison(
        os.path.join(FIG_DIR, "comparison_methods.pdf"),
        printable,
    )


if __name__ == "__main__":
    main()