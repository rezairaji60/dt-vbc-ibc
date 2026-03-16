"""
Run stable surrogate synthesis experiments.

Author: Reza Iraji
Date:   March 2026
"""
from __future__ import annotations

import itertools
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from dt_vbc.plotting_sos import plot_dt_vbc_components
from dt_vbc.synthesis_sos import (
    solve_backward_dt_vbc,
    solve_backward_ibc,
    solve_forward_dt_vbc,
    solve_forward_ibc,
)
from dt_vbc.systems_sos import SYSTEMS


DEGREE = 2
GRID_N_DOMAIN = 25
GRID_N_BOUNDARY = 10

RESULTS_DIR = os.path.join("results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)


def status_ok(status: str) -> bool:
    return status in ("optimal", "optimal_inaccurate")


def candidate_forward_As() -> List[np.ndarray]:
    vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    mats = []
    for a00, a01, a11 in itertools.product(vals, vals, vals):
        A = np.array([[a00, a01], [0.0, a11]], dtype=float)
        mats.append(A)
    return mats


def candidate_backward_As() -> List[np.ndarray]:
    vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    mats = []
    for a00, a10, a11 in itertools.product(vals, vals, vals):
        A = np.array([[a00, 0.0], [a10, a11]], dtype=float)
        mats.append(A)
    return mats


def candidate_lambdas(n_frames: int) -> List[Tuple[float, ...]]:
    vals = [0.5, 1.0, 1.5]
    return list(itertools.product(vals, repeat=n_frames))


def best_feasible_forward(system_key: str):
    sys = SYSTEMS[system_key]
    best = None
    for A in candidate_forward_As():
        res = solve_forward_dt_vbc(
            dynamics=sys.dynamics,
            domain=sys.domain,
            x0_box=sys.x0_box,
            xu_box=sys.xu_box,
            degree=DEGREE,
            n_components=2,
            comparison_matrix=A,
            grid_n_domain=GRID_N_DOMAIN,
            grid_n_boundary=GRID_N_BOUNDARY,
            eps_hi=0.1,
            bisection_steps=10,
            solver="SCS",
            solver_kwargs={"verbose": False, "max_iters": 20000, "eps": 1e-5},
        )
        if status_ok(res["status"]):
            if best is None or res["epsilon"] > best["epsilon"]:
                best = res
    return best


def best_feasible_backward(system_key: str):
    sys = SYSTEMS[system_key]
    best = None
    for A in candidate_backward_As():
        res = solve_backward_dt_vbc(
            dynamics=sys.dynamics,
            domain=sys.domain,
            x0_box=sys.x0_box,
            xu_box=sys.xu_box,
            degree=DEGREE,
            n_components=2,
            comparison_matrix=A,
            grid_n_domain=GRID_N_DOMAIN,
            grid_n_boundary=GRID_N_BOUNDARY,
            eps_hi=0.1,
            bisection_steps=10,
            solver="SCS",
            solver_kwargs={"verbose": False, "max_iters": 20000, "eps": 1e-5},
        )
        if status_ok(res["status"]):
            if best is None or res["epsilon"] > best["epsilon"]:
                best = res
    return best


def best_feasible_forward_ibc(system_key: str):
    sys = SYSTEMS[system_key]
    best = None
    for lambdas in candidate_lambdas(3):
        res = solve_forward_ibc(
            dynamics=sys.dynamics,
            domain=sys.domain,
            x0_box=sys.x0_box,
            xu_box=sys.xu_box,
            degree=DEGREE,
            n_frames=3,
            lambdas=lambdas,
            grid_n_domain=GRID_N_DOMAIN,
            grid_n_boundary=GRID_N_BOUNDARY,
            eps_hi=0.1,
            bisection_steps=10,
            solver="SCS",
            solver_kwargs={"verbose": False, "max_iters": 20000, "eps": 1e-5},
        )
        if status_ok(res["status"]):
            if best is None or res["epsilon"] > best["epsilon"]:
                best = res
    return best


def best_feasible_backward_ibc(system_key: str):
    sys = SYSTEMS[system_key]
    best = None
    for lambdas in candidate_lambdas(3):
        res = solve_backward_ibc(
            dynamics=sys.dynamics,
            domain=sys.domain,
            x0_box=sys.x0_box,
            xu_box=sys.xu_box,
            degree=DEGREE,
            n_frames=3,
            lambdas=lambdas,
            grid_n_domain=GRID_N_DOMAIN,
            grid_n_boundary=GRID_N_BOUNDARY,
            eps_hi=0.1,
            bisection_steps=10,
            solver="SCS",
            solver_kwargs={"verbose": False, "max_iters": 20000, "eps": 1e-5},
        )
        if status_ok(res["status"]):
            if best is None or res["epsilon"] > best["epsilon"]:
                best = res
    return best


def record_row(system_key: str, formulation: str, functions: int, res: Dict) -> Dict:
    return {
        "system": system_key,
        "formulation": formulation,
        "degree": DEGREE,
        "functions": functions,
        "status": res["status"] if res is not None else "infeasible",
        "epsilon": float(res["epsilon"]) if res is not None else 0.0,
        "runtime_sec": float(res["runtime_sec"]) if res is not None else np.nan,
        "parameter": str(res.get("A", res.get("lambdas", ""))) if res is not None else "",
    }


def main():
    ensure_dirs()

    rows = []

    # S1
    s1_fwd = best_feasible_forward("S1")
    s1_bwd = best_feasible_backward("S1")
    s1_fibc = best_feasible_forward_ibc("S1")
    s1_bibc = best_feasible_backward_ibc("S1")

    rows.append(record_row("S1", "Forward DT-VBC", 2, s1_fwd))
    rows.append(record_row("S1", "Backward DT-VBC", 2, s1_bwd))
    rows.append(record_row("S1", "Forward IBC", 3, s1_fibc))
    rows.append(record_row("S1", "Backward IBC", 3, s1_bibc))

    # S2
    s2_fwd = best_feasible_forward("S2")
    s2_bwd = best_feasible_backward("S2")
    s2_fibc = best_feasible_forward_ibc("S2")
    s2_bibc = best_feasible_backward_ibc("S2")

    rows.append(record_row("S2", "Forward DT-VBC", 2, s2_fwd))
    rows.append(record_row("S2", "Backward DT-VBC", 2, s2_bwd))
    rows.append(record_row("S2", "Forward IBC", 3, s2_fibc))
    rows.append(record_row("S2", "Backward IBC", 3, s2_bibc))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(TABLE_DIR, "sos_results_summary.csv"), index=False)

    if s1_fwd is not None:
        plot_dt_vbc_components(
            output_path=os.path.join(FIG_DIR, "case1_forward_dt_vbc_synthesized.pdf"),
            title="Case Study 1: synthesized forward DT-VBC",
            vars_xy=SYSTEMS["S1"].vars,
            degree=DEGREE,
            coefficients=s1_fwd["coefficients"],
            coeff_names=["fwdB0", "fwdB1"],
            dynamics=SYSTEMS["S1"].dynamics,
            domain=SYSTEMS["S1"].domain,
            x0_box=SYSTEMS["S1"].x0_box,
            xu_box=SYSTEMS["S1"].xu_box,
            seed=1,
        )

    if s2_bwd is not None:
        plot_dt_vbc_components(
            output_path=os.path.join(FIG_DIR, "case2_backward_dt_vbc_synthesized.pdf"),
            title="Case Study 2: synthesized backward DT-VBC",
            vars_xy=SYSTEMS["S2"].vars,
            degree=DEGREE,
            coefficients=s2_bwd["coefficients"],
            coeff_names=["bwdB0", "bwdB1"],
            dynamics=SYSTEMS["S2"].dynamics,
            domain=SYSTEMS["S2"].domain,
            x0_box=SYSTEMS["S2"].x0_box,
            xu_box=SYSTEMS["S2"].xu_box,
            seed=2,
        )


if __name__ == "__main__":
    main()