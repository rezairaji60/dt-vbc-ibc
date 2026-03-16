"""
Run finite-parameter SOS surrogate synthesis experiments.

This script performs an outer sweep over fixed comparison matrices (for DT-VBC)
and fixed scaling vectors (for IBC), and for each candidate solves the inner
convex synthesis problem. The best feasible candidate per formulation/system is
reported and used for figure generation.

Author: Reza Iraji
Date:   March 2026
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dt_vbc.synthesis_sos import (
    solve_backward_dt_vbc,
    solve_backward_ibc,
    solve_forward_dt_vbc,
    solve_forward_ibc,
)
from dt_vbc.plotting_sos import plot_dt_vbc_components


# -----------------------------------------------------------------------------
# GLOBAL SETTINGS
# -----------------------------------------------------------------------------

DEGREE = 2

OUTPUT_DIR = "results"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
CSV_PATH = os.path.join(OUTPUT_DIR, "sos_results.csv")

EPS_SEARCH = {
    "eps_hi": 1.0,
    "bisection_steps": 18,
}

SOLVER = "SCS"
SOLVER_KWARGS = {
    "verbose": False,
    "eps": 1e-5,
    "max_iters": 30000,
}

PLOT_SETTINGS = {
    "grid_n": 220,
    "horizon": 18,
    "n_traj": 8,
}


# -----------------------------------------------------------------------------
# SYSTEMS
# -----------------------------------------------------------------------------

def dynamics_S1(points: np.ndarray) -> np.ndarray:
    x1 = points[:, 0]
    x2 = points[:, 1]
    y1 = 0.72 * x1 + 0.10 * x2 - 0.12 * (x1 ** 3)
    y2 = -0.08 * x1 + 0.68 * x2 - 0.08 * (x2 ** 3)
    return np.column_stack([y1, y2])


def dynamics_S2(points: np.ndarray) -> np.ndarray:
    x1 = points[:, 0]
    x2 = points[:, 1]
    y1 = 1.05 * x1 + 0.18 * x2 - 0.10 * (x1 ** 3)
    y2 = 0.02 * x1 + 0.92 * x2
    return np.column_stack([y1, y2])


SYSTEMS: Dict[str, Dict[str, Any]] = {
    "S1": {
        "vars": ("x1", "x2"),
        "f": [
            "0.72*x1 + 0.10*x2 - 0.12*x1**3",
            "-0.08*x1 + 0.68*x2 - 0.08*x2**3",
        ],
        "dynamics": dynamics_S1,
        "domain": ((-1.4, 1.4), (-1.4, 1.4)),
        "X0_box": ((-0.2, 0.2), (-0.2, 0.2)),
        "Xu_box": ((0.95, 1.25), (0.95, 1.25)),
    },
    "S2": {
        "vars": ("x1", "x2"),
        "f": [
            "1.05*x1 + 0.18*x2 - 0.10*x1**3",
            "0.02*x1 + 0.92*x2",
        ],
        "dynamics": dynamics_S2,
        "domain": ((-1.4, 1.4), (-1.2, 1.2)),
        "X0_box": ((-0.12, 0.12), (-0.12, 0.12)),
        "Xu_box": ((0.85, 1.10), (-0.10, 0.10)),
    },
}


# -----------------------------------------------------------------------------
# FINITE PARAMETER FAMILIES
# -----------------------------------------------------------------------------

CANDIDATE_PARAMETERS: Dict[str, Dict[str, List[Any]]] = {
    "S1": {
        "Forward DT-VBC": [
            np.array([[0.60, 0.00], [0.00, 0.60]], dtype=float),
            np.array([[0.70, 0.00], [0.00, 0.70]], dtype=float),
            np.array([[0.80, 0.00], [0.00, 0.80]], dtype=float),
            np.array([[0.90, 0.00], [0.00, 0.90]], dtype=float),
            np.array([[0.80, 0.05], [0.00, 0.80]], dtype=float),
            np.array([[0.85, 0.05], [0.00, 0.85]], dtype=float),
            np.array([[0.90, 0.05], [0.00, 0.90]], dtype=float),
        ],
        "Backward DT-VBC": [
            np.array([[1.00, 0.00], [0.00, 1.00]], dtype=float),
            np.array([[1.10, 0.00], [0.00, 1.10]], dtype=float),
            np.array([[1.20, 0.00], [0.00, 1.20]], dtype=float),
            np.array([[1.30, 0.00], [0.00, 1.30]], dtype=float),
            np.array([[1.10, 0.05], [0.00, 1.10]], dtype=float),
            np.array([[1.20, 0.05], [0.00, 1.20]], dtype=float),
            np.array([[1.20, 0.10], [0.00, 1.20]], dtype=float),
        ],
        "Forward IBC": [
            [0.50, 0.50, 1.00],
            [0.50, 0.50, 1.50],
            [0.75, 0.75, 1.00],
            [0.75, 0.75, 1.50],
            [1.00, 1.00, 1.00],
        ],
        "Backward IBC": [
            [1.00, 1.00, 1.00],
            [1.20, 1.20, 1.20],
            [1.50, 1.50, 1.50],
            [1.00, 1.20, 1.50],
        ],
    },
    "S2": {
        "Forward DT-VBC": [
            np.array([[0.65, 0.00], [0.00, 0.65]], dtype=float),
            np.array([[0.75, 0.00], [0.00, 0.75]], dtype=float),
            np.array([[0.85, 0.00], [0.00, 0.85]], dtype=float),
            np.array([[0.80, 0.05], [0.00, 0.80]], dtype=float),
            np.array([[0.90, 0.10], [0.00, 0.90]], dtype=float),
        ],
        "Backward DT-VBC": [
            np.array([[1.20, 0.00], [0.00, 1.20]], dtype=float),
            np.array([[1.40, 0.00], [0.00, 1.40]], dtype=float),
            np.array([[1.60, 0.00], [0.00, 1.60]], dtype=float),
            np.array([[1.80, 0.10], [0.00, 1.80]], dtype=float),
            np.array([[2.00, 0.15], [0.00, 2.00]], dtype=float),
            np.array([[2.20, 0.20], [0.00, 2.20]], dtype=float),
        ],
        "Forward IBC": [
            [0.50, 0.50, 1.00],
            [0.75, 0.75, 1.25],
            [0.75, 0.75, 1.50],
            [1.00, 1.00, 1.50],
        ],
        "Backward IBC": [
            [1.20, 1.20, 1.20],
            [1.50, 1.50, 1.50],
            [1.80, 1.80, 1.80],
            [1.20, 1.50, 1.80],
            [1.50, 1.80, 2.00],
        ],
    },
}


# -----------------------------------------------------------------------------
# FORMULATION SPECS
# -----------------------------------------------------------------------------

FORMULATIONS = [
    {
        "name": "Forward DT-VBC",
        "functions": 2,
        "solver_fn": solve_forward_dt_vbc,
        "plot_prefixes": ["fwdB0", "fwdB1"],
    },
    {
        "name": "Backward DT-VBC",
        "functions": 2,
        "solver_fn": solve_backward_dt_vbc,
        "plot_prefixes": ["bwdB0", "bwdB1"],
    },
    {
        "name": "Forward IBC",
        "functions": 3,
        "solver_fn": solve_forward_ibc,
        "plot_prefixes": None,
    },
    {
        "name": "Backward IBC",
        "functions": 3,
        "solver_fn": solve_backward_ibc,
        "plot_prefixes": None,
    },
]


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def is_feasible_status(status: str) -> bool:
    s = str(status).lower()
    return ("optimal" in s) and ("infeasible" not in s) and ("unbounded" not in s)


def parameter_to_string(p: Any) -> str:
    if isinstance(p, np.ndarray):
        return np.array2string(p, precision=3, suppress_small=True)
    if isinstance(p, (list, tuple)):
        return str([float(x) for x in p])
    return str(p)


def coeff_key_candidates(base_name: str) -> List[str]:
    return [
        base_name,
        base_name + "_",
        base_name.replace("B", "B_"),
        base_name.replace("I", "I_"),
    ]


def remap_coefficients(coeffs: Dict[str, np.ndarray], desired_names: List[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    used = set()

    for desired in desired_names:
        found_key = None
        for cand in coeff_key_candidates(desired):
            if cand in coeffs:
                found_key = cand
                break
        if found_key is None:
            for k in coeffs.keys():
                if k not in used:
                    found_key = k
                    break
        if found_key is not None:
            out[desired] = coeffs[found_key]
            used.add(found_key)

    return out


def solve_one_candidate(
    system_data: Dict[str, Any],
    formulation_name: str,
    solver_fn,
    parameter: Any,
) -> Dict[str, Any]:
    t0 = time.time()

    result = solver_fn(
        dynamics=system_data["dynamics"],
        domain=system_data["domain"],
        x0_box=system_data["X0_box"],
        xu_box=system_data["Xu_box"],
        degree=DEGREE,
        n_components=2 if "DT-VBC" in formulation_name else None,
        n_frames=3 if "IBC" in formulation_name else None,
        comparison_matrix=parameter if "DT-VBC" in formulation_name else None,
        lambdas=parameter if "IBC" in formulation_name else None,
        eps_hi=EPS_SEARCH["eps_hi"],
        bisection_steps=EPS_SEARCH["bisection_steps"],
        solver=SOLVER,
        solver_kwargs=SOLVER_KWARGS,
    )

    runtime_sec = time.time() - t0
    result["runtime_sec_outer"] = runtime_sec
    result["parameter"] = parameter
    return result


def run_best_of_candidates(
    system_name: str,
    system_data: Dict[str, Any],
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    formulation_name = spec["name"]
    solver_fn = spec["solver_fn"]
    num_functions = spec["functions"]

    candidates = CANDIDATE_PARAMETERS[system_name][formulation_name]

    best_row = None
    best_eps = -np.inf

    for idx, parameter in enumerate(candidates):
        t0 = time.time()

        if formulation_name == "Forward DT-VBC":
            result = solver_fn(
                dynamics=system_data["dynamics"],
                domain=system_data["domain"],
                x0_box=system_data["X0_box"],
                xu_box=system_data["Xu_box"],
                degree=DEGREE,
                n_components=2,
                comparison_matrix=parameter,
                eps_hi=EPS_SEARCH["eps_hi"],
                bisection_steps=EPS_SEARCH["bisection_steps"],
                solver=SOLVER,
                solver_kwargs=SOLVER_KWARGS,
            )
        elif formulation_name == "Backward DT-VBC":
            result = solver_fn(
                dynamics=system_data["dynamics"],
                domain=system_data["domain"],
                x0_box=system_data["X0_box"],
                xu_box=system_data["Xu_box"],
                degree=DEGREE,
                n_components=2,
                comparison_matrix=parameter,
                eps_hi=EPS_SEARCH["eps_hi"],
                bisection_steps=EPS_SEARCH["bisection_steps"],
                solver=SOLVER,
                solver_kwargs=SOLVER_KWARGS,
            )
        elif formulation_name == "Forward IBC":
            result = solver_fn(
                dynamics=system_data["dynamics"],
                domain=system_data["domain"],
                x0_box=system_data["X0_box"],
                xu_box=system_data["Xu_box"],
                degree=DEGREE,
                n_frames=3,
                lambdas=parameter,
                eps_hi=EPS_SEARCH["eps_hi"],
                bisection_steps=EPS_SEARCH["bisection_steps"],
                solver=SOLVER,
                solver_kwargs=SOLVER_KWARGS,
            )
        elif formulation_name == "Backward IBC":
            result = solver_fn(
                dynamics=system_data["dynamics"],
                domain=system_data["domain"],
                x0_box=system_data["X0_box"],
                xu_box=system_data["Xu_box"],
                degree=DEGREE,
                n_frames=3,
                lambdas=parameter,
                eps_hi=EPS_SEARCH["eps_hi"],
                bisection_steps=EPS_SEARCH["bisection_steps"],
                solver=SOLVER,
                solver_kwargs=SOLVER_KWARGS,
            )
        else:
            raise ValueError(f"Unknown formulation: {formulation_name}")

        runtime_sec = time.time() - t0
        status = result.get("status", "unknown")
        eps_val = float(result.get("epsilon", 0.0) or 0.0)

        row = {
            "system": system_name,
            "formulation": formulation_name,
            "degree": DEGREE,
            "functions": num_functions,
            "status": status,
            "epsilon": eps_val if np.isfinite(eps_val) else np.nan,
            "runtime_sec": runtime_sec,
            "parameter": parameter,
            "parameter_index": idx,
            "coefficients": result.get("coefficients", {}),
        }

        if is_feasible_status(status):
            if eps_val > best_eps:
                best_eps = eps_val
                best_row = row

    if best_row is not None:
        return best_row

    return {
        "system": system_name,
        "formulation": formulation_name,
        "degree": DEGREE,
        "functions": num_functions,
        "status": "infeasible",
        "epsilon": 0.0,
        "runtime_sec": np.nan,
        "parameter": candidates[0] if len(candidates) > 0 else None,
        "parameter_index": -1,
        "coefficients": {},
    }


def maybe_plot_dt_vbc(row: Dict[str, Any], system_data: Dict[str, Any]) -> None:
    if row["formulation"] not in ("Forward DT-VBC", "Backward DT-VBC"):
        return
    if not is_feasible_status(row["status"]):
        return
    if not row["coefficients"]:
        return

    if row["formulation"] == "Forward DT-VBC":
        desired_names = ["fwdB0", "fwdB1"]
    else:
        desired_names = ["bwdB0", "bwdB1"]

    coeffs = remap_coefficients(row["coefficients"], desired_names)
    if len(coeffs) != 2:
        return

    if row["system"] == "S1" and row["formulation"] == "Forward DT-VBC":
        filename = os.path.join(FIG_DIR, "case1_forward_dt_vbc_synthesized.pdf")
        title = "S1 forward DT-VBC"
    elif row["system"] == "S2" and row["formulation"] == "Backward DT-VBC":
        filename = os.path.join(FIG_DIR, "case2_backward_dt_vbc_synthesized.pdf")
        title = "S2 backward DT-VBC"
    else:
        safe_name = row["formulation"].lower().replace(" ", "_").replace("-", "_")
        filename = os.path.join(FIG_DIR, f"{row['system']}_{safe_name}.pdf")
        title = f"{row['system']} {row['formulation']}"

    plot_dt_vbc_components(
        filename,
        title,
        system_data["vars"],
        DEGREE,
        coeffs,
        desired_names,
        system_data["f"],          # <-- pass expression strings, not dynamics function
        system_data["domain"],
        system_data["X0_box"],
        system_data["Xu_box"],
        seed=1,
        grid_n=PLOT_SETTINGS["grid_n"],
        trajectory_horizon=PLOT_SETTINGS["horizon"],
        trajectory_count=PLOT_SETTINGS["n_traj"],
    )


def generate_comparison_plot(df: pd.DataFrame) -> None:
    systems = ["S1", "S2"]
    formulations = ["Forward DT-VBC", "Backward DT-VBC", "Forward IBC", "Backward IBC"]

    values = np.zeros((len(formulations), len(systems)))
    labels = [["" for _ in systems] for _ in formulations]

    for i, formulation in enumerate(formulations):
        for j, system in enumerate(systems):
            match = df[(df["system"] == system) & (df["formulation"] == formulation)]
            if len(match) == 0:
                continue
            row = match.iloc[0]
            if is_feasible_status(row["status"]):
                values[i, j] = float(row["epsilon"])
                labels[i][j] = f"{float(row['epsilon']):.2f}"
            else:
                values[i, j] = 0.0
                labels[i][j] = "inf."

    x = np.arange(len(systems))
    width = 0.18

    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    bars = []
    bars.append(ax.bar(x - 1.5 * width, values[0], width, label="Forward DT-VBC"))
    bars.append(ax.bar(x - 0.5 * width, values[1], width, label="Backward DT-VBC"))
    bars.append(ax.bar(x + 0.5 * width, values[2], width, label="Forward IBC"))
    bars.append(ax.bar(x + 1.5 * width, values[3], width, label="Backward IBC"))

    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel(r"Best synthesis margin $\varepsilon$")
    ax.set_title("Best feasible result over finite parameter families")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    ymax = max(0.2, float(np.max(values)) * 1.18 if np.max(values) > 0 else 0.2)
    ax.set_ylim(0.0, ymax)

    for i in range(len(formulations)):
        for j in range(len(systems)):
            xpos = x[j] + (-1.5 + i) * width
            ypos = values[i, j]
            txt = labels[i][j]
            ax.text(
                xpos,
                ypos + 0.02 * ymax,
                txt,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "comparison_methods.pdf"), bbox_inches="tight")
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    printable = df.copy()
    printable["parameter"] = printable["parameter"].apply(parameter_to_string)
    cols = [
        "system",
        "formulation",
        "degree",
        "functions",
        "status",
        "epsilon",
        "runtime_sec",
        "parameter_index",
        "parameter",
    ]
    print(printable[cols].to_string(index=False))


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main() -> None:
    ensure_dirs()

    all_rows: List[Dict[str, Any]] = []

    for system_name, system_data in SYSTEMS.items():
        for spec in FORMULATIONS:
            best_row = run_best_of_candidates(system_name, system_data, spec)
            all_rows.append(best_row)

    df = pd.DataFrame(all_rows)

    csv_df = df.copy()
    csv_df["parameter"] = csv_df["parameter"].apply(parameter_to_string)
    csv_df["coefficients"] = csv_df["coefficients"].apply(lambda _: "<stored in memory>")
    csv_df.to_csv(CSV_PATH, index=False)

    print_summary(df)

    for _, row in df.iterrows():
        maybe_plot_dt_vbc(row.to_dict(), SYSTEMS[row["system"]])

    generate_comparison_plot(df)


if __name__ == "__main__":
    main()