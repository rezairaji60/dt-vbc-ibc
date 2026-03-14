
"""
Author: Reza Iraji
Date:   March 2026
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from dt_vbc.common import sample_box, make_grid
from dt_vbc.systems import SYSTEM1, SYSTEM2, encoding_scalar_map, encoding_frames
from dt_vbc.synthesis_lp import (
    synth_forward_dt_vbc, synth_backward_dt_vbc,
    synth_forward_ibc, synth_backward_ibc
)
from dt_vbc.plotting import plot_certificate

def best_forward_vbc(system):
    X0 = sample_box(system["X0"], 5)
    Xu = sample_box(system["Xu"], 5)
    _, _, D = make_grid(system["domain"], 25)
    best = None
    for a in [0.65, 0.75, 0.85, 0.95]:
        for b in [0.0, 0.05, 0.10, 0.15]:
            A = np.array([[a, b], [b, a]])
            res = synth_forward_dt_vbc(system["f"], X0, Xu, D, A, m=2, sep_idx=0)
            if res["success"] and (best is None or res["epsilon"] > best["epsilon"]):
                best = res
    return best

def best_backward_vbc(system):
    X0 = sample_box(system["X0"], 5)
    Xu = sample_box(system["Xu"], 5)
    _, _, D = make_grid(system["domain"], 25)
    best = None
    for alpha in [1.0, 1.2, 1.5, 2.0]:
        A = np.array([[alpha, 0.0], [0.0, alpha]])
        res = synth_backward_dt_vbc(system["f"], X0, Xu, D, A, m=2, sep_idx=0)
        if res["success"] and (best is None or res["epsilon"] > best["epsilon"]):
            best = res
    return best

def best_forward_ibc(system):
    X0 = sample_box(system["X0"], 5)
    Xu = sample_box(system["Xu"], 5)
    _, _, D = make_grid(system["domain"], 25)
    best = None
    for lam in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        res = synth_forward_ibc(system["f"], X0, Xu, D, [lam, lam, lam], k=2)
        if res["success"] and (best is None or res["epsilon"] > best["epsilon"]):
            best = res
    return best

def best_backward_ibc(system):
    X0 = sample_box(system["X0"], 5)
    Xu = sample_box(system["Xu"], 5)
    _, _, D = make_grid(system["domain"], 25)
    best = None
    for lam in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        res = synth_backward_ibc(system["f"], X0, Xu, D, [lam, lam, lam], k=2)
        if res["success"] and (best is None or res["epsilon"] > best["epsilon"]):
            best = res
    return best

def run_encoding_case(results_dir):
    fig_dir = os.path.join(results_dir, "figures")
    data_dir = os.path.join(results_dir, "data")
    xs = np.linspace(-1.5, 1.5, 500)
    fx = encoding_scalar_map(xs)
    frames = encoding_frames(xs)
    frames_next = encoding_frames(fx)
    B = -frames
    Bnext = -frames_next
    A = np.array([[0,1,0],[0,0,1],[0,0,1]], dtype=float)
    bibc_slack_0 = frames_next[:,0] - frames[:,1]
    bibc_slack_1 = frames_next[:,1] - frames[:,2]
    bibc_slack_2 = frames_next[:,2] - frames[:,2]
    vbc_slack = (A @ B.T).T - Bnext

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(xs, frames[:,0], label=r"$b_0(x)$")
    axes[0].plot(xs, frames[:,1], label=r"$b_1(x)$")
    axes[0].plot(xs, frames[:,2], label=r"$b_2(x)$")
    axes[0].axhline(0.0, linewidth=1.0)
    axes[0].set_title("Backward IBC frames")
    axes[0].legend()
    axes[1].plot(xs, B[:,0], label=r"$B_1(x)$")
    axes[1].plot(xs, B[:,1], label=r"$B_2(x)$")
    axes[1].plot(xs, B[:,2], label=r"$B_3(x)$")
    axes[1].axhline(0.0, linewidth=1.0)
    axes[1].set_title("Encoded forward DT-VBC")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "case3_bibc_to_forward_vbc.png"), dpi=220)
    plt.close(fig)

    summary = {
        "A": A.tolist(),
        "min_bibc_slack_0": float(np.min(bibc_slack_0)),
        "min_bibc_slack_1": float(np.min(bibc_slack_1)),
        "min_bibc_slack_2": float(np.min(bibc_slack_2)),
        "min_vbc_slack_1": float(np.min(vbc_slack[:,0])),
        "min_vbc_slack_2": float(np.min(vbc_slack[:,1])),
        "min_vbc_slack_3": float(np.min(vbc_slack[:,2])),
    }
    with open(os.path.join(data_dir, "case3_encoding_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary

def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(repo_root, "results")
    fig_dir = os.path.join(results_dir, "figures")
    data_dir = os.path.join(results_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    systems = [SYSTEM1, SYSTEM2]
    rows = []
    chosen = {}

    for sys in systems:
        fwd_vbc = best_forward_vbc(sys)
        bwd_vbc = best_backward_vbc(sys)
        fwd_ibc = best_forward_ibc(sys)
        bwd_ibc = best_backward_ibc(sys)

        chosen[sys["name"]] = {
            "forward_dt_vbc": fwd_vbc,
            "backward_dt_vbc": bwd_vbc,
            "forward_ibc": fwd_ibc,
            "backward_ibc": bwd_ibc,
        }

        for label, res in [
            ("Forward DT-VBC", fwd_vbc),
            ("Backward DT-VBC", bwd_vbc),
            ("Forward IBC", fwd_ibc),
            ("Backward IBC", bwd_ibc),
        ]:
            rows.append({
                "system": sys["name"],
                "formulation": label,
                "degree": res["degree"],
                "num_functions": res["num_functions"],
                "success": bool(res["success"]),
                "epsilon": float(res["epsilon"]) if res["success"] else None,
                "runtime_sec": float(res["runtime_sec"]) if res["success"] else None,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(data_dir, "formulation_comparison.csv"), index=False)

    plot_certificate(
        os.path.join(fig_dir, "case1_forward_dt_vbc_synthesized.png"),
        "Case 1 forward DT-VBC",
        chosen["S1"]["forward_dt_vbc"]["coeffs"],
        SYSTEM1["f"], SYSTEM1["domain"], SYSTEM1["X0"], SYSTEM1["Xu"], seed=1
    )
    plot_certificate(
        os.path.join(fig_dir, "case2_backward_dt_vbc_synthesized.png"),
        "Case 2 backward DT-VBC",
        chosen["S2"]["backward_dt_vbc"]["coeffs"],
        SYSTEM2["f"], SYSTEM2["domain"], SYSTEM2["X0"], SYSTEM2["Xu"], seed=2
    )

    with open(os.path.join(data_dir, "comparison_details.json"), "w") as f:
        json.dump(chosen, f, indent=2)

    encoding_summary = run_encoding_case(results_dir)

    summary = {
        "comparison_rows": df.to_dict(orient="records"),
        "encoding_summary": encoding_summary,
    }
    with open(os.path.join(data_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(df.to_string(index=False))
    print(json.dumps(encoding_summary, indent=2))

if __name__ == "__main__":
    main()
