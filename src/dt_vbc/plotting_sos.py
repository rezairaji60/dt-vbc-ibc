"""
Paper-quality plotting utilities for collocation/SOS surrogate experiments.

Author: Reza Iraji
Date:   March 2026
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from dt_vbc.polynomials import eval_poly_2d, monomial_exponents_2d
from dt_vbc.systems_sos import sample_box, simulate_trajectories


def _box_to_rect(box, **kwargs):
    (xlo, xhi), (ylo, yhi) = box
    return Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fill=False, **kwargs)


def _make_dynamics_from_strings(vars_xy: Sequence[str], f_exprs: Sequence[str]) -> Callable[[np.ndarray], np.ndarray]:
    if len(vars_xy) != 2 or len(f_exprs) != 2:
        raise ValueError("This plotting utility currently supports 2D systems only.")

    v0, v1 = vars_xy
    expr0, expr1 = f_exprs

    allowed_globals = {
        "__builtins__": {},
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "abs": np.abs,
    }

    def dynamics(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        x0 = pts[:, 0]
        x1 = pts[:, 1]
        local_env = {v0: x0, v1: x1}
        y0 = eval(expr0, allowed_globals, local_env)
        y1 = eval(expr1, allowed_globals, local_env)
        return np.column_stack([np.asarray(y0, dtype=float), np.asarray(y1, dtype=float)])

    return dynamics


def _resolve_coeff_name(coefficients: Dict[str, np.ndarray], prefix: str) -> str:
    if prefix in coefficients:
        return prefix

    matches = [k for k in coefficients.keys() if k.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]

    # fallback: exact stem without underscore issues, e.g. "fwdB0_" vs "fwdB0"
    stem = prefix.rstrip("_")
    matches = [k for k in coefficients.keys() if k == stem or k.startswith(stem)]
    if len(matches) >= 1:
        return matches[0]

    raise KeyError(f"Could not resolve coefficient name/prefix '{prefix}' from keys {list(coefficients.keys())}")


def plot_dt_vbc_components(
    filename: str,
    title: str,
    vars: Sequence[str],
    degree: int,
    coefficients: Dict[str, np.ndarray],
    prefixes: Sequence[str],
    f: Sequence[str],
    domain,
    X0_box,
    Xu_box,
    seed: int = 0,
    grid_n: int = 301,
    trajectory_horizon: int = 25,
    trajectory_count: int = 12,
):
    """
    Interface matched to experiments/run_all_sos.py.
    """

    coeff_names = [_resolve_coeff_name(coefficients, p) for p in prefixes]
    exponents = monomial_exponents_2d(degree)
    dynamics = _make_dynamics_from_strings(vars, f)

    (xlo, xhi), (ylo, yhi) = domain
    xs = np.linspace(xlo, xhi, grid_n)
    ys = np.linspace(ylo, yhi, grid_n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])

    x0_samples = sample_box(X0_box, trajectory_count, seed=seed)
    trajs = simulate_trajectories(dynamics, x0_samples, trajectory_horizon)

    n_panels = len(coeff_names)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 5.0), constrained_layout=False)
    if n_panels == 1:
        axes = [axes]

    last_im = None

    for i, (ax, cname) in enumerate(zip(axes, coeff_names)):
        coeff = np.asarray(coefficients[cname], dtype=float).reshape(-1)
        z = eval_poly_2d(pts, coeff, exponents).reshape(X.shape)

        zmin = float(np.nanmin(z))
        zmax = float(np.nanmax(z))
        vabs = max(abs(zmin), abs(zmax), 1e-6)
        levels = np.linspace(-vabs, vabs, 21)

        last_im = ax.contourf(
            X, Y, z,
            levels=levels,
            cmap="coolwarm",
            extend="both",
        )

        has_zero = (zmin <= 0.0 <= zmax)
        if has_zero:
            ax.contour(X, Y, z, levels=[0.0], colors="white", linewidths=3.0)
            ax.contour(X, Y, z, levels=[0.0], colors="black", linewidths=1.2)
        else:
            ax.text(
                0.03, 0.97, "0-level set outside plot",
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5"),
            )

        ax.add_patch(_box_to_rect(X0_box, edgecolor="tab:blue", linewidth=2.6))
        ax.add_patch(_box_to_rect(Xu_box, edgecolor="tab:orange", linewidth=2.6))

        for tr in trajs:
            ax.plot(tr[:, 0], tr[:, 1], color="black", linewidth=1.0, alpha=0.8)

        ax.set_title(rf"$B_{{{i+1}}}(x)$", fontsize=13)
        ax.set_xlabel(vars[0], fontsize=12)
        if i == 0:
            ax.set_ylabel(vars[1], fontsize=12)

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)

    fig.suptitle(title, fontsize=16, y=0.97)

    legend_handles = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="tab:blue", linewidth=2.6, label=r"Initial set $X_0$"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="tab:orange", linewidth=2.6, label=r"Unsafe set $X_u$"),
        Line2D([0], [0], color="black", linewidth=1.0, label="Sample trajectories"),
        Line2D([0], [0], color="black", linewidth=1.2, label=r"0-level set"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.91),
    )

    # Put colorbar fully outside the axes so it does not cut through the plots
    cax = fig.add_axes([0.92, 0.18, 0.018, 0.62])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Certificate value", fontsize=11)

    fig.subplots_adjust(top=0.83, right=0.89, wspace=0.18)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)