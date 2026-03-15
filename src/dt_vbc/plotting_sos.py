"""
Author: Reza Iraji
Date:   March 2026
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from .polynomials import eval_poly_2d, monomial_exponents_2d
from .systems_sos import sample_box, simulate_trajectories


Box = Tuple[Tuple[float, float], Tuple[float, float]]


def _eval_on_grid(
    coeffs: np.ndarray,
    degree: int,
    box: Box,
    n: int = 240,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    (xlo, xhi), (ylo, yhi) = box
    xs = np.linspace(xlo, xhi, n)
    ys = np.linspace(ylo, yhi, n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    exps = monomial_exponents_2d(degree)
    z = eval_poly_2d(pts, coeffs, exps)
    if np.isscalar(z) or getattr(z, "ndim", 1) == 0:
        z = np.full(X.shape, float(z))
    else:
        z = np.asarray(z).reshape(X.shape)
    return X, Y, z


def _add_box(ax: plt.Axes, box: Box, color: str, label: str) -> None:
    (xlo, xhi), (ylo, yhi) = box
    rect = patches.Rectangle(
        (xlo, ylo),
        xhi - xlo,
        yhi - ylo,
        linewidth=2.0,
        edgecolor=color,
        facecolor="none",
        label=label,
    )
    ax.add_patch(rect)


def plot_dt_vbc_components(
    outfile: str,
    title: str,
    degree: int,
    coeffs_map: Dict[str, np.ndarray],
    coeff_names: Sequence[str],
    dynamics: Callable[[np.ndarray], np.ndarray],
    domain: Box,
    x0_box: Box,
    xu_box: Box,
    seed: int = 0,
    n_grid: int = 220,
) -> None:
    fig, axes = plt.subplots(1, len(coeff_names), figsize=(5.2 * len(coeff_names), 4.4), squeeze=False)
    axes = axes[0]

    x0_samples = sample_box(x0_box, n=10, seed=seed)
    trajectories = simulate_trajectories(dynamics, x0_samples, horizon=25)

    for ax, name in zip(axes, coeff_names):
        coeffs = coeffs_map[name]
        X, Y, Z = _eval_on_grid(coeffs, degree, domain, n=n_grid)

        im = ax.contourf(X, Y, Z, levels=30, cmap="coolwarm")
        try:
            ax.contour(X, Y, Z, levels=[0.0], colors="k", linewidths=1.8)
        except Exception:
            pass

        for traj in trajectories:
            ax.plot(traj[:, 0], traj[:, 1], color="k", linewidth=0.8, alpha=0.6)

        _add_box(ax, x0_box, "tab:blue", r"$X_0$")
        _add_box(ax, xu_box, "tab:orange", r"$X_u$")

        ax.set_title(name)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_aspect("equal")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    fig.suptitle(title)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.03, pad=0.04)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def plot_bar_comparison(
    outfile: str,
    rows: list[dict],
) -> None:
    labels = [f"{r['system']} / {r['formulation']}" for r in rows if r["epsilon"] is not None]
    values = [float(r["epsilon"]) for r in rows if r["epsilon"] is not None]

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"collocation margin $\varepsilon$")
    ax.set_title("Comparison across formulations")
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)