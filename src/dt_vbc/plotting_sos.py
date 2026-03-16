"""
Paper-quality plotting utilities for collocation/SOS surrogate experiments.

Author: Reza Iraji
Date:   March 2026
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from dt_vbc.polynomials import eval_poly_2d, monomial_exponents_2d
from dt_vbc.systems_sos import sample_box, simulate_trajectories


def _box_to_rect(box, **kwargs):
    (xlo, xhi), (ylo, yhi) = box
    return Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fill=False, **kwargs)


def plot_dt_vbc_components(
    output_path: str,
    title: str,
    vars_xy: Tuple[str, str],
    degree: int,
    coefficients: Dict[str, np.ndarray],
    coeff_names: Sequence[str],
    dynamics: Callable[[np.ndarray], np.ndarray],
    domain,
    x0_box,
    xu_box,
    seed: int = 0,
    horizon: int = 18,
    n_traj: int = 8,
    grid_n: int = 220,
):
    exponents = monomial_exponents_2d(degree)
    (xlo, xhi), (ylo, yhi) = domain
    xs = np.linspace(xlo, xhi, grid_n)
    ys = np.linspace(ylo, yhi, grid_n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])

    x0_samples = sample_box(x0_box, n_traj, seed=seed)
    trajs = simulate_trajectories(dynamics, x0_samples, horizon=horizon)

    fig, axes = plt.subplots(1, len(coeff_names), figsize=(6.2 * len(coeff_names), 5.2))
    if len(coeff_names) == 1:
        axes = [axes]

    last_im = None
    comp_labels = [rf"$B_{{{i+1}}}(x)$" for i in range(len(coeff_names))]

    for ax, name, comp_label in zip(axes, coeff_names, comp_labels):
        coeff = coefficients[name]
        z = eval_poly_2d(pts, coeff, exponents).reshape(X.shape)

        vabs = max(abs(float(np.nanmin(z))), abs(float(np.nanmax(z))), 1e-6)
        levels = np.linspace(-vabs, vabs, 19)
        last_im = ax.contourf(X, Y, z, levels=levels, cmap="coolwarm", extend="both")

        has_zero = float(np.nanmin(z)) <= 0.0 <= float(np.nanmax(z))
        if has_zero:
            ax.contour(X, Y, z, levels=[0.0], colors="white", linewidths=3.0)
            ax.contour(X, Y, z, levels=[0.0], colors="black", linewidths=1.4)
        else:
            ax.text(
                0.03, 0.97, "0-level set outside plot",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5")
            )

        ax.add_patch(_box_to_rect(x0_box, edgecolor="tab:blue", linewidth=2.4, label=r"$X_0$"))
        ax.add_patch(_box_to_rect(xu_box, edgecolor="tab:orange", linewidth=2.4, label=r"$X_u$"))

        for k, tr in enumerate(trajs):
            ax.plot(tr[:, 0], tr[:, 1], color="black", linewidth=1.0, alpha=0.8)

        ax.set_title(comp_label, fontsize=14)
        ax.set_xlabel(vars_xy[0], fontsize=13)
        ax.set_ylabel(vars_xy[1], fontsize=13)
        ax.set_aspect("equal")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.grid(alpha=0.15)

    fig.suptitle(title, fontsize=16, y=0.98)

    handles = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="tab:blue", linewidth=2.4, label=r"Initial set $X_0$"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="tab:orange", linewidth=2.4, label=r"Unsafe set $X_u$"),
        plt.Line2D([0], [0], color="black", linewidth=1.2, label="Sample trajectories"),
        plt.Line2D([0], [0], color="black", linewidth=1.4, linestyle="-", label=r"0-level set"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.91))

    cax = fig.add_axes([0.92, 0.18, 0.018, 0.62])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Certificate value", fontsize=11)

    fig.tight_layout(rect=[0.02, 0.05, 0.90, 0.88])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)