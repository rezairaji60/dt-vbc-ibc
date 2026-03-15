"""
Author: Reza Iraji
Date:   March 2026
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from .common import make_grid, rollout_box, phi2

def eval_components(coeffs, pts):
    P = phi2(pts)
    vals = []
    for c in coeffs:
        vals.append(P @ np.asarray(c))
    return np.stack(vals, axis=-1)

def plot_certificate(fig_path, title_prefix, coeffs, f, bounds, X0_box, Xu_box, seed=0):
    rng = np.random.default_rng(seed)

    X, Y, pts = make_grid(bounds, 220)
    vals = eval_components(coeffs, pts).reshape(X.shape + (len(coeffs),))
    trajs = rollout_box(f, X0_box, count=12, steps=25, rng=rng)

    fig, axes = plt.subplots(1, len(coeffs), figsize=(5.3 * len(coeffs), 4.2))
    if len(coeffs) == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        Z = vals[..., i]

        # Background color map: value of B_i(x)
        im = ax.contourf(X, Y, Z, levels=30, cmap="coolwarm")

        # Zero level set
        ax.contour(X, Y, Z, levels=[0.0], colors="black", linewidths=2)

        # Initial set X0
        x0, x1, y0, y1 = X0_box
        init_rect = Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor="tab:blue", linewidth=2.5
        )
        ax.add_patch(init_rect)

        # Unsafe set Xu
        u0, u1, v0, v1 = Xu_box
        unsafe_rect = Rectangle(
            (u0, v0), u1 - u0, v1 - v0,
            fill=False, edgecolor="tab:orange", linewidth=2.5
        )
        ax.add_patch(unsafe_rect)

        # Sample trajectories
        for tr in trajs:
            ax.plot(tr[:, 0], tr[:, 1], color="black", linewidth=1.0, alpha=0.9)

        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_title(f"{title_prefix} component {i+1}")

        # Legend
        legend_handles = [
            Line2D([0], [0], color="black", linewidth=2, label=r"$B_i(x)=0$"),
            Line2D([0], [0], color="tab:blue", linewidth=2.5, label=r"Initial set $X_0$"),
            Line2D([0], [0], color="tab:orange", linewidth=2.5, label=r"Unsafe set $X_u$"),
            Line2D([0], [0], color="black", linewidth=1.0, label="Sample trajectories"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(rf"Value of $B_{{{i+1}}}(x)$")

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)