"""
Author: Reza Iraji
Date:   March 2026
"""
import numpy as np
import matplotlib.pyplot as plt
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

    fig, axes = plt.subplots(1, len(coeffs), figsize=(5 * len(coeffs), 4))
    if len(coeffs) == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        Z = vals[..., i]
        ax.contourf(X, Y, Z, levels=25)
        ax.contour(X, Y, Z, levels=[0.0], linewidths=1.5)
        x0, x1, y0, y1 = X0_box
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], linewidth=2)
        u0, u1, v0, v1 = Xu_box
        ax.plot([u0, u1, u1, u0, u0], [v0, v0, v1, v1, v0], linewidth=2)
        for tr in trajs:
            ax.plot(tr[:, 0], tr[:, 1], linewidth=1.0)
        ax.set_title(f"{title_prefix} component {i+1}")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
