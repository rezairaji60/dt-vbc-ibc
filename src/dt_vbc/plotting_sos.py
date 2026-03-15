"""
Author: Reza Iraji
Date:   March 2026
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from .common import make_grid, rollout_box
from .poly_basis import monomials


def build_poly_from_solution(vars_, degree, prefix, coeff_dict):
    mons = monomials(vars_, degree)
    expr = sp.Integer(0)
    for idx, m in enumerate(mons):
        expr += coeff_dict.get(f"{prefix}{idx}", 0.0) * m
    return sp.expand(expr)


def evaluate_expr(expr, vars_, X, Y):
    f = sp.lambdify(vars_, expr, 'numpy')
    Z = f(X, Y)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 0:
        Z = np.full_like(X, float(Z), dtype=float)
    return Z


def system_func_from_sympy(exprs, vars_):
    f = sp.lambdify(vars_, exprs, 'numpy')

    def wrapped(x):
        out = f(*x)
        return np.array(out, dtype=float).reshape(-1)

    return wrapped


def plot_dt_vbc_components(path, title_prefix, vars_, degree, coeffs, prefixes,
                           f_exprs, bounds, X0_box, Xu_box, seed=0):
    X, Y, _ = make_grid(bounds, 220)
    exprs = [build_poly_from_solution(vars_, degree, p, coeffs) for p in prefixes]
    vals = [evaluate_expr(e, vars_, X, Y) for e in exprs]
    f_num = system_func_from_sympy(f_exprs, vars_)
    trajs = rollout_box(f_num, X0_box, 10, 15, np.random.default_rng(seed))

    fig, axes = plt.subplots(1, len(exprs), figsize=(5.1 * len(exprs), 4.2))
    if len(exprs) == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        Z = vals[i]
        im = ax.contourf(X, Y, Z, levels=28, cmap='coolwarm')
        ax.contour(X, Y, Z, levels=[0.0], colors='black', linewidths=1.8)

        x0, x1, y0, y1 = X0_box
        u0, u1, v0, v1 = Xu_box
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, edgecolor='tab:blue', linewidth=2.0))
        ax.add_patch(plt.Rectangle((u0, v0), u1 - u0, v1 - v0,
                                   fill=False, edgecolor='tab:orange', linewidth=2.0))

        for tr in trajs:
            ax.plot(tr[:, 0], tr[:, 1], color='black', alpha=0.35, linewidth=1.0)

        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_title(f'{title_prefix} component {i+1}')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_bar(df, path):
    order = ['Forward DT-VBC', 'Backward DT-VBC', 'Forward IBC', 'Backward IBC']
    pivot = df.pivot(index='formulation', columns='system', values='epsilon').reindex(order)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(order))
    w = 0.35

    for j, col in enumerate(pivot.columns):
        ax.bar(x + (j - 0.5) * w, pivot[col].values, width=w, label=col)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha='right')
    ax.set_ylabel(r'$\epsilon$')
    ax.set_title('SOS synthesis margins')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)