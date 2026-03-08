from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def contour_with_sets_and_trajs(
    xx: np.ndarray,
    yy: np.ndarray,
    values: np.ndarray,
    init_pts: np.ndarray,
    unsafe_pts: np.ndarray,
    trajectories: list[np.ndarray],
    outpath: str | Path,
    title: str,
    xlabel: str = '$x_1$',
    ylabel: str = '$x_2$',
) -> None:
    plt.figure(figsize=(5.4, 4.4))
    plt.contourf(xx, yy, values, levels=40)
    plt.contour(xx, yy, values, levels=[0.0], linewidths=2.0)
    if len(init_pts):
        plt.scatter(init_pts[:, 0], init_pts[:, 1], s=7, marker='o', label='initial set')
    if len(unsafe_pts):
        plt.scatter(unsafe_pts[:, 0], unsafe_pts[:, 1], s=7, marker='x', label='unsafe set')
    for idx, tr in enumerate(trajectories):
        plt.plot(tr[:, 0], tr[:, 1], linewidth=1.4, label='trajectory' if idx == 0 else None)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def line_plot(series_dict: dict[str, np.ndarray], outpath: str | Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(5.4, 4.0))
    for label, values in series_dict.items():
        plt.plot(values, linewidth=2.0, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()
