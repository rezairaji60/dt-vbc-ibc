from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from experiments.common import load_yaml, Grid2D, FIGURES, DATA
from experiments.systems import contraction_map, box_mask, iterate_map
from experiments.plot_utils import contour_with_sets_and_trajs


def run(config_path: str | Path) -> dict:
    cfg = load_yaml(config_path)
    grid = Grid2D(**cfg['grid'])
    pts = grid.points()
    xx, yy = grid.mesh()

    init_cfg = cfg['initial_box']
    unsafe_cfg = cfg['unsafe_box']
    init_mask = box_mask(pts, **init_cfg)
    unsafe_mask = box_mask(pts, **unsafe_cfg)

    radius_sq = float(cfg['barrier_radius_sq'])
    # Forward certificate for contraction: B(x)=x^2+y^2-R^2, and B(f(x)) <= B(x)
    vals = pts[:, 0] ** 2 + pts[:, 1] ** 2 - radius_sq

    seeds = np.asarray(cfg['trajectory_seeds'], dtype=float)
    trajectories = [iterate_map(contraction_map, seed, int(cfg['steps'])) for seed in seeds]

    contour_with_sets_and_trajs(
        xx, yy, vals.reshape(xx.shape), pts[init_mask], pts[unsafe_mask], trajectories,
        FIGURES / 'case1_forward_dt_vbc.png',
        'Case 1: forward DT-VBC on a contracting map'
    )

    pd.DataFrame({
        'x': pts[:, 0],
        'y': pts[:, 1],
        'B': vals,
        'is_init': init_mask.astype(int),
        'is_unsafe': unsafe_mask.astype(int),
    }).to_csv(DATA / 'case1_forward_grid.csv', index=False)

    return {
        'certificate': 'B(x)=x1^2+x2^2-R^2',
        'radius_sq': radius_sq,
        'n_init': int(init_mask.sum()),
        'n_unsafe': int(unsafe_mask.sum()),
        'steps': int(cfg['steps']),
    }


if __name__ == '__main__':
    run(Path(__file__).with_name('configs') / 'case1_forward.yaml')
