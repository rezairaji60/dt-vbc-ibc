from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from experiments.common import load_yaml, Grid2D, FIGURES, DATA
from experiments.systems import expansion_map, annulus_mask, disk_mask, iterate_map
from experiments.plot_utils import contour_with_sets_and_trajs


def run(config_path: str | Path) -> dict:
    cfg = load_yaml(config_path)
    grid = Grid2D(**cfg['grid'])
    pts = grid.points()
    xx, yy = grid.mesh()

    init_mask = annulus_mask(pts, cfg['initial_annulus']['r_lo'], cfg['initial_annulus']['r_hi'])
    unsafe_mask = disk_mask(pts, cfg['unsafe_disk_radius'])

    radius_sq = float(cfg['unsafe_disk_radius']) ** 2
    # Backward certificate for expansion: B(x)=x^2+y^2-r_u^2, and B(x) <= B(f(x))
    vals = pts[:, 0] ** 2 + pts[:, 1] ** 2 - radius_sq

    seeds = np.asarray(cfg['trajectory_seeds'], dtype=float)
    trajectories = [iterate_map(expansion_map, seed, int(cfg['steps'])) for seed in seeds]

    contour_with_sets_and_trajs(
        xx, yy, vals.reshape(xx.shape), pts[init_mask], pts[unsafe_mask], trajectories,
        FIGURES / 'case2_backward_dt_vbc.png',
        'Case 2: backward DT-VBC on an expanding map'
    )

    pd.DataFrame({
        'x': pts[:, 0],
        'y': pts[:, 1],
        'B': vals,
        'is_init': init_mask.astype(int),
        'is_unsafe': unsafe_mask.astype(int),
    }).to_csv(DATA / 'case2_backward_grid.csv', index=False)

    return {
        'certificate': 'B(x)=x1^2+x2^2-r_u^2',
        'unsafe_radius_sq': radius_sq,
        'n_init': int(init_mask.sum()),
        'n_unsafe': int(unsafe_mask.sum()),
        'steps': int(cfg['steps']),
    }


if __name__ == '__main__':
    run(Path(__file__).with_name('configs') / 'case2_backward.yaml')
