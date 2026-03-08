from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from experiments.common import load_yaml, Grid2D, FIGURES, DATA
from experiments.systems import linear_nonlinear_map, box_mask
from experiments.sampled_vbc import fit_scalar_barrier_sampled, eval_scalar
from experiments.plot_utils import contour_plot


def run(config_path: str | Path) -> dict:
    cfg = load_yaml(config_path)
    grid = Grid2D(**cfg['grid'])
    pts = grid.points()
    next_pts = linear_nonlinear_map(pts)

    init_cfg = cfg['initial_box']
    unsafe_cfg = cfg['unsafe_box']
    init_mask = box_mask(pts, **init_cfg)
    unsafe_mask = box_mask(pts, **unsafe_cfg)

    alpha_f = cfg['alpha_forward']
    alpha_b = cfg['alpha_backward']

    cf = fit_scalar_barrier_sampled(pts, next_pts, init_mask, unsafe_mask, alpha_f)
    cb = fit_scalar_barrier_sampled(pts, next_pts, unsafe_mask, init_mask, alpha_b)

    xx, yy = grid.mesh()
    vf = eval_scalar(cf, pts).reshape(xx.shape)
    vb = eval_scalar(cb, pts).reshape(xx.shape)

    init_pts = pts[init_mask]
    unsafe_pts = pts[unsafe_mask]

    contour_plot(xx, yy, vf, init_pts, unsafe_pts, FIGURES / 'case1_forward_dt_vbc.png', 'Case 1: sampled forward DT-VBC')
    contour_plot(xx, yy, vb, unsafe_pts, init_pts, FIGURES / 'case1_backward_dt_vbc.png', 'Case 1: sampled backward DT-VBC')

    df = pd.DataFrame({
        'coeff_forward': cf,
        'coeff_backward': cb,
    })
    df.to_csv(DATA / 'case1_coefficients.csv', index=False)

    return {
        'forward_coeffs': cf.tolist(),
        'backward_coeffs': cb.tolist(),
        'n_points': int(len(pts)),
        'n_init': int(init_mask.sum()),
        'n_unsafe': int(unsafe_mask.sum()),
    }


if __name__ == '__main__':
    run(Path(__file__).with_name('configs') / 'case1_linear2d.yaml')
