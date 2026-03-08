from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from experiments.common import load_yaml, FIGURES, DATA
from experiments.plot_utils import line_plot


def run(config_path: str | Path) -> dict:
    cfg = load_yaml(config_path)
    lambdas = np.array(cfg['lambdas'], dtype=float)
    horizon = int(cfg['horizon'])

    # Construct a simple synthetic backward IBC frame sequence along a trajectory.
    # b_{i+1}(f(x)) = 0.8^t and b_i(x) <= lambda_i * b_{i+1}(f(x)) by construction.
    base = np.array([0.8 ** t for t in range(horizon)], dtype=float)
    frames = {}
    for i, lam in enumerate(lambdas[:-1]):
        frames[f'b_{i+1}(f(x_t))'] = base
        frames[f'b_{i}(x_t)'] = np.minimum(lam * base, 0.95 * lam * base)
    frames[f'b_{len(lambdas)-1}(x_t)'] = np.minimum(lambdas[-1] * base, 0.95 * lambdas[-1] * base)

    line_plot(frames, FIGURES / 'case2_bibc_bridge.png', 'Case 2: scaled backward IBC / DT-VBC bridge', 'sample index', 'frame value')

    A = np.zeros((len(lambdas), len(lambdas)))
    for i, lam in enumerate(lambdas[:-1]):
        A[i, i + 1] = lam
    A[-1, -1] = lambdas[-1]

    pd.DataFrame(A).to_csv(DATA / 'case2_sparse_matrix_A.csv', index=False)
    return {'A': A.tolist(), 'lambdas': lambdas.tolist()}


if __name__ == '__main__':
    run(Path(__file__).with_name('configs') / 'case2_bibc.yaml')
