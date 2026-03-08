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

    base = np.array([0.85 ** t for t in range(horizon)], dtype=float)
    series = {}
    table = {'t': np.arange(horizon)}
    for i, lam in enumerate(lambdas[:-1]):
        next_vals = base
        curr_vals = 0.9 * lam * base
        series[f'$b_{i+1}(f(x_t))$'] = next_vals
        series[f'$b_{i}(x_t)$'] = curr_vals
        table[f'b{i}_x'] = curr_vals
        table[f'b{i+1}_fx'] = next_vals
    last_vals = 0.9 * lambdas[-1] * base
    series[f'$b_{len(lambdas)-1}(x_t)$'] = last_vals
    table[f'b{len(lambdas)-1}_x'] = last_vals

    line_plot(series, FIGURES / 'case3_bibc_bridge.png', 'Case 3: scaled backward IBC to backward DT-VBC bridge', 'sample index', 'frame value')

    A = np.zeros((len(lambdas), len(lambdas)))
    for i, lam in enumerate(lambdas[:-1]):
        A[i, i + 1] = lam
    A[-1, -1] = lambdas[-1]

    pd.DataFrame(table).to_csv(DATA / 'case3_bibc_series.csv', index=False)
    pd.DataFrame(A).to_csv(DATA / 'case3_sparse_matrix_A.csv', index=False)
    return {'A': A.tolist(), 'lambdas': lambdas.tolist(), 'horizon': horizon}


if __name__ == '__main__':
    run(Path(__file__).with_name('configs') / 'case3_bibc.yaml')
