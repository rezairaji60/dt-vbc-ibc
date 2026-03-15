"""
Author: Reza Iraji
Date:   March 2026
"""

from __future__ import annotations
import json
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dt_vbc.systems_sos import SYSTEMS
from dt_vbc.synthesis_sos import (
    build_forward_dt_vbc_problem,
    build_backward_dt_vbc_problem,
    build_forward_ibc_problem,
    build_backward_ibc_problem,
    solve_built_problem,
)
from dt_vbc.plotting_sos import plot_dt_vbc_components, plot_comparison_bar

SOLVER = os.environ.get('DTVBC_SOLVER', 'SCS')
DEGREE = 2


def try_problem(builder, *args, **kwargs):
    built = builder(*args, **kwargs)
    t0 = time.time()
    out = solve_built_problem(built, solver=SOLVER, verbose=False)
    out['runtime_sec'] = time.time() - t0
    return out


def better(a, b):
    if a is None:
        return True
    if b['status'] not in ('optimal', 'optimal_inaccurate'):
        return False
    if a['status'] not in ('optimal', 'optimal_inaccurate'):
        return True
    return (b['epsilon'] or -1e18) > (a['epsilon'] or -1e18)


def search_all_for_system(name, cfg):
    best = {'Forward DT-VBC': None, 'Backward DT-VBC': None, 'Forward IBC': None, 'Backward IBC': None}
    common = dict(f_exprs=cfg['f'], vars_=cfg['vars'], X=cfg['X'], X0=cfg['X0'], Xu=cfg['Xu'], degree=DEGREE)

    for A in cfg['A_candidates_forward']:
        out = try_problem(build_forward_dt_vbc_problem, **common, A_value=np.array(A, dtype=float), m=2, unsafe_component=0)
        out['system'] = name
        if better(best['Forward DT-VBC'], out):
            best['Forward DT-VBC'] = out

    for A in cfg['A_candidates_backward']:
        out = try_problem(build_backward_dt_vbc_problem, **common, A_value=np.array(A, dtype=float), m=2, init_component=0)
        out['system'] = name
        if better(best['Backward DT-VBC'], out):
            best['Backward DT-VBC'] = out

    for lambdas in cfg['lambda_candidates']:
        out = try_problem(build_forward_ibc_problem, **common, lambdas=lambdas, k=2)
        out['system'] = name
        if better(best['Forward IBC'], out):
            best['Forward IBC'] = out
        out = try_problem(build_backward_ibc_problem, **common, lambdas=lambdas, k=2)
        out['system'] = name
        if better(best['Backward IBC'], out):
            best['Backward IBC'] = out

    return best


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(repo_root, 'results')
    fig_dir = os.path.join(results_dir, 'figures')
    data_dir = os.path.join(results_dir, 'data')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    all_best = {}
    rows = []
    for name, cfg in SYSTEMS.items():
        best = search_all_for_system(name, cfg)
        all_best[name] = best
        for formulation, out in best.items():
            rows.append({
                'system': name,
                'formulation': formulation,
                'degree': out['degree'],
                'functions': out.get('m', out.get('k', 0) + 1),
                'status': out['status'],
                'epsilon': out['epsilon'],
                'runtime_sec': out['runtime_sec'],
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(data_dir, 'sos_comparison.csv'), index=False)
    with open(os.path.join(data_dir, 'sos_details.json'), 'w') as f:
        json.dump(all_best, f, indent=2)

    # Figures only for the best forward S1 and best backward S2 DT-VBCs.
    s1 = all_best['S1']['Forward DT-VBC']
    s2 = all_best['S2']['Backward DT-VBC']
    plot_dt_vbc_components(
        os.path.join(fig_dir, 'case1_forward_dt_vbc_synthesized.pdf'),
        'S1 forward DT-VBC', SYSTEMS['S1']['vars'], DEGREE, s1['coefficients'], ['fwdB0_', 'fwdB1_'],
        SYSTEMS['S1']['f'], SYSTEMS['S1']['domain'], SYSTEMS['S1']['X0_box'], SYSTEMS['S1']['Xu_box'], seed=1,
    )
    plot_dt_vbc_components(
        os.path.join(fig_dir, 'case2_backward_dt_vbc_synthesized.pdf'),
        'S2 backward DT-VBC', SYSTEMS['S2']['vars'], DEGREE, s2['coefficients'], ['bwdB0_', 'bwdB1_'],
        SYSTEMS['S2']['f'], SYSTEMS['S2']['domain'], SYSTEMS['S2']['X0_box'], SYSTEMS['S2']['Xu_box'], seed=2,
    )
    plot_comparison_bar(df, os.path.join(fig_dir, 'comparison_methods.pdf'))
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()