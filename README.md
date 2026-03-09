# DT-VBC / IBC Duality Experiments

This repository contains reproducible Python case studies for the paper:

**Duality of Vector Barrier Certificates and Interpolation-Inspired Barrier Certificates for Discrete-Time Safety Verification**

## Included experiments

1. **Forward DT-VBC on a contracting 2D map**
   - uses an analytic quadratic certificate,
   - plots the zero-level set, initial set, unsafe set, and sample trajectories.

2. **Backward DT-VBC on an expanding 2D map**
   - uses an analytic quadratic certificate,
   - illustrates backward-style safety reasoning.

3. **Backward IBC to backward DT-VBC bridge**
   - visualizes frame inequalities,
   - exports the sparse comparison matrix from the paper.

These examples are intentionally lightweight, reproducible, and easy to adapt to the final systems and figures used in the paper.

## Run all experiments

```bash
python -m experiments.run_all
```

## Outputs

- figures: `results/figures/`
- CSV/JSON summaries: `results/data/`

## Adapting to final paper systems

The current paper draft did not yet specify the final numerical case-study systems and parameter values. This repository therefore provides a clean baseline structure with publication-ready plotting. To swap in your final systems:

- edit the maps in `experiments/systems.py`,
- update the configuration files under `experiments/cases/configs/`,
- rerun `python -m experiments.run_all`.

## Repository layout

```
.
├── experiments/
│   ├── common.py
│   ├── systems.py
│   ├── plot_utils.py
│   ├── run_all.py
│   └── cases/
│       ├── case1_forward_dt_vbc.py
│       ├── case2_backward_dt_vbc.py
│       ├── case3_bibc_bridge.py
│       └── configs/
│           ├── case1_forward.yaml
│           ├── case2_backward.yaml
│           └── case3_bibc.yaml
├── results/
│   ├── data/
│   └── figures/
├── requirements.txt
├── .gitignore
└── LICENSE
```
