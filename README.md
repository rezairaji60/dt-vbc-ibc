# DT-VBC / IBC Case Studies for CDC 2026

This repository contains a GitHub-ready experimental package for the paper:

**Duality of Vector Barrier Certificates and Interpolation-Inspired Barrier Certificates for Discrete-Time Safety Verification**

## What is included

- synthesis of:
  - forward DT-VBCs
  - backward DT-VBCs
  - forward IBCs
  - backward IBCs
- a structural encoding example for:
  - scaled backward IBC -> forward DT-VBC
- figure generation
- CSV/JSON summaries
- an Overleaf-ready LaTeX section for the case studies

## Method

The paper presents SOS-style synthesis for polynomial systems.  
This repository implements a lightweight **sample-based LP synthesis prototype** using fixed quadratic templates and dense collocation grids. It is intended for:
- reproducing the case-study figures,
- comparing formulations,
- and supporting the paper's experimental section.

It is **not** a full SOS/SDP implementation.

## Repository layout

- `src/dt_vbc/`: reusable synthesis and plotting code
- `experiments/`: experiment driver
- `results/`: generated figures and tables
- `paper/`: Overleaf-ready LaTeX section

## Setup

```bash
pip install -r requirements.txt
python experiments/run_all.py
```

## Outputs

- `results/figures/*.png`
- `results/data/formulation_comparison.csv`
- `results/data/summary.json`
- `paper/experiments_section.tex`
