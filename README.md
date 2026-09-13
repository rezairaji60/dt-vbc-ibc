# Verified barrier certificates for discrete-time safety

Reviewer-facing implementation for **Beyond path-structured barrier certificates: exact embeddings, degree separation, and verified SOS synthesis**.

## Run everything

Install Julia 1.10.10, clone this repository, and run from its root:

```sh
julia --project=julia_sos julia_sos/run_all.jl
```

This instantiates the pinned environment, executes both test suites, and regenerates the synthesis/replay results in `julia_sos/results/`. CSDP is the default solver; no commercial license is required. Numerical synthesis is entirely Julia/JuMP/SumOfSquares. Solver statuses alone are not accepted as proofs.

To verify the archived certificates without solving another SDP:

```sh
julia --project=julia_sos julia_sos/experiments/replay_all.jl
```

The replay checks archived file hashes, reconstructs obligations from exact benchmark definitions, checks rational Gram matrices and whole-box residual bounds, and requires a valid invariant domain. `evidence/reviewer/` records the tested source revision and resolved environment. New computations do not overwrite that archive.

## Contents

- `julia_sos/src/`: synthesis, exact replay, versioned benchmarks and transformations.
- `julia_sos/test/`: original audit and structural/provenance regression tests.
- `julia_sos/experiments/run_reviewer.jl`: all numerical examples and scalar baselines.
- `paper/main.tex`: complete Elsevier-format manuscript; `paper/README.md` has build instructions.
- `docs/BENCHMARK_PROVENANCE.md`: literature and selection rationale.
- `docs/REVIEWER_RESOLUTION.md`: reviewer concerns mapped to evidence.
- `paper/AUTHOR_CHECKLIST.md`: author approvals and declarations required before submission.

The suite distinguishes four free-synthesis problems (S1, repaired S2, a source-matched BarrierBench contraction, and an explicitly adapted logistic map) from two exact finite-order structural examples. The rotation examples prove an affine-versus-quadratic degree separation for **global scaled IBCs**, not for unrestricted implication IBCs. Their cyclic propagation is exact with zero reserve. No general performance or scalability superiority is claimed.

## Legacy material

The existing Python `src/dt_vbc/`, `experiments/run_all_sos.py`, and historical `results/` are preserved for traceability. Despite their original names, the active historical method is finite-grid collocation, not a full-domain SOS proof. Do not use its margins or infeasibility table as current evidence. `docs/SCL_SIMULATION_AUDIT.md` documents the corrected normalization, reciprocal scales, strict separation, and S2 domain.

This development repository remains private. Provide the source/evidence artifact or reviewer access when submitting; a private URL alone is not a reproducibility artifact. No visibility or licensing change is implied. Acceptance by a journal is not guaranteed by passing tests.
