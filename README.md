# Verified barrier certificates for discrete-time safety

Reviewer-facing implementation for **Beyond path-structured barrier certificates: exact embeddings, degree separation, and verified SOS synthesis**.

## Run everything

Install Julia 1.10.10, clone this repository, and run from its root:

```sh
julia --project=julia_sos julia_sos/run_all.jl
```

This instantiates the pinned environment, executes both test suites, and regenerates synthesis/replay results in `julia_sos/results/`. CSDP is the default solver; no commercial license is required. Numerical synthesis is Julia/JuMP/SumOfSquares. Solver statuses alone are not proofs.

Verify archived certificates without solving another SDP:

```sh
julia --project=julia_sos julia_sos/experiments/replay_all.jl
```

Replay checks SHA256 hashes, rebuilds obligations from exact benchmark definitions, checks rational Gram matrices and whole-box residual budgets, and requires invariant domains. `evidence/reviewer/` records the tested source revision and resolved environment. New computations do not overwrite it.

## Scientific scope

The exact bidirectional representation theorem concerns robust, anchored, full-domain **globally scaled** IBCs. The stronger degree theorem concerns **implication-style** IBCs as well: for the finite-order rotation examples, no affine IBC of any finite length exists, whereas cyclic affine VBCs do. An orbit-average argument proves nonexistence; it is not inferred from SDP failures. Quadratic scalar and repeated-frame certificates exist, so the minimum degrees are exactly one versus two. Numerical regression tests support the identities; they are not proof-assistant mechanization.

Four free-synthesis problems comprise S1, repaired S2, a source-matched BarrierBench contraction, and an explicitly adapted logistic map. Scalar baselines are reported honestly; all pass. Two exact finite-order structural witnesses supply the degree distinction, with zero propagation reserve. No general runtime or scalability superiority is claimed.

## Contents

- `julia_sos/src/`: synthesis, exact replay, versioned problems and transformations.
- `julia_sos/test/`: correctness, provenance, transport, tamper rejection and orbit-average tests.
- `julia_sos/experiments/run_reviewer.jl`: complete fixed reviewer suite.
- `paper/main.tex`: Elsevier-format manuscript; `paper/README.md` explains compilation.
- `docs/BENCHMARK_PROVENANCE.md`: literature and selection rationale.
- `docs/REVIEWER_RESOLUTION.md`: concerns mapped to evidence.
- `paper/AUTHOR_CHECKLIST.md`: author checks before submission.

## Legacy

The original Python source, experiment and historical results remain for traceability. Despite their names, the active old method is finite-grid collocation, not a full-domain SOS proof. Its margins/infeasibility table are not current evidence. See `docs/SCL_SIMULATION_AUDIT.md` for the earlier diagnosis and `docs/IMPLICATION_OBSTRUCTION.md` for the strengthened theorem.

The development repository remains private. Provide the full source/evidence artifact or reviewer access with submission; a private URL alone is insufficient. No visibility or licensing change is implied, and journal acceptance cannot be guaranteed by passing tests.
