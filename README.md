# Duality and complementarity of vector and interpolation-inspired barrier certificates

Computational and analytical baseline for the manuscript with fixed paper title:

**Duality and Complementarity of Vector and Interpolation-Inspired Barrier Certificates for Safety Verification: Toward Reduced Conservatism and Complexity**

The repository separates three claims that must not be conflated:

1. **Structural duality:** robust anchored full-domain globally scaled IBCs correspond to path-structured VBCs under the stated sign/scaling map. This exact conversion preserves polynomial degree and can transport an SOS proof without another SDP solve.
2. **VBC-side complementarity:** on finite-order rotations, cyclic vector coupling admits affine VBCs while the orbit-average theorem excludes every affine implication IBC of finite length; an invariant quadratic IBC exists, so the minimum degrees are exactly 1 versus 2 in the stated classes.
3. **IBC-side complementarity:** the author-constructed `ImplicationGap1D` example has an affine implication IBC, while a convex-combination theorem excludes every affine constant-comparison VBC, regardless of finite component count, on the fixed domain. A quartic scalar global VBC is verified as a recovery witness; degree four is an upper bound, not a minimum theorem.

The scientific message is therefore not universal dominance and not that exact conversion reduces degree. The intended workflow is to identify whether a failed low-degree search is limited by the polynomial template or by the propagation structure, and to consider the complementary formulation before automatically increasing degree.

## Reproduce the fixed computational evidence

Use Julia **1.10.10** and Git. From the repository root, run:

```sh
julia --startup-file=no --project=julia_sos julia_sos/run_all.jl
```

With a standalone Windows installation, invoke that executable directly. Do not update the pinned `Manifest.toml` merely to silence a version warning.

The command checks release fingerprints, runs scientific and portability tests, replays the immutable SOS archive, regenerates the fixed numerical table, verifies the analytical complementarity report, independently replays every fresh global-scaled/VBC SOS proof, and replays the separate affine implication-IBC proof bundle. It succeeds only after printing `REVIEWER_RELEASE_VERIFIED`.

To check the original committed SOS proofs without another solve:

```sh
julia --startup-file=no --project=julia_sos julia_sos/experiments/replay_all.jl
```

The original 36-proof archive remains immutable. The complementarity evidence is stored separately under `evidence/complementarity/`; it is analytical evidence, not a retroactive rewrite of the numerical archive.

## Results and limits

The numerical table remains 23 identified rows: 16 degree-two multi-function successes, four degree-two scalar successes, two exact affine cyclic VBC witnesses, and one intentional normalization failure. The added complementarity report is intentionally separate from solver-result rows so that analytical nonexistence is never inferred from an optimizer failure.

The repository does **not** claim that lower polynomial degree always means lower wall-clock time. `complexity_profile` reports transparent dense monomial/Gram sizing only. Component count, multiplier degrees, sparsity, conditioning, and search strategy also determine cost.

The affine VBC obstruction in `ImplicationGap1D` assumes a fixed verification domain and a constant nonnegative comparison matrix. State-dependent comparison maps or a changed domain are different certificate classes/problems. The finite-order rotation result excludes affine implication IBCs but not nonlinear frames. These boundaries are part of the paper story.

## Reviewer navigation

- [Duality/complementarity statement and exact examples](docs/COMPLEMENTARITY.md)
- [Freeze scope and paper handoff](docs/REPOSITORY_FREEZE.md)
- [Execution evidence](docs/EXECUTION_EVIDENCE.md)
- [Julia commands, outputs and interpretation](julia_sos/README.md)
- [Benchmark provenance and prior-work limits](docs/BENCHMARK_PROVENANCE.md)
- [Reviewer concern-to-evidence map](docs/REVIEWER_RESOLUTION.md)
- [Finite-order IBC obstruction](docs/IMPLICATION_OBSTRUCTION.md)
- [Manuscript handoff](paper/README.md) and [author checks](paper/AUTHOR_CHECKLIST.md)

Historical Python files are retained as collocation-era material and are not current SOS evidence. Historical diagnostics are under `docs/history/`.

The repository remains private. For external review, export the complete source/evidence package or arrange authorized access. This baseline supports coauthor review; it is not an acceptance guarantee or a substitute for independent theorem/novelty review.

## Export for a reviewer without repository access

```sh
git archive --format=zip --output=../SCL_reviewer_source.zip HEAD README.md .gitattributes .github SOURCE_REVISION.txt docs julia_sos evidence paper
```

The exported package is checked by CI without a source `.git` directory. It contains the release workflow as a provenance artifact; running the Julia verifier does not require a GitHub account.
