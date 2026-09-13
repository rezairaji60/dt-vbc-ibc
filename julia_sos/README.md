# Julia implementation for the duality/complementarity paper

Fixed paper title:

**Duality and Complementarity of Vector and Interpolation-Inspired Barrier Certificates for Safety Verification: Toward Reduced Conservatism and Complexity**

The implementation deliberately separates numerical SOS evidence from analytical nonexistence/degree arguments.

## Supported reproduction

Use Julia 1.10.10 and Git. From the repository root:

```sh
julia --startup-file=no --project=julia_sos julia_sos/run_all.jl
```

A standalone Julia 1.10.10 executable is valid; Juliaup registration is optional. Do not update or re-resolve the pinned environment merely to suppress a version warning.

The single command checks source/evidence fingerprints, scientific tests, Windows checkout/repair tests, release-contract negative tests, the immutable SOS archive, all fixed numerical experiments, the deterministic complementarity report, exact replay of every fresh globally scaled/VBC proof, and exact replay of the separate affine implication-IBC bundle. `REVIEWER_RELEASE_VERIFIED` appears only after every stage passes.

Outputs use a fresh `julia_sos/results/run-*` directory. The immutable `evidence/reviewer/` archive and the committed `evidence/complementarity/` analytical report are never overwritten.

## Evidence layers

### 1. Scaled/path structural duality

`src/obligations.jl`, `src/exact.jl`, and the transport tests implement the robust anchored full-domain globally scaled subclass. An IBC proof is sign/scaling-transformed into the corresponding path-structured VBC proof and replayed without another solve. Degree is preserved.

### 2. VBC-side complementarity

`Rotation2` and `Rotation4` have exact affine cyclic VBC witnesses. `rotation_degree_report` verifies the finite-order/spectral hypotheses used by the orbit-average theorem and verifies the invariant quadratic `x1^2+x2^2-3/2` that supplies a degree-two IBC. The code checks the concrete hypotheses/witnesses; the universal theorem remains mathematical text to be independently reviewed.

### 3. IBC-side complementarity

`ImplicationGap1D` uses `x+ = x(x+1)/2` on `[-1,1]`. `exact_implication_gap` verifies:

- an affine implication-style IBC with frame `b=x`, an exact conditional SOS identity, and an independently replayed implication-proof bundle,
- the exact convex-combination data used by the theorem excluding affine constant-comparison VBCs of any finite component count on the fixed domain,
- a degree-four scalar global VBC recovery witness.

The affine nonexistence claim is theorem-based, not a solver return. The quartic witness is an upper bound; degree four is not claimed minimal. The implication verifier checks a fixed certificate and fixed SOS multipliers; joint search over an unknown frame and unknown state-dependent multiplier would be bilinear and is not labeled as a convex SDP.

### 4. Complexity accounting

`complexity_profile` reports dense monomial counts and degree-matched Gram dimensions. It is transparent combinatorial sizing, not a runtime predictor. Component count, multiplier degrees, sparsity, conditioning and search strategy remain important.

## Numerical inventory

The historical fixed numerical table remains 23 rows: 16 degree-two multi-function exact positives, four degree-two scalar exact positives, two exact affine cyclic VBC witnesses, and one intentional normalization negative. Eight numerical IBC proof transports are replayed without resynthesis.

The generated `complementarity_report.json` is **not** another solver-result row. It is separately validated analytical evidence. `ImplicationGap1D_forward_implication_ibc_analytical.json` is a separate exact proof bundle for the affine implication witness.

## Independent replay

Original SOS archive only:

```sh
julia --startup-file=no --project=julia_sos julia_sos/experiments/replay_all.jl
```

Release/evidence preflight:

```sh
julia --startup-file=no --project=julia_sos julia_sos/experiments/release_check.jl
```

An old Windows checkout with archive line-ending changes should follow `../docs/WINDOWS_ARCHIVE_REPLAY.md`; expected hashes must never be regenerated to hide a mismatch.

See `../docs/COMPLEMENTARITY.md` for the scientific interpretation and `../docs/REPOSITORY_FREEZE.md` for the implementation/paper boundary.
