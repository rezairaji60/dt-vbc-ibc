# Julia reviewer implementation

The current implementation is on `main`; no deleted development branch is required. Scientific inputs and the original proof archive are identified in `../docs/REVIEWER_RELEASE.toml`.

## Supported reproduction

Use Julia 1.10.10 and Git. Run from the repository root:

```sh
julia --startup-file=no --project=julia_sos julia_sos/run_all.jl
```

A standalone Julia executable is equally valid; Juliaup registration is optional. Do not use a different default Julia version and resolve/update the pinned Manifest to silence a warning. The first run downloads dependencies and precompiles them; later runs reuse the installed environment. Tests create temporary local Git fixtures without credentials or network access.

The single command performs all stages: input fingerprint/hash checks; scientific tests; Windows-style checkout/repair tests; result-contract negative tests; original archive replay; fixed numerical experiments; exact replay of all freshly generated proofs; final input checks. Any failed stage exits nonzero. `REVIEWER_RELEASE_VERIFIED` is printed only after every stage passes.

Outputs go to a fresh `julia_sos/results/run-*` directory. `results/LATEST_RUN.json` points to the latest completed run, so an interrupted attempt cannot masquerade as a successful one. Previous outputs and `evidence/reviewer/` are not overwritten. Use the `reviewer_summary.json` in the printed directory, not an old `results/summary.json` from an earlier release.

## Independent replay

```sh
julia --startup-file=no --project=julia_sos julia_sos/experiments/replay_all.jl
```

This checks all 69 indexed raw-byte hashes and mathematically replays 36 committed proof bundles. It does not run synthesis. A separate release preflight also checks the archive index fingerprint and nine frozen scientific source/environment files:

```sh
julia --startup-file=no --project=julia_sos julia_sos/experiments/release_check.jl
```

For a single newly generated proof:

```sh
julia --startup-file=no --project=julia_sos julia_sos/experiments/replay.jl PATH_TO_CERTIFICATE.json
```

For an old Windows checkout that fails a digest check, follow `../docs/WINDOWS_ARCHIVE_REPLAY.md`. The repair is explicit, accepts only CRLF expansion, validates all files before writing, and backs up originals. It never regenerates the index or bypasses mathematics. Fresh Git checkouts and the reviewer ZIP preserve archived bytes.

## Mathematical implementation

`src/obligations.jl` defines robust anchored separation and full-domain propagation. `src/synthesis.jl` constructs explicit PSD Gram matrices and polynomial coefficient identities with fixed comparison parameters. `src/exact.jl` verifies rational PSD, bounds full-box polynomial residuals, constructs analytical S1 witnesses and transports complete IBC proofs. `src/problems.jl` and `src/benchmarks.jl` bind exact nominal benchmark data and domain proofs. `src/structural.jl` provides the analytical cyclic examples and scaling identities.

The default solver is CSDP. Alternate solvers are not part of this frozen result set. `EXACT_RATIONAL_VERIFIED` means successful checks by this exact-arithmetic program, not proof-assistant mechanization. `NO_CERTIFIED_CANDIDATE` and `UNVERIFIED_CANDIDATE` never mean general nonexistence. The original S2 domain is deliberately `NOT_INVARIANT`; the revised problem is explicitly named `S2_repaired`.

## Fixed experiment inventory

S1, S2_repaired, BB_rotation and Logistic_adapted each have four three-function degree-two searches and one scalar forward baseline. Rotation2 and Rotation4 are separate analytical degree-one cyclic witnesses, not optimizer successes. The S1 legacy-positive-trace ablation is an intentional negative. Expect 23 identified rows: 22 exact positives plus that negative. Eight numerical IBC transports are independently replayed without another SDP solve.

The lower-level `run_audit.jl` and `run_reviewer.jl` scripts are diagnostic entry points, not the complete release check. `archive_evidence.jl` is the retained historical archive creator; it refuses an existing archive and is not needed for reproduction. CI no longer writes new evidence into the frozen archive.

See `../docs/REPOSITORY_FREEZE.md` for the boundary between this completed computational baseline and author review of the paper.
