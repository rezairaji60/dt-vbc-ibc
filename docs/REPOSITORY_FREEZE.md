# Computational freeze for coauthor review

## Scope

This baseline supports moving from implementation repair to Majid and Vishnu's manuscript review. It freezes the existing certificate definitions, benchmark equations and sets, fixed solver settings, and exact proof archive. It does not introduce another benchmark, alter a polynomial, or revise the general mathematical claims.

The proposed conservative title is **Structural Relationships Between Vector and Interpolation-Inspired Barrier Certificates for Safety Verification**. The manuscript's title, theorem presentation and novelty positioning remain part of paper review; a new title is not evidence of a new numerical method.

## Release requirements

`julia_sos/run_all.jl` is the authoritative execution command with Julia 1.10.10. It must pass all scientific, portability and release-contract tests; replay the immutable archive; regenerate the fixed reviewer table; and independently replay all current proof bundles. Success is `REVIEWER_RELEASE_VERIFIED`, not merely process startup or a solver's OPTIMAL status.

`REVIEWER_RELEASE.toml` binds nine scientific/environment files by canonical Git blob fingerprint and the archive index by SHA-256. Archive bytes are checked without line-ending normalization. There are 69 indexed files and 36 mathematical proof bundles. Numerical outputs are allowed to differ across platforms, but every accepted result must satisfy the same exact verifier and identified result contract.

The fixed table has exactly 23 distinct rows: 16 multi-function numerical successes; four scalar numerical successes; two analytical cyclic affine witnesses; and one deliberately unsuccessful normalization ablation. Missing, duplicated, relabeled, unverified or degree-mismatched rows fail the release. All eight numerical IBC-to-VBC transports are required among the fresh proof bundles.

Fresh computations have isolated result directories. A failed attempt cannot reuse an old certificate as its evidence. The committed archive and pinned environment are checked again at the end; no repair, regeneration or dependency update is performed automatically.

## Sources of evidence

The original complete proof archive was generated from `fd820218a04ed0621bbc8bd5a20d51b53a6cdae4` in Actions run `34746809943`, archived by `2cd33a9a30f7c0aca24c62e4289b13e940659787`, and merged in `8ae73c6a0e38b5b7868c2d1ba5d16e90cc3c0fd4`. The Windows byte-preservation repair was merged as `c1d5c52f2061598859278439d91cf148dff76929`. This release hardens execution/documentation around the same scientific inputs; these historical IDs are not represented as the current checkout SHA.

The release's current commit is obtained using `git rev-parse HEAD`, or `SOURCE_REVISION.txt` in a `git archive` export. CI is run on the exact head and on merged main. Both Windows and Linux run the same full reviewer entry point. The exported source package is also checked without a repository checkout. The final PR and Actions runs record execution against the actual release SHA; this document does not claim that an untested future commit inherits a pass.

## Reviewer documentation

Start with `../README.md` and `../julia_sos/README.md`. `EXECUTION_EVIDENCE.md` explains the evidence layers. `BENCHMARK_PROVENANCE.md` identifies unmodified, adapted and constructed cases. `REVIEWER_RESOLUTION.md` maps the original concerns to code or manuscript sections. `WINDOWS_ARCHIVE_REPLAY.md` documents only the old-checkout repair. Initial-stage reports are preserved under `history/` and are not current run instructions.

The repository stays private. Export and provide the complete source/evidence ZIP to reviewers, or arrange authorized access. No account credential is needed to execute an exported artifact after ordinary dependency installation. No license or repository visibility change is made by this freeze.

## What is complete, and what belongs to the paper

The engineering scope is complete only when the release CI and archive checks pass. That permits freezing code while working on prose, title, proofs, references and author metadata. The experimental scope is a reproducibility/diagnostic study plus exact structural examples; it is not a large-scale performance comparison.

Author review must still assess the new general theorem, the full original IBC article, related graph-certificate work, assumptions, and journal requirements. The four numerical examples also have successful scalar baselines; the logistic task is adapted; Rotation4 is a block extension; no optimized-margin or speed ranking is established. Those limitations are intentional and should remain explicit in the paper.

No additional implementation change is planned for this fixed scope. A new benchmark, altered model, changed verifier, different polynomial template or new computational claim would be a new scope and would require new evidence. Passing this freeze is not a guarantee that peer review can never request such a change, nor a claim of journal acceptance or independent coauthor approval.
