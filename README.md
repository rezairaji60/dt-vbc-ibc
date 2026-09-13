# Vector and interpolation-inspired barrier certificates

Computational baseline for Majid and Vishnu's manuscript review. The proposed title is **Structural Relationships Between Vector and Interpolation-Inspired Barrier Certificates for Safety Verification**. Title and manuscript approval are author decisions; the computational freeze does not silently rewrite the paper.

## Reproduce the frozen results

Use Julia **1.10.10** and Git. From the repository root, run:

```sh
julia --startup-file=no --project=julia_sos julia_sos/run_all.jl
```

The `julia` executable in that command must report version 1.10.10. With a standalone Windows installation, use its executable directly:

```powershell
& "C:\path\to\julia-1.10.10\bin\julia.exe" --startup-file=no --project=julia_sos julia_sos/run_all.jl
```

The command checks frozen inputs, runs the scientific and release tests, replays the committed archive, synthesizes all fixed cases, and independently replays every new proof. It succeeds only after printing `REVIEWER_RELEASE_VERIFIED`. A new run gets its own `julia_sos/results/run-*` directory; `LATEST_RUN.json` identifies the most recent completed run. The immutable archive is never overwritten. CSDP is the default SDP solver; no commercial solver license or repository secret is required.

To check the committed proofs without another SDP solve:

```sh
julia --startup-file=no --project=julia_sos julia_sos/experiments/replay_all.jl
```

Success is `ARCHIVE_REPLAY_VERIFIED: 36 proof bundles; all indexed SHA256 digests checked.` An older Windows checkout with changed line endings may need the one-time, backup-preserving repair in [Windows archive replay](docs/WINDOWS_ARCHIVE_REPLAY.md). Do not update dependencies or regenerate expected hashes to suppress an error.

## Results and limits

The fixed table contains 16 degree-two multi-function cases, four degree-two scalar baselines, two exact analytical affine rotation witnesses, and one intentionally unsuccessful legacy-normalization ablation. Each positive certificate passes rational Gram-matrix and full-box residual verification with an established invariant domain. S2 uses the explicitly repaired domain. The logistic task is adapted, not a reproduction of a control-synthesis problem.

The representation theorem concerns robust anchored full-domain globally scaled IBCs. The separate finite-order degree argument concerns implication-style IBCs. The programs check concrete witnesses and regression identities, not the general theorem in a proof assistant. The four numerical problems also admit scalar certificates; they do not establish general vector superiority, optimized margins, or runtime/scalability rankings.

## Reviewer navigation

- [Freeze scope and paper handoff](docs/REPOSITORY_FREEZE.md)
- [Execution evidence](docs/EXECUTION_EVIDENCE.md)
- [Julia commands, outputs and interpretation](julia_sos/README.md)
- [Benchmark provenance and prior-work limits](docs/BENCHMARK_PROVENANCE.md)
- [Reviewer concern-to-evidence map](docs/REVIEWER_RESOLUTION.md)
- [Manuscript compilation](paper/README.md) and [author checks](paper/AUTHOR_CHECKLIST.md)

The original Python code and results are historical collocation, not current SOS evidence. Historical diagnostics are under `docs/history/`. No legacy values are substituted into the new table.

## Export for a reviewer without repository access

```sh
git archive --format=zip --output=../SCL_reviewer_source.zip HEAD README.md .gitattributes SOURCE_REVISION.txt docs julia_sos evidence paper
```

Extract the ZIP and run the same command from its root. Git is used for isolated portability fixtures, not to access an account. `SOURCE_REVISION.txt` records the exported commit. The ZIP contains source, pinned environment, proofs and documentation; a results-only Actions artifact does not replace it. CI checks an exported copy as well as the Git checkout.

The repository remains private. Send the source ZIP or arrange authorized access for external reviewers. This freeze is for coauthor review, not a journal submission, licensing grant, independent novelty approval, or acceptance guarantee.
