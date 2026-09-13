# Execution evidence for the reviewer baseline

The current release command and requirements are in [REPOSITORY_FREEZE.md](REPOSITORY_FREEZE.md). The first simulation-only checkpoint is preserved in [history/INITIAL_SOS_CHECKPOINT.md](history/INITIAL_SOS_CHECKPOINT.md); its early test count and artifact-retention period are historical, not current instructions.

## Immutable scientific archive

The original verified archive is committed under `evidence/reviewer/`, with 69 indexed files and 36 proof bundles. Its index SHA-256 is `bba014707d2252ec30bc24863204cc88fcd2ee5e96acec0d8c9ba24436224a29`. It was generated from source `fd820218a04ed0621bbc8bd5a20d51b53a6cdae4`, run `34746809943`. The archive includes the resolved environment, original logs and source identity. New runs do not replace these files or their checksums.

## Fresh verification

The complete reviewer command checks frozen inputs, executes the tests, replays the archive, runs all fixed synthesis cases, and independently replays all freshly generated bundles. Its final marker is `REVIEWER_RELEASE_VERIFIED`. Every new run has a separate directory and current source/runtime metadata. The fixed numerical table has 16 multi-function cases and four scalar baselines, all exact-positive; two analytical structural witnesses are separate; the legacy normalization ablation remains an intentional negative.

The archived scientific tests contain 85 passing assertions. The current command additionally executes checkout/repair and release-contract regression tests. Do not confuse repeated tests on two operating systems with additional mathematical results.

## Cross-platform evidence

The Windows byte-preservation repair passed Linux and Windows in runs `34760382582` and `34760662764`. The full reviewer workflow also passed on the merged repair revision in run `34760662758`. The present release's CI checks the full entry point on both systems and an exported source copy. Read the current commit's Actions runs for its exact execution status; earlier successes do not certify later changes automatically.

## Interpretation

`EXACT_RATIONAL_VERIFIED` denotes verification by the supplied exact-arithmetic checker. It is not proof-assistant mechanization. `NO_CERTIFIED_CANDIDATE` is not an exact infeasibility theorem. Runtime values cover optimize! only and may include first-call compilation; they do not establish end-to-end rankings. Historical collocation margins are not current evidence. Reviewer access to a complete source/evidence package must accompany an external submission because the development repository remains private.
