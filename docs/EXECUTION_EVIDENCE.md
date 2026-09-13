# Execution evidence for the duality/complementarity baseline

The current release command and requirements are in [REPOSITORY_FREEZE.md](REPOSITORY_FREEZE.md). The first simulation-only checkpoint is preserved in [history/INITIAL_SOS_CHECKPOINT.md](history/INITIAL_SOS_CHECKPOINT.md); its early test count and artifact-retention period are historical, not current instructions.

## Immutable numerical SOS archive

The original verified archive is committed under `evidence/reviewer/`, with 69 indexed files and 36 globally scaled/VBC proof bundles. Its index SHA-256 is `bba014707d2252ec30bc24863204cc88fcd2ee5e96acec0d8c9ba24436224a29`. It was generated before the complementarity extension and remains byte-for-byte immutable. New runs do not replace these files or their checksums.

## Complementarity evidence

`evidence/complementarity/complementarity_report.json` records deterministic analytical data for the two fixed-degree separation arguments and the transparent complexity counts. The release contract fingerprints its exact bytes and reconstructs its identities/hypotheses from source.

Each fresh reviewer run additionally creates `ImplicationGap1D_forward_implication_ibc_analytical.json`. This is a distinct exact implication-IBC proof schema. The replay reconstructs the fixed affine frame, robust separation margins, invariant domain, box/antecedent generators, and the displayed conditional SOS decomposition in exact rational polynomial arithmetic. This is fixed-certificate implication verification; it is deliberately not represented as a convex free-synthesis result.

## Fresh verification

The complete reviewer command checks source/evidence fingerprints, executes the tests, replays the immutable numerical archive, runs the fixed synthesis table, reconstructs the complementarity report, replays the implication bundle, and independently replays all fresh globally scaled/VBC proof bundles. Its final marker is `REVIEWER_RELEASE_VERIFIED`. Every run has a separate output directory and current source/runtime metadata.

The numerical table remains exactly 23 rows: 16 multi-function exact positives, four scalar exact positives, two analytical cyclic VBC positives, and one intentional normalization negative. The complementarity report and implication proof bundle are checked separately so theorem-based nonexistence is never encoded as solver infeasibility.

## Cross-platform evidence

CI executes the complete reviewer entry point on both Windows and Linux and checks an exported source copy without a source Git checkout. Exact evidence bytes under both `evidence/reviewer/` and `evidence/complementarity/` are marked non-text in `.gitattributes`, preventing Windows line-ending conversion from silently changing fingerprints.

## Interpretation

`EXACT_RATIONAL_VERIFIED` denotes verification by the supplied exact-arithmetic checker, not proof-assistant mechanization. `NO_CERTIFIED_CANDIDATE` is not an exact infeasibility theorem. Runtime values cover `optimize!` only and may include first-call compilation; they do not establish end-to-end rankings. The complexity profiles are combinatorial sizing formulas, not runtime or scalability theorems.

The universal orbit-average and convex-combination obstruction arguments remain mathematical theorems for independent author review. The code verifies their concrete hypotheses and positive witnesses. Reviewer access to a complete source/evidence package should accompany an external submission because the development repository remains private.
