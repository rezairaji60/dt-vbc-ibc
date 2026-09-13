# Computational freeze for the duality/complementarity manuscript

## Scope

This baseline closes the implementation work needed for the manuscript with fixed paper title:

**Duality and Complementarity of Vector and Interpolation-Inspired Barrier Certificates for Safety Verification: Toward Reduced Conservatism and Complexity**.

It preserves the previously verified Julia SOS implementation and immutable 36-proof archive, and adds a separate exact analytical complementarity layer. The new layer does not rewrite earlier solver outcomes or infer nonexistence from SDP failure.

## Scientific scope frozen here

1. Robust anchored full-domain globally scaled IBC/path-VBC correspondence and proof transport.
2. Rotation2/Rotation4: exact cyclic affine VBC witnesses, orbit-average exclusion of affine implication IBCs, and invariant quadratic IBC witnesses establishing minimum degrees 1 versus 2 in the stated classes.
3. ImplicationGap1D: exact affine implication-IBC proof, exact convex-combination hypotheses excluding affine constant-comparison VBCs for any finite component count on the fixed domain, and an exact quartic scalar global-VBC recovery witness.
4. Transparent combinatorial coefficient/Gram sizing to explain why avoiding unnecessary degree escalation can reduce optimization size, without claiming a universal runtime ranking.

No universal dominance claim is made. Exact conversion preserves degree. Complementarity appears only outside the common globally scaled path subclass.

## Release requirements

`julia_sos/run_all.jl` is the authoritative command and requires Julia 1.10.10. Success is the final `REVIEWER_RELEASE_VERIFIED` marker. The run checks frozen source/evidence fingerprints, scientific tests, portability tests, the original proof archive, the fixed numerical table, the deterministic complementarity report, the separately replayed affine implication-IBC proof bundle, and every fresh globally scaled/VBC SOS proof.

The numerical table remains exactly 23 rows: 16 multi-function positives, four scalar positives, two analytical cyclic VBC positives, and one intentional normalization negative. Complementarity/nonexistence evidence is deliberately **not** encoded as solver-result rows; it is checked separately.

The old `evidence/reviewer/` archive remains byte-for-byte immutable. The added `evidence/complementarity/` report has its own release fingerprint. Each current reviewer run also emits a standalone exact implication proof bundle and replays it independently. New computations use isolated output directories.

## Evidence provenance

The original SOS archive was generated before this complementarity extension and remains the source of the numerical proof bundles. The new analytical evidence is reconstructed by the current source and compared against a committed deterministic report. A final release is valid only after CI passes on the exact merged revision on Windows and Linux.

## Paper boundary

After this freeze, code changes should not be needed merely to rewrite the manuscript around duality, complementary expressiveness, reduced conservatism, and degree/complexity consequences. The paper must still undergo independent author review of theorem statements, novelty positioning, citations, assumptions, and journal declarations.

Changing a benchmark, comparison class, implication semantics, verifier, solver settings, or a computational claim is new scientific scope and requires new evidence. Peer review may of course request such work; this freeze does not guarantee acceptance.
