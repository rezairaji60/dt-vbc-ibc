# Author checks before Majid/Vishnu review and journal submission

## Scientific checks

- Independently verify the exact scaled/path duality statements, reciprocal factors, strict margins and propagation domains.
- Independently check the finite-order orbit-average theorem: its hypotheses, forward/backward implication-IBC scope, and the claim that the quadratic witness makes the rotation degree bound tight (1 for VBC versus 2 for IBC).
- Independently check the affine global-VBC convex-combination obstruction for ImplicationGap1D and its assumptions: fixed domain, affine components, finite dimension and constant nonnegative comparison matrix.
- Check the affine implication-IBC identity and quartic global-VBC recovery witness. Do not state that degree four is minimal; the established global-VBC bound is `2 <= d_min <= 4`.
- Keep exact conversion and complementary structure distinct: exact conversion preserves degree; degree reduction occurs by changing admissible structure outside the common subclass.

## Novelty/positioning checks

- Compare the complete original IBC article (Oumer, Murali, Trivedi, Zamani, IEEE Control Systems Letters 2024), not only metadata/abstracts.
- Compare Sogokon et al. on vector comparison systems and the path-complete barrier literature, including ordering/completeness results. Graph structure and the general idea that weaker conditions reduce conservatism are prior work.
- Position the contribution as the precise common-subclass correspondence plus opposite fixed-degree obstructions/complementary expressiveness, subject to the stated assumptions.
- Avoid “state of the art,” universal dominance, universal speedup, or generic scalability claims unless separately demonstrated.

## Complexity/reporting checks

- Explain why lower degree can reduce monomial/PSD sizes, but report component count, multiplier order and actual solver dimensions where quantitative complexity claims are made.
- Treat repository runtime fields as `optimize!` timing only, not end-to-end performance rankings.
- Keep solver failures (`NO_CERTIFIED_CANDIDATE`) separate from theorem-based nonexistence.

## Submission metadata

- Confirm author order, affiliations, corresponding author, NSF acknowledgments, competing interests and no-concurrent-submission declaration.
- Review any required AI-use disclosure and all final text/code/proofs personally.
- Provide a complete source/evidence artifact or authorized repository access to reviewers as appropriate.
- Confirm current Systems & Control Letters portal/template requirements immediately before submission.
