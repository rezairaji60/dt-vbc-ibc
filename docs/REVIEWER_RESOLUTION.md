# Reviewer concern to evidence

| Concern | Revision | Evidence and limit |
|---|---|---|
| Reviewer 5: unclear goal / benefit | Reframe the paper around **structural duality plus complementary expressiveness**: change formulation before automatically increasing degree. | `COMPLEMENTARITY.md`; exact common-subclass transport plus opposite fixed-degree separations. No universal dominance claim. |
| Reviewer 5: bilinear scales and unspecified multipliers | Numerical scaled problems fix comparison/scaling parameters per solve and use explicit PSD Gram matrices/coefficient identities. | Genuine Julia SOS, not collocation. ImplicationGap1D is analytical; it is not mislabeled as a convex free-synthesis result. |
| Reviewer 5: strict positivity gap | Uniform positive separation margins in the exact representation theorem and computational obligations. | Explicitly strengthened class, not silently equated with weak separation. |
| Reviewer 7: theory/numerics mismatch | Same number of functions, mapped matrices and symmetric normalization in the exact transport tests. | Complete IBC SOS proofs transport and replay as VBC proofs without another SDP solve. |
| Reviewer 7: trajectories stay inside initial set | S1 is a sanity check; S2 and added cases leave their initial boxes. | Leaving a box is not itself proof of difficulty. |
| Reviewer 3: missing converse | Both directions are stated for the exact robust anchored path-matrix subclass. | No arbitrary-matrix equivalence. |
| Reviewer 3: no strictness / relative expressiveness | Two complementary analytical separations. Rotation examples: affine cyclic VBC exists but no affine implication IBC; quadratic IBC exists. ImplicationGap1D: affine implication IBC exists but every affine constant-comparison VBC is excluded; quartic global VBC recovery is verified. | Nonexistence comes from theorem arguments, not failed solvers. ImplicationGap1D does not exclude state-dependent comparison maps or changed domains. |
| Editor: modest conceptual/practical contribution | The relationship now answers a formulation-selection question: distinguish template-degree limitations from propagation-structure conservatism before escalating degree. | Complexity benefit is supported by transparent monomial/Gram sizing, not a claimed universal runtime speedup. |
| Omitted proof | Complete proofs must appear in the revised manuscript. | Repository checks concrete hypotheses/witnesses; theorems still require independent author review. |
| Historical SOS claim | Old active solver is labeled collocation; historical files retained. | Current numerical tables come from genuine SOS and exact replay. |
| Invalid S2 domain | Explicit enlarged invariant domain; original retained and rejected. | Dynamics and safety sets unchanged. |
| Reproducibility | Single Julia 1.10.10 command, pinned environment, immutable SOS archive, separate complementarity evidence. | Numerical candidates accepted only by exact checking; analytical nonexistence is kept outside solver-result rows. |

The fixed paper title is **Duality and Complementarity of Vector and Interpolation-Inspired Barrier Certificates for Safety Verification: Toward Reduced Conservatism and Complexity**.

This matrix records technical responses to the prior reviews. It is not an acceptance prediction and does not substitute for Majid/Vishnu's theorem and novelty review.
