# Structural duality and complementary expressiveness

## Fixed paper title

**Duality and Complementarity of Vector and Interpolation-Inspired Barrier Certificates for Safety Verification: Toward Reduced Conservatism and Complexity**

The title is intentionally two-part. **Duality** denotes the exact bidirectional correspondence proved only for the aligned robust, anchored, full-domain, globally scaled path subclass. **Complementarity** denotes fixed-degree separations outside that common subclass. The word duality is not optimization duality and is not a claim that arbitrary VBCs and arbitrary implication IBCs are interchangeable.

## 1. Common subclass: degree-preserving structural duality

For the globally scaled path subclass, the sign/scaling map converts an IBC to its corresponding VBC and conversely. Polynomial degree and function count are preserved. The Julia implementation transports the complete SOS witness for the tested IBCs without another optimization solve.

This part answers *when the two descriptions encode the same proof*. It does not itself reduce polynomial degree.

## 2. VBC-side advantage: cyclic coupling can lower the required degree

`Rotation2` and `Rotation4` are finite-order autonomous maps. Exact affine facet functions are permuted by the dynamics, giving cyclic degree-one VBCs with zero propagation residual. The orbit-average theorem in `IMPLICATION_OBSTRUCTION.md` excludes every affine forward or backward implication IBC of finite length under the stated assumptions. The invariant quadratic

`x1^2 + x2^2 - 3/2`

separates the selected initial and unsafe boxes and can be repeated as an IBC frame. Hence the minimum degrees on these examples are exactly

- VBC: 1,
- implication/scaled IBC: 2.

The advantage comes from richer cyclic vector coupling, not from sign-flipping a completed path certificate.

## 3. IBC-side advantage: conditional induction can avoid global-comparison conservatism

`ImplicationGap1D` is the author-constructed map

`x+ = x(x+1)/2`, with `X=[-1,1]`, `X0=[-3/5,-2/5]`, `Xu=[1/4,1/3]`.

The affine frame `b(x)=x` is an implication-style forward IBC. Robust separation is immediate. Its propagation implication has the exact identity

`-f = f^2 + (x/2)^2(1-x^2) + ((x+1)^2/2)(-x)`.

On `X`, the box factor `1-x^2` is nonnegative; under the antecedent `b(x)<=0`, `-x` is nonnegative; every multiplier displayed above is a square or a positive multiple of a square. Thus the implication is certified algebraically at degree one. The repository stores the certificate in a separate implication-proof schema and reconstructs the fixed affine frame, robust margins, invariant domain, box/antecedent generators, and displayed SOS decomposition in exact rational polynomial arithmetic. This is fixed-certificate verification; it is not a claim that joint frame/multiplier synthesis is convex.

By contrast, take genuine transitions from `x=-1` and `x=1` with weights `3/4` and `1/4`. Their weighted source is `-1/2 in X0`, while the weighted image is `1/4 in Xu`. Any affine vector B obeys convex combinations, and any constant nonnegative comparison matrix A preserves componentwise nonpositivity. Therefore a global forward comparison `B(f(x)) <= A B(x)` would force `B(1/4)<=0`, contradicting unsafe separation. The same convex-combination argument gives the backward analogue. This excludes **every affine VBC with any finite component count and constant nonnegative A on this fixed domain**; it is not a failed-SDP inference.

A quartic scalar global VBC is also checked:

`p(x)=x^2(x+1)^2-3/40`, `A=1`,

with exact margins `1/80` on the initial set and `29/1280` on the unsafe set and propagation identity

`p(x)-p(f(x)) = f(x)^2(1-f(x))(3+f(x)) >= 0`.

Thus the implication IBC succeeds at degree one, affine global VBCs are impossible in the stated class, and a degree-four global VBC is an explicit recovery witness. We do **not** claim that degree four is minimal; the established bound is `2 <= d_min <= 4` for the global VBC class.

## 4. What “reduced conservatism and complexity” means

The scientific principle is: **before increasing polynomial degree, determine whether failure is caused by the template degree or by the propagation structure.** The two examples show that different structures can exclude different low-degree proofs.

Lower degree can reduce optimization size because a dense n-variable polynomial of degree d has `binomial(n+d,d)` coefficients, while an SOS Gram basis grows combinatorially with its relaxation degree. The repository reports these counts transparently through `complexity_profile`. For six variables with quadratic dynamics, the illustrative dense degree-matched counts are:

| certificate degree | coefficients/component | Gram dimension | symmetric Gram entries |
|---:|---:|---:|---:|
| 2 | 28 | 28 | 406 |
| 4 | 210 | 210 | 22,155 |

These are sizing formulas, not runtime claims. A lower-degree vector certificate can still use more total coefficients than a higher-degree scalar certificate, and multiplier degrees, sparsity, conditioning, and the comparison-structure search also matter.

## 5. Relation to prior work and novelty boundary

Vector comparison systems are established prior work (Sogokon et al., FM 2018). IBCs were introduced by Oumer et al. (IEEE Control Systems Letters, 2024) specifically to find fixed-template multi-function certificates when a standard barrier search fails. Path-complete barrier work studies how graph structure changes conservatism in switched systems. These facts motivate, but do not by themselves establish, the two exact separations above.

The potentially publishable contribution is the precise characterization of the aligned common subclass together with **opposite fixed-degree obstructions** showing complementary expressiveness under clearly stated assumptions. Coauthors must still independently check novelty against the complete IBC article and related graph/certificate-ordering literature before submission.

## 6. Evidence boundary

`evidence/complementarity/complementarity_report.json` is deterministic analytical evidence. The tests reconstruct its identities and theorem hypotheses using rational arithmetic. Each fresh reviewer run additionally writes and independently replays `ImplicationGap1D_forward_implication_ibc_analytical.json`, an exact implication-IBC proof bundle. The universal obstruction arguments remain mathematical theorems to be reviewed by the authors; the code checks their concrete hypotheses and witnesses rather than claiming proof-assistant mechanization.
