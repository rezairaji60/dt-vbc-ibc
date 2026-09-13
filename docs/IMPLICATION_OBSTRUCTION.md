# Finite-order IBC obstruction: the VBC-side complementarity result

This document records one side of the complementary-expressiveness story. See `COMPLEMENTARITY.md` for the reverse implication-IBC advantage.

## Statement

Let `f(x)=R*x`, `R^r=I`, `1` not an eigenvalue of R, and `R(X)=X`. For nonempty initial and unsafe sets as in the paper, no finite forward or backward implication IBC with affine frames exists. The forward propagation may be restricted to `X minus Xu`. No uniform negative propagation margin is required for this obstruction.

## Proof mechanism

A forward IBC chain from an initial state creates a nonempty terminal nonpositive sublevel set, disjoint from the unsafe set and forward invariant. A backward IBC gives the analogous backward-invariant set. Finite order makes the corresponding set invariant over whole orbits.

Because `(I-R)(I+R+...+R^(r-1))=0` and `I-R` is invertible, the orbit-sum matrix is zero. Every affine frame `p(x)=c'x+d` therefore has orbit average `d` on every orbit. One orbit must be nonpositive while another is strictly positive, forcing simultaneously `d<=0` and `d>0`, a contradiction.

The tests verify the exact zero orbit-sum identity for the two- and four-dimensional rotations and a countercase with eigenvalue one. They check theorem hypotheses; they are not a proof-assistant formalization of the universal statement.

## Tightness

Exact cyclic affine VBCs exist. The invariant quadratic `x1^2+x2^2-3/2` supplies an IBC witness on both Rotation2 and Rotation4, with initial margin `7/25` and unsafe margin `3/4`. Hence the minimum degrees are exactly one for VBCs and two for implication/scaled IBCs on these examples.

This result does not imply that VBCs universally dominate IBCs. `ImplicationGap1D` supplies the opposite affine separation under the global constant-comparison VBC assumptions.
