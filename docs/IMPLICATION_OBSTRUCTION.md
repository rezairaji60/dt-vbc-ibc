# Strengthened finite-order result

This supersedes the scaled-only scope used during the first implementation of the rotation examples. The proof is in `paper/main.tex`; exact SOS transport still concerns only the robust anchored global-scaled subclass.

## Statement

Let `f(x)=R*x`, `R^r=I`, `1` not an eigenvalue of R, and `R(X)=X`. For nonempty initial and unsafe sets, no finite forward or backward implication IBC with affine frames exists. The forward propagation may be restricted to `X minus Xu`, as in the supplied manuscript. No interior assumption or uniform negative margin is required for this obstruction.

## Proof

For a forward IBC, the chain from an initial state produces a nonpositive value of the terminal frame p. Each active nonpositive frame excludes the unsafe set, so the next restricted-domain implication is valid. The terminal sublevel set S={x in X:p(x)<=0} is nonempty, disjoint from Xu, and forward invariant. Finite order makes R(S)=S.

For a backward IBC, propagate from an unsafe state through successive preimages (R is bijective). The terminal sublevel set is nonempty, disjoint from X0, and backward invariant; finite order again makes R(S)=S.

Thus one entire orbit is nonpositive and another is strictly positive. But `(I-R)*(I+R+...+R^(r-1))=0` and invertibility of I-R imply the matrix sum is zero. For any affine p(x)=c'x+d, the average of p on every orbit is therefore d. The nonpositive orbit gives d<=0; the positive orbit gives d>0, a contradiction.

The code tests the exact zero orbit-sum identity for the two/four-dimensional rotations and a countercase with eigenvalue one. These are algebraic regression checks, not a mechanized universal proof.

## Tightness and limits

The existing cyclic facet witnesses have degree one and exact zero residual. No constant vector can separate nonempty sets. The invariant quadratic x1^2+x2^2-3/2 separates the chosen initial/unsafe boxes and supplies repeated frames in both directions. Hence the minimum degrees are one for VBCs and two for both implication and scaled IBCs on these examples.

This says nothing about nonexistence of nonlinear frames, arbitrary-function certificates, broad numerical superiority or high-dimensional scalability. Coauthors should independently check the proof and its positioning against the original IBC and path-complete barrier literature before submission.
