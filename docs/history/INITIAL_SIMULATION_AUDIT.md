# Simulation-first audit for the Systems & Control Letters revision

## Scope and source binding

Target repository: AC-Disaster-Consulting/Test (private).
Audited legacy main: b2fbd9be3a8b5de0212dc86b88965f90d0c32257.
Uploaded LaTeX SHA256: 9941013aceded4efb0e5a61df02abfb7cece3d3192428fe00fac02573a0d074e.
The newly uploaded and earlier LaTeX texts have the same SHA256.
No claim is made that the manuscript is submission-ready or that publication is guaranteed.
The paper is not rewritten in this change: resolve the existing examples first, as requested by Majid and Vishnu.

## What is established independently of an optimizer

S1 admits quadratic certificates in ALL FOUR formulations on the original
verification domain [-1.4,1.4]^2, with the original initial and unsafe boxes.
This does not contradict the narrow statement that the previous restricted search
failed. It does invalidate using that failure as evidence of intrinsic backward
quadratic infeasibility or of a forward-only example.

Let V=x1^2+x2^2 and c=1/10. On the original domain,

    V(f(x)) <= (82/125) V(x).

The code contains an explicit rational SOS identity for this inequality, not a
sampling test. Consequently g=V-c and h=c-V satisfy

    (4/5)g - g(f) >= (18/125)V + 1/50,
    (6/5)h(f) - h >= (133/625)V + 1/50.

Both are strict throughout the domain. On the initial box, V<=2/25, so g<=-1/50
and h>=1/50. On the unsafe box, V>=361/200, so g>=341/200 and h<=-341/200.

Repeat g or h as needed to produce equal-size vectors/frames:

| Formulation | Functions | Comparison/scales |
|---|---|---|
| Forward VBC | g,g,g | path entries 4/5 (also (4/5)I works) |
| Backward VBC | h,h,h | path entries 6/5 (also (6/5)I works) |
| Forward IBC | g,g,g | all lambda=6/5 |
| Backward IBC | h,h,h | all lambda=5/4 |

The backward IBC propagation residual is (5/4) times the forward VBC residual.
The forward IBC residual equals the backward VBC residual. Also, h repeated with
all backward IBC lambda=6/5 works, since its residual is the second inequality
above; this scale was included in the old S1 search. The backward VBC matrix
(6/5)I was included there too. Hence merely widening the old parameter scan is
not an adequate explanation or repair: the old normalization excluded h.

These repeated scalar functions diagnose feasibility. They are NOT evidence that
vector or frame certificates outperform a scalar barrier.

## Exact SOS decomposition underlying the S1 result

Set x=x1, y=x2, g1=49/25-x^2, g2=49/25-y^2, and

    a0=303/625, d0=327/625,
    t=3/25, u=2/25, b=1/10, c2=2/25,
    r=3/50, alpha=41/50, delta=1/50.

The polynomial H=(82/125)(x^2+y^2)-f1^2-f2^2 is exactly

    H = s0 + g1*s1 + g2*s2,

where

    s0 = b*a0*(x-y)^2 + t*a0*x^4 + t*b*x^2*y^2
       + c2*d0*(x+y)^2 + r*c2*x^2 + r*d0*y^2
       + u*c2*x^2*y^2 + u*d0*y^4
       + alpha*t*x^4 + alpha*delta*y^2 + alpha*u*y^4,
    s1 = b*t*(x-y)^2 + t^2*x^4,
    s2 = c2*u*(x+y)^2 + u*r*y^2 + u^2*y^4.

All displayed weights are positive rationals; all summands are weighted squares.
The implementation checks the expanded identity exactly, adds explicit SOS
separation proofs on the initial/unsafe boxes, and replays all four certificates
through the same independent checker used for numerical candidates.

## S2: preserve the original failure and distinguish the repaired domain

Original domain: [-7/5,7/5] x [-6/5,6/5].
At (7/5,6/5), f1=3529/2500=1.4116 > 7/5. Thus f(X) subset X is false.

Repaired domain: [-3/2,3/2] x [-6/5,6/5].
No dynamics, initial set or unsafe set is changed. On this box,

    df1/dx1 = 21/20-(3/10)x1^2 >= 3/8 > 0,
    |f1| <= 2907/2000 = 1.4535 < 1.5,
    |f2| <= 567/500 = 1.134 < 1.2.

Coordinatewise monotonicity and odd symmetry give an exact invariant-box proof.
The original domain is retained as S2_original and is explicitly refused an
unqualified safety status. The changed verification domain is S2_repaired.

The S2 initial box is NOT invariant: f1(3/25,3/25)=46071/312500=0.1474272>0.12.
Do not repeat the blanket claim that trajectories stay in the initial box for
both examples. That statement is correct for S1, not S2.

## Theory defects that the new implementation does not hide

1. bIBC-to-forward-VBC uses 1/lambda, including the terminal diagonal entry.
2. A sign flip of b0<=0 supplies only B0>=0. Strict unsafe separation must be
   justified, not asserted. The new code explicitly uses robust margins.
3. Forward IBC propagation in the uploaded manuscript is on X minus Xu, whereas
   the target backward VBC is required on all X. A direct all-X correspondence
   needs an all-X hypothesis. The code labels this conservative strengthening.
4. The displayed IBC SOS polynomial has the wrong sign: certify
   b_i-lambda_i*b_{i+1}(f), not its negative.
5. Fixed scales/matrices give convex SDPs. Joint unknown A and B is bilinear.
6. A matrix support-pattern statement is not proof of strict expressive
   superiority at a fixed polynomial degree. The earlier proposed graph-based
   plan must not be presented as an established theorem or novel contribution.
7. Positive diagonal scaling preserves feasibility but not an arbitrarily fixed
   coefficient normalization, margin objective or numerical conditioning.

## Reviewer traceability

| Concern | Code response | Remaining paper task |
|---|---|---|
| Reviewer 5: non-strict unsafe sign | explicit robust separation; exact replay | correct and qualify theorem |
| Reviewer 5: missing multipliers/bilinearity | explicit PSD Gram identities; fixed numeric A/lambda | write full SOS constraints |
| Reviewer 7: mismatched theoretical comparison | same component counts, mapped matrices, sign-symmetric norm; proof transport without resynthesis | replace misleading comparisons |
| Reviewer 7: trajectories confined in X0 | exact S1 invariance and exact S2 counterexample | accurate figure interpretation |
| Reviewer 3/editor: novelty and usefulness | no invented novelty claim; reusable proof transport and audit | literature audit and substantive new result |
| All: weak numerical evidence | exact rational PSD checks and error bounds; no automatic optimal_inaccurate acceptance | new verified experimental tables |

## Interpretation of numerical results

The first corrected experiment is a feasibility audit, NOT a best-margin
competition. It uses degree-two polynomials, order-three SOS identities, fixed
separation 1/1000, full-X propagation, a 1/1000000 SOS reserve, and an l1 coefficient
bound of one per component. The positive reserve is an explicit strengthening
used to absorb roundoff during exact replay. It can exclude zero-slack witnesses.
Failure under this profile is not failure of the manuscript's weaker conditions.

The legacy-positive-trace ablation changes only the normalization of one corrected
S1 backward run; it does not pretend to reproduce every old collocation detail.
Original Python files and historical numbers are retained unmodified.

Exact replay rationalizes coefficients and Gram entries, checks every PSD matrix
using rational arithmetic, recomputes the target polynomials from the benchmark,
and bounds the full coefficient residual on the box. If

    p = reserve + sum(g_j*z_j'Q_j*z_j) + e,
    |e| <= beta <= reserve,

then p>=0 everywhere on that box. A Gram diagonal shift is fully charged to e.
Dense sampling, floating-point eigenvalues, and solver statuses are not proofs.
The checker is a small auditable arithmetic program, not a proof-assistant kernel.

The main branch and manuscript must not be updated with new performance claims
until the run artifacts are inspected and independent replay succeeds.
