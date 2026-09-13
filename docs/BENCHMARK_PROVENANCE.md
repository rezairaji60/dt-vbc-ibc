# Benchmark selection and relation to prior work

## Decision

The evaluation separates source-matched reproduction, a deliberately changed nonlinear problem, and constructed structural examples. Popularity alone does not make a benchmark suitable. The relevant semantics are deterministic autonomous discrete-time polynomial transitions, infinite-horizon safety, and an invariant domain for comparison inequalities.

Added cases: BB_rotation, Logistic_adapted, Rotation2, Rotation4. The first is source-matched; the second explicitly changes a controlled task; the last two are theorem witnesses. None guarantees acceptance or establishes broad performance superiority.

## Literature map

Sogokon, Ghorbal, Tan and Platzer develop vector barriers through comparison systems [1]. Their continuous-dynamics foundation precedes this work; analogous matrices alone do not establish a discrete-time theorem.

Oumer, Murali, Trivedi and Zamani introduce the IBC framework [2]. Primary metadata verifies its identity. The supplied manuscript provides the implication and scaled definitions audited here. The full original article was not independently accessible; coauthors must check exact definitions and prior expressiveness discussion. The exact representation result is limited to robust anchored full-domain scaled inequalities. The separate orbit-average obstruction now excludes affine implication IBCs too, including the forward propagation domain X minus Xu.

Anand, Jungers, Zamani and Allgower study completeness and ordering of path-complete barrier functions for switched systems [3]. Graph structure is not itself new. Their mode-labeled setting differs from nonnegative matrix combinations under an autonomous map. The present degree theorem concerns a finite-order autonomous transition and the terminal invariant sublevel set of an IBC, not a general graph-ordering theorem.

Peyrl and Parrilo establish numerical-symbolic rational SOS decompositions [4]. Rational reconstruction is not independently novel. This artifact combines explicit Gram matrices, exact residual budgets, and transport of an entire proof. Every Gram shift is included in the error budget. Solver status and floating-point eigenvalue estimates are not proofs.

BarrierBench is a 2026 L4DC/PMLR publication with public JSON data [5,6], describing 100 systems including 68 controlled ones across both time semantics. It is a relevant recent source, not a justified basis for calling it a long-established standard. Exact retained equations, safety sets, and controller changes are identified below.

The logistic family is classical, including May's 1976 paper [7]. Recognition does not imply that the selected task is difficult or that parameters 3.2 and 2.8 are chaotic. Here a coordinate image bound explains safety; the successful scalar baseline is reported.

ARCH-COMP provides established nonlinear continuous/hybrid reachability benchmarks [8]. Euler discretization does not preserve the original ODE safety question without an approximation-error argument. We therefore do not import Van der Pol, Lorenz or hybrid models and silently change semantics.

## BB_rotation: source-matched

Published autonomous entry:

    x1+ = x1 + 0.01*(-100*x1 - x2)
    x2+ = x2 + 0.01*(x1 - 100*x2)

It simplifies to `(-x2/100,x1/100)`. X0=`[0.1,0.4] x [0.1,0.55]`; Xu=`[0.45,0.5] x [0.6,1]`. Transition and safety sets are unchanged. We add invariant verification domain `[-1.2,1.2]^2`, image bounded by +/-0.012. The source's supplied certificate is not imported as a successful solve. This is a provenance/replay test, not a difficult expressiveness benchmark.

## Logistic_adapted: explicit modification

The source entry is controlled. Fix both inputs to zero:

    x1+ = (16/5)*x1*(1-x1)
    x2+ = (14/5)*x2*(1-x2)
    X0 = [0.1,0.3] x [0.2,0.4]
    Xu = [0.8,1] x [0.8,1]
    X  = [0,1]^2

This does not reproduce the original controller-synthesis task. No supplied controller/barrier is trusted. Since `0 <= r*x*(1-x) <= r/4`, the image lies in `[0,0.8] x [0,0.7]`. All four degree-two formulations and a scalar baseline are tested under explicit fixed settings.

## Rotation2 and Rotation4: constructed tight degree separation

Use `R(x1,x2)=(-x2,x1)` and its direct sum. X=`[-2,2]^n`; odd initial coordinates `[0.9,1.1]`, even ones `[-0.1,0.1]`; unsafe first coordinate `[1.5,1.7]`, all others `[-0.1,0.1]`.

The affine facets `(x_i-6/5,-x_i-6/5)` are permuted by R, with initial margin >=1/10 and unsafe anchor margin >=3/10. Finite order makes any nonempty terminal IBC sublevel set and its complement invariant. Because R has no eigenvalue one, every orbit has state average zero. An affine terminal frame has the same orbit average everywhere, so it cannot be nonpositive on one orbit and strictly positive on another. This excludes both forward and backward affine implication IBCs of every finite length, and hence globally scaled ones. The complete proof is in the manuscript and `IMPLICATION_OBSTRUCTION.md`.

The invariant quadratic `x1^2+x2^2-3/2` supplies scalar and repeated-frame certificates. Minimum degrees are therefore exactly one versus two in the stated classes. This is an analytical nonexistence proof, not SDP failure. It does not exclude nonlinear frames. The four-dimensional block extension tests code generality, not scalability. Positive propagation reserve is inconsistent with periodic cyclic comparison; the exact witnesses use zero reserve.

## Reporting and limits

The exact global-scaled correspondence enables proof reuse; the stronger implication-class theorem supplies a genuine degree distinction. Numerical cases test implementation, diagnosis and provenance with honest scalar baselines. All outcomes, including the failed normalization ablation, remain in the summary. Optimizer-only timings do not establish end-to-end rankings. Accepted proofs require exact arithmetic and invariant domains. The committed environment and proof archive avoid dependence on expiring Actions artifacts.

## Sources

1. A. Sogokon, K. Ghorbal, Y. K. Tan, A. Platzer, Vector Barrier Certificates and Comparison Systems, FM 2018, LNCS10951,418-437. DOI10.1007/978-3-319-95582-7_25. https://logic.kastel.kit.edu/pub/vector-barrier.pdf
2. M. A. Oumer, V. Murali, A. Trivedi, M. Zamani, Safety Verification of Discrete-Time Systems via Interpolation-Inspired Barrier Certificates, IEEE CSL8(2024),3183-3188. DOI10.1109/LCSYS.2024.3521356. Primary metadata https://experts.colorado.edu/display/pubid_377068 . Full article not independently retrieved; supplied manuscript definitions audited.
3. M. Anand, R. Jungers, M. Zamani, F. Allgower, On the Completeness and Ordering of Path-Complete Barrier Functions, arXiv2503.19561(2025). https://arxiv.org/html/2503.19561v1
4. H. Peyrl, P. A. Parrilo, Computing sum of squares decompositions with rational coefficients, TCS409(2)(2008),269-281. DOI10.1016/j.tcs.2008.09.025. https://old.control.ee.ethz.ch/publications/2008/3087.html
5. A. Taheri, A. Taban, S. Soudjani, A. Trivedi, BarrierBench: Evaluating Large Language Models for Safety Verification in Dynamical Systems, PMLR331(2026),640-661. https://proceedings.mlr.press/v331/taheri26a.html
6. HyCoDeV, BarrierBench JSON, accessed2026-09-13. https://hycodev.com/data/BarrierBench.json . The equations and sets above identify the selected entries; solution strings are not trusted proofs.
7. R. M. May, Simple mathematical models with very complicated dynamics, Nature261(1976),459-467. DOI10.1038/261459a0.
8. L. Geretti et al., ARCH-COMP25 Category Report: Continuous and Hybrid Systems with Nonlinear Dynamics, EPiC108(2025),39-70. DOI10.29007/7br2. https://easychair.org/publications/paper/m9FM

The SCL author-guide page returned an access error. Standard Elsevier elsarticle formatting is used. Current portal requirements and personal author declarations require confirmation. This review is not an exhaustive proof of novelty or an acceptance guarantee.
