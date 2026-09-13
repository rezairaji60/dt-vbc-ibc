# Benchmark selection and relation to prior work

## Decision

The evaluation separates source-matched reproduction, a deliberately changed nonlinear problem, and a constructed structural counterexample. Popularity alone does not make a benchmark suitable. The relevant semantics are deterministic autonomous discrete-time polynomial transitions, infinite-horizon safety, and an invariant domain for the comparison inequalities.

The additions are BB_rotation, Logistic_adapted, Rotation2, and Rotation4. The first is source-matched; the second explicitly changes a controlled task; the last two are author-constructed theorem witnesses. None guarantees journal acceptance or establishes a general performance ranking.

## Literature map

Sogokon, Ghorbal, Tan and Platzer develop vector barriers through comparison systems [1]. That foundation precedes this work. Their continuous-dynamics setting is not a source for an unrestricted discrete-time theorem merely because matrices look similar. The new manuscript asks what the globally scaled IBC terminal loop imposes on degree.

Oumer, Murali, Trivedi and Zamani introduce the relevant IBC framework [2]. Primary publication metadata verifies its identity. The supplied manuscript provides the implication and scaled definitions used in this audit. The full original article was not independently accessible, so the coauthors must check exact wording and prior expressiveness discussion against their copy. Full-domain scaled inequalities, implication conditions, and propagation restricted to the complement of the unsafe set are distinct classes.

Anand, Jungers, Zamani and Allgower investigate completeness and ordering of path-complete barrier functions for switched systems [3]. Graph structure is not new by itself. Their mode-labeled graph setting differs from a nonnegative matrix combining certificate values under one autonomous map. Our degree theorem concerns a finite-order autonomous transition and an IBC terminal scalar loop, not general graph ordering.

Peyrl and Parrilo establish numerical-symbolic rational SOS decompositions [4]. Rational reconstruction is not independently novel. This artifact combines explicit Gram matrices, exact residual budgets and transport of an entire proof. Every Gram shift is counted in the residual. A solver status or floating-point eigenvalue estimate is not a proof.

BarrierBench is a 2026 L4DC/PMLR publication with public JSON data [5,6]. It describes 100 systems, including 68 controlled systems, across both time semantics. It is a relevant recent source, not a justified basis for calling it a long-established standard. Selected entries must disclose which dynamics, sets, inputs and supplied solutions are retained.

The logistic family is classical, including May's 1976 paper [7]. Recognition does not imply that a selected safety task is hard or that parameters 3.2 and 2.8 are chaotic. Here a coordinate image bound explains safety; the example is a nonlinear implementation check with an honest scalar baseline.

ARCH-COMP reports established nonlinear continuous and hybrid reachability benchmarks [8]. An Euler-discretized surrogate does not preserve the ODE safety problem without an approximation-error argument. We therefore do not import Van der Pol, Lorenz or hybrid models and silently change semantics. A sampled-data extension needs a justified error model first.

## BB_rotation: source-matched

The selected autonomous JSON entry is:

    x1+ = x1 + 0.01*(-100*x1 - x2)
    x2+ = x2 + 0.01*(x1 - 100*x2)

It simplifies to `(-x2/100,x1/100)`. Initial box: `[0.1,0.4] x [0.1,0.55]`; unsafe box: `[0.45,0.5] x [0.6,1]`. These dynamics and safety sets are unchanged. We add verification domain `[-1.2,1.2]^2`, with image in `[-0.012,0.012]^2`. The dataset's supplied certificate is not imported as a successful solve.

This is a provenance and replay test, not a difficult expressiveness benchmark. States leave the initial box after one step, but that does not make verification difficult. The successful scalar baseline is reported.

## Logistic_adapted: explicit modification

The source entry is controlled. Both inputs are fixed to zero; the retained transition and sets are:

    x1+ = (16/5)*x1*(1-x1)
    x2+ = (14/5)*x2*(1-x2)
    X0 = [0.1,0.3] x [0.2,0.4]
    Xu = [0.8,1] x [0.8,1]
    X  = [0,1]^2

This is not reproduction of the original controller-synthesis task. No source-provided controller/barrier is trusted as evidence. On [0,1], `0 <= r*x*(1-x) <= r/4`, so the image is in `[0,0.8] x [0,0.7]`. It is nonlinear but safety has a simple explanation. All four formulations and a scalar baseline are tested under stated finite synthesis settings.

## Rotation2 and Rotation4: constructed degree obstruction

Rotation2 uses `R(x1,x2)=(-x2,x1)`; Rotation4 is the direct sum. Domain: `[-2,2]^n`. Odd initial coordinates lie in `[0.9,1.1]`, even ones in `[-0.1,0.1]`. The unsafe first coordinate is in `[1.5,1.7]`, all others in `[-0.1,0.1]`.

These are not attributed to an external benchmark collection. The affine facets `(x_i-6/5,-x_i-6/5)` are permuted exactly by the dynamics, with initial margin >=1/10 and unsafe anchor margin >=3/10. A terminal sign-changing scalar function under finite-order dynamics must have unit gain and be invariant. No nonconstant affine invariant exists because R-I is nonsingular. Therefore every finite global scaled IBC with affine frames is excluded analytically, in either direction. A quadratic invariant `x1^2+x2^2-3/2` supplies a scalar/repeated-frame certificate. The minimum degrees are exactly one versus two in the stated classes.

This is not an inference from SDP infeasibility. The four-dimensional block extension tests code generality, not scalability. The impossibility result does not cover unrestricted implication IBCs. A positive uniform propagation reserve is inconsistent with the periodic cyclic comparison; the exact zero-residual witnesses avoid this artificial restriction.

## Reporting and limits

The exact correspondence provides proof reuse; the finite-order theorem supplies a strict template distinction. The numerical suite tests implementation, diagnosis, provenance and scalar baselines. A larger table of easy problems is not the principal contribution.

All outcomes, including the failed legacy-normalization ablation, remain in the summary. An unsuccessful finite SDP is not certificate nonexistence. Optimizer-only timings do not support end-to-end speed rankings. Every accepted candidate requires exact-rational checking and an invariant domain. The committed environment and proof files, with source revision and hashes, avoid dependence on expiring Actions artifacts.

## Sources

1. A. Sogokon, K. Ghorbal, Y. K. Tan, A. Platzer, Vector Barrier Certificates and Comparison Systems, FM 2018, LNCS10951,418-437. DOI10.1007/978-3-319-95582-7_25. https://logic.kastel.kit.edu/pub/vector-barrier.pdf
2. M. A. Oumer, V. Murali, A. Trivedi, M. Zamani, Safety Verification of Discrete-Time Systems via Interpolation-Inspired Barrier Certificates, IEEE CSL8(2024),3183-3188. DOI10.1109/LCSYS.2024.3521356. Primary metadata https://experts.colorado.edu/display/pubid_377068 . Full original article not independently retrieved; supplied manuscript definitions audited.
3. M. Anand, R. Jungers, M. Zamani, F. Allgower, On the Completeness and Ordering of Path-Complete Barrier Functions, arXiv2503.19561(2025). https://arxiv.org/html/2503.19561v1
4. H. Peyrl, P. A. Parrilo, Computing sum of squares decompositions with rational coefficients, TCS409(2)(2008),269-281. DOI10.1016/j.tcs.2008.09.025. https://old.control.ee.ethz.ch/publications/2008/3087.html
5. A. Taheri, A. Taban, S. Soudjani, A. Trivedi, BarrierBench: Evaluating Large Language Models for Safety Verification in Dynamical Systems, PMLR331(2026),640-661. https://proceedings.mlr.press/v331/taheri26a.html
6. HyCoDeV, BarrierBench JSON, accessed2026-09-13. https://hycodev.com/data/BarrierBench.json . Equations and sets above identify entries; solution strings are not trusted proofs.
7. R. M. May, Simple mathematical models with very complicated dynamics, Nature261(1976),459-467. DOI10.1038/261459a0.
8. L. Geretti et al., ARCH-COMP25 Category Report: Continuous and Hybrid Systems with Nonlinear Dynamics, EPiC108(2025),39-70. DOI10.29007/7br2. https://easychair.org/publications/paper/m9FM

The SCL author-guide page returned an access error. The manuscript uses installed Elsevier elsarticle formatting. Current portal requirements and author declarations require confirmation. This is not an exhaustive novelty search or an acceptance guarantee.
