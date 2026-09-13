# Verified execution checkpoint

First complete successful run: 34720696816 (2026-09-12).
Source branch head: 2a94b08c6a89c07883d3d0e3384d454968ee52b8.
GitHub PR test checkout: 292db55258d9daeb36aa43a1bc72a543fd5a688d.
Base main: b2fbd9be3a8b5de0212dc86b88965f90d0c32257.
Julia: 1.10.10; SDP solver: CSDP.
Resolved Manifest SHA256: eca708ba661f0388f64026f04992f5b17a4d22e3d8c8455f64aac73980b76fa4.

Evidence: https://github.com/AC-Disaster-Consulting/Test/actions/runs/34720696816
PR log: https://github.com/AC-Disaster-Consulting/Test/pull/1#issuecomment-5648891717

## Results actually observed

- 40 tests passed, including a free scalar SOS synthesis plus exact replay.
- S1: all four free, degree-two, three-function SOS searches were OPTIMAL and
  EXACT_RATIONAL_VERIFIED.
- S2_repaired: all four corresponding searches were OPTIMAL and
  EXACT_RATIONAL_VERIFIED.
- All eight accepted runs used separation 1/1000, an explicit SOS reserve of
  1/1000000, order-three identities, and per-component coefficient l1 <= 1.
- All eight accepted rationalized Gram representations were PSD without any
  added diagonal shift (maximum shift 0).
- Both IBC directions on both systems were transported to their corresponding
  VBC proofs without solving a second SDP, and all four replays passed.
- The S1 backward normalization ablation returned INFEASIBLE/INFEASIBLE_POINT
  and did not produce a verified certificate. This is a solver outcome for that
  particular restricted model, not a general theorem of nonexistence.
- Four additional analytical S1 witnesses and their rational SOS identities
  passed exact verification independently of numerical synthesis.

This checkpoint remains tied to the source/test commits above. Subsequent
hardening changes must have their own successful CI run; this file does not
claim that an untested later commit inherits execution status.

## Interpretation limits

The S2 dynamics, initial set and unsafe set were not changed. Only the verification
box was enlarged from [-1.4,1.4] x [-1.2,1.2] to
[-1.5,1.5] x [-1.2,1.2], with an exact invariant-domain proof.
The original box is explicitly recorded as non-invariant.

Raw optimize! timings contain first-call compilation and exclude model building
and exact proof replay. They are not suitable for a performance comparison.
The historical epsilon maxima have not been reproduced or endorsed.

The full certificates, logs and generated Manifest are in the run artifact.
GitHub artifact retention is 14 days; archive those files for any paper release.
