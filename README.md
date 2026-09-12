# DT-VBC / IBC: simulation-first Julia SOS audit

The current reconstruction for **Systems & Control Letters** is in [`julia_sos/`](julia_sos/README.md).
It uses genuine polynomial coefficient identities and PSD Gram matrices, followed
by independent exact-rational verification. Start with the
[scientific audit](docs/SCL_SIMULATION_AUDIT.md) and
[recorded execution evidence](docs/EXECUTION_EVIDENCE.md).

**Historical implementation warning:** `src/dt_vbc/synthesis_sos.py` is sampled
collocation, despite the old filename and README calling it SOS. Its historical
outputs are not formal SOS certificates and must not be reused as the revised
paper's verified results. Python source and old result files are retained intact.
The baseline is `b2fbd9be3a8b5de0212dc86b88965f90d0c32257`.

The Julia audit establishes quadratic feasibility of all four formulations for
S1 and for S2 with an explicitly enlarged invariant verification box. These are
feasibility and correctness results, not a new best-margin competition, an
expressiveness theorem, or a declaration that the paper is ready for submission.

The implementation branch is `fix/scl-julia-sos-audit`, reviewed in draft PR #1.
