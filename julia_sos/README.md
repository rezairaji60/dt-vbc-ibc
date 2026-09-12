# Julia SOS current-case audit

This is the simulation-first repair for the rejected DT-VBC/IBC manuscript.
It is isolated from the historical Python collocation implementation.

## Run

Use Julia 1.10.10 (the CI version). From the repository root:

```sh
git fetch origin
git switch fix/scl-julia-sos-audit
julia --project=julia_sos -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=julia_sos julia_sos/test/runtests.jl
julia --project=julia_sos julia_sos/experiments/run_audit.jl
```

The default solver is CSDP: no commercial license or repository secret is needed.
The synthesis API accepts an alternative JuMP optimizer, including a separately
installed and licensed MOSEK optimizer. No MOSEK result is claimed by this audit.

Read `results/summary.json` and the individual certificate JSON files. Each run
records the Julia version, code SHA when run in GitHub, and generated Manifest
hash. The workflow retains the resolved Manifest, logs and certificates for 14
days. Retain the exact Manifest and artifacts with any later research release;
the initial Project.toml alone is not a fully pinned transitive environment.
The workflow posts bounded test logs and result summaries only to its own private
same-repository pull request, using a PR-scoped write permission.

Independently replay a certificate without solving another SDP:

```sh
julia --project=julia_sos julia_sos/experiments/replay.jl julia_sos/results/S1_backward_vbc_analytical.json
```

`EXACT_RATIONAL_VERIFIED` requires exact PSD and polynomial residual checks AND
an established invariant verification domain. `NO_CERTIFIED_CANDIDATE` and
`UNVERIFIED_CANDIDATE` do not mean that no certificate exists. `DOMAIN_NOT_ESTABLISHED`
means polynomial obligations alone are insufficient for this safety argument.

## Implementation

- `problems.jl`: exact nominal dynamics and boxes; original versus repaired S2.
- `obligations.jl`: one explicit sign convention for all four formulations.
- `exact.jl`: analytical S1 SOS certificates; rational PSD tests; independent
  proof replay; IBC-to-VBC transport of the actual SOS proof, without re-solving.
- `synthesis.jl`: free polynomial coefficients, sign-symmetric normalization,
  PSD Gram matrices and polynomial coefficient matching.
- `test/runtests.jl`: reciprocal/terminal-loop tests, whole-domain tests,
  tamper rejection, JSON round trips and a real SOS optimization test.
- `experiments/run_audit.jl`: current-case free synthesis and normalization ablation.

Matrices/scales remain numerical constants; the models are convex SDPs, not
unacknowledged bilinear programs. Comparison classes have matched dimensions.
No plots or sampled inequalities are used to certify safety.

See `../docs/SCL_SIMULATION_AUDIT.md` for the exact S1 proof and reviewer mapping.
These examples diagnose the old experiments. They do not establish new VBC
expressiveness or make the manuscript publication-ready.
