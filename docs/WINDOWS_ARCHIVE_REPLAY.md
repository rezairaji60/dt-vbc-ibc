# Windows reviewer-archive replay

The standalone Julia 1.10.10 run reported 85 passing tests, 16 verified
multi-function cases, four scalar baselines and two analytical witnesses.
Its subsequent archive replay stopped before mathematical verification with
`Digest mismatch: BB_rotation_backward_ibc_free_certificate.json`.

The log does not contain the local mismatching bytes. CRLF conversion is a
reproduced cause, not an assumption that arbitrary corruption is harmless.
The committed archive contains 69 indexed files and 36 proof bundles. All
69 original digests pass. A clone with core.autocrlf=true and no attributes
changes all 69 digests. `/evidence/reviewer/** -text` prevents this conversion.
See https://git-scm.com/docs/gitattributes (text/Unset).

## Existing Windows checkout

Pull the corrected main without changing package versions. Then invoke the
already installed Julia 1.10.10 executable (no new download is needed):

```
julia --project=julia_sos julia_sos/experiments/repair_archive_checkout.jl
julia --project=julia_sos julia_sos/experiments/repair_archive_checkout.jl --apply
julia --project=julia_sos julia_sos/experiments/replay_all.jl
```

Replace `julia` by the absolute executable when Juliaup has no matching channel.
The first command is read-only and optional. The --apply command reads exact
blobs and the index from the current Git HEAD, verifies every original digest,
and permits only an exact LF-to-CRLF expansion of committed bytes. It validates
the ENTIRE archive before writing. Any unrelated edit, missing file, symlink,
or committed digest disagreement aborts without a repair. It backs up affected
checkout bytes outside the repository and restores the original bytes. An
already-correct checkout is a no-op. It does not rehash an altered archive,
change coefficients, update packages, or bypass the replay verifier.

Fresh clones use byte-preserving attributes automatically. Tests run under
both Linux and Windows, exercise core.autocrlf=true, require all 69 original
digests, reject intentional corruption, check backups and idempotence, and
then replay all 36 proofs with the unchanged scientific verifier.

The scientific archive and its index remain unchanged. This fixes artifact
transport, not mathematical results; a test log is not a substitute for
independent review of the theorems or the checker implementation.
