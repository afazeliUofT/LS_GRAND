# LS-GRAND Field-Potential Gate v1.0

This package is a falsification-first scientific gate for the proposal
**Latent-State GRAND: A Field-Defining Research Program**.  It does not assume
that the proposal is correct, novel, scalable, or practically useful.  It asks
whether the idea survives the smallest decisive mathematical and numerical
experiments before a long research program is funded.

## What the gate tests

1. **Exactness:** exhaustive marginal-ML agreement on small instances, including
   cases where first-hit and joint-MAP rules are wrong.
2. **Identifiability:** affine rotation-orbit collisions and irreducible ambiguity
   under QPSK cycle slips.
3. **Structural search advantage:** whether a low-description-length latent slip
   replaces a high-Hamming-weight apparent bit corruption.
4. **Matched competition:** latent-state OSD with complete marginal rescoring,
   per-state GRAND, plain reliability-ordered GRAND, and exhaustive ML where
   feasible.
5. **Certificate tightness:** whether exact unseen-codeword bounds stop early or
   force most state queues to be explored.
6. **Robustness:** code family, rate, blocklength, SNR, slip location, and model
   mismatch.

## Run profiles

- `smoke`: fast installation and correctness check.
- `gate`: the default decision campaign used by the supplied WSL wrapper.
- `stress`: a larger campaign for a later independent audit.

All random seeds, caps, metrics, and verdict thresholds are frozen in JSON
configuration files.  The adjudicator reports one of:

- `CONTINUE_FIELD_DEFINING_CANDIDATE`
- `CONTINUE_SIGNIFICANT_BUT_NARROW`
- `PIVOT_THEORETICAL_OR_APPROXIMATE_ONLY`
- `STOP_CURRENT_FORM`
- `INCONCLUSIVE_COMPUTE_OR_STATISTICS`

A positive verdict means only that the proposal cleared this early gate.  It is
not a publication claim.

## Repository outputs

The campaign writes a timestamped directory under `results/`.  The wrapper
commits only source/configuration files and compact review artifacts:

- `FINAL_GATE_VERDICT.json`
- `FINAL_GATE_REPORT.md`
- aggregate CSV files
- figures
- environment and reproducibility manifests
- test logs

Raw per-frame traces remain local unless explicitly requested.  `.venv`, caches,
large raw traces, and temporary files are never staged by the wrapper.
