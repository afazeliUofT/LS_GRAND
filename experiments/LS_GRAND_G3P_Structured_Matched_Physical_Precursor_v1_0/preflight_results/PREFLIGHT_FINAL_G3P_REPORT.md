# LS-GRAND G3P structured matched precursor report

## Classification

**CONTINUE_TO_FINAL_PHYSICAL_MATCHED_GATE**

The approximate event-triggered route survived orbit-screened structured codes, code-specific baselines, normal-frame neutrality, and the synthetic rare-event screen; one final measured/standard-derived matched physical gate is justified.

This result cannot authorize a field-defining claim.  The exact certificate route
is frozen as a mathematical oracle because v1.1 rejected its practical efficiency.

## Gate status

- **G0_CHAIN_LOCK_AND_CLAIM_DISCIPLINE**: PASS
- **C1_STRUCTURED_CODE_ORBIT_SAFETY**: PASS
- **C2_EVENT_TRIGGER**: PASS
- **C3_APPROXIMATE_STRUCTURED_CODE_FRONTIER**: PASS
- **C4_CODE_SPECIFIC_BASELINES**: PASS
- **C5_NO_SLIP_NEUTRALITY**: PASS
- **C6_SYNTHETIC_PHYSICAL_MIXTURE_SCREEN**: PASS
- **C7_REPRODUCIBILITY**: PASS

## Frozen practical receiver

- normal operation: frozen code-specific decoder (Chase-BCH, polar SC-Flip, or OSD control);
- event trigger: code-aided normalized residual calibrated on independent no-slip frames;
- recovery candidates: globally ordered LS first-valid, path-multiplicity-triggered adaptive L=2, and fixed L=2;
- matched baselines: state-pruned OSD for every code, state-sweep Chase-BCH for eBCH, and state-sweep polar SC-Flip for the polar code;
- channel screen: QPSK with rare persistent +/-pi/2 state jumps, AWGN, residual Wiener phase, and small frequency-offset mismatch;
- code screen: extended BCH(64,45), polar(64,48), and a random-linear control, each with deterministic orbit screening/interleaving.

## Claim discipline

The G0 claim matrix is an immutable scientific chain lock; narrow C06/C07 novelty and patent freedom to operate remain provisional.
The first-valid/tiny-list route is less foundational than the exact fiber theorem and must
be published only if the structured/physical evidence is strong and the narrowed claim
survives independent review.

## Required review files

- `FINAL_G3P_VERDICT.json`
- `gate_status.csv`
- `code_orbit_summary.csv`
- `detector_summary.csv`
- `performance_aggregate.csv`
- `performance_comparisons.csv`
- `no_slip_overhead.csv`
- `physical_mixture.csv`
- `G0_CLAIM_MATRIX_FROZEN.csv`
- `VALIDATION_REPORT.json`
