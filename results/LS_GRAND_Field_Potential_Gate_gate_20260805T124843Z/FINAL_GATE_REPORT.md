# LS-GRAND field-potential gate report

**Profile:** `gate`

**Automated verdict:** `CONTINUE_SIGNIFICANT_BUT_NARROW`

A matched-baseline advantage exists, but robustness, identifiability, or fiber significance remains limited.

## Gate outcomes

| Gate | Pass |
|---|---:|
| G0_EXACTNESS | PASS |
| G1_FIBER_EFFECT | FAIL |
| G2_MATCHED_SCALING_ADVANTAGE | PASS |
| G3_CERTIFICATE_USEFULNESS | FAIL |
| G4_IDENTIFIABILITY_AND_ROBUSTNESS | FAIL |
| G5_REPRODUCIBILITY | PASS |

## Scientific interpretation

A positive result is meaningful only if the matched latent-OSD baseline is competitive in BLER and the LS-GRAND advantage remains after state-codeword likelihood operations, membership queries, caps, and wall-clock time are shown separately.  A failed certificate gate means that the exact theorem may remain correct while the practical exact-decoding claim fails.

## Files to inspect

- `gate_status.csv`
- `exactness_trials.csv`
- `performance_aggregate.csv`
- `certificate_aggregate.csv`
- `collision_aggregate.csv`
- `rank_separation_aggregate.csv`
- `mismatch_aggregate.csv`
- `REPRODUCIBILITY_MANIFEST.json`

## Claim discipline

This is an early evidence gate, not proof that future work will be field-defining.
