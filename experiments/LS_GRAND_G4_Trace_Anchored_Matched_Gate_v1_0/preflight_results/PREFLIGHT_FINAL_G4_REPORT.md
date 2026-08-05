# LS-GRAND G4 trace-anchored matched-gate report

## Classification

**INCONCLUSIVE_SMOKE_ONLY**

Smoke mode validates software and schemas only.

This gate cannot revive the rejected exact-foundational route and cannot authorize
a field-defining claim.  Its strongest possible positive result is authorization
to develop a narrowly scoped substantial paper around event-triggered LS-FV/LS-A2.

## Gate status

- **G0_CHAIN_LOCK**: PASS
- **P1_STANDARD_DERIVED_TRACE**: PASS
- **P2_ORBIT_SAFETY**: PASS
- **P3_EVENT_TRIGGER**: PASS
- **P4_STRONG_MATCHED_RECOVERY**: FAIL
- **P5_NO_SLIP_NEUTRALITY**: PASS
- **P6_OBSERVED_TRACE_MIXTURE**: FAIL
- **P7_RATE_ACCOUNTED_CONTROLS**: FAIL
- **P8_REPRODUCIBILITY**: PASS

## Physical provenance

The executable trace is standard-derived rather than measured hardware data.  It
uses a QPSK Wiener laser-phase model and causal fourth-power Viterbi--Viterbi
carrier recovery.  One-slip frames are produced endogenously by the carrier
recovery; no discrete slip is forced into the final trace.

- Total carrier-recovery frames generated: 500
- Observed one-slip frames: 37
- Observed one-slip rate: 0.074
- Symbol rate: 28000000000.0
- Combined linewidth: 20000000.0
- VV window: 9

## Corrected baseline status

The earlier precursor's top-eight state OSD result is not treated as an all-state
baseline.  This gate includes full-state OSD audits, all-state code-specific
recovery, and a faster code-aided posterior-pruned code-specific competitor.  A
positive verdict requires LS to survive both full and pruned code-specific
baselines in both structured code families.

## Rate controls

The gate includes executable differential-QPSK control and an optimistic
literature-anchored pilot-overhead bound.  The pilot bound assumes perfect cycle
slip removal and is deliberately favorable to the conventional alternative.

## Claim discipline

- C01/C02/C03/C04/C10/C11/C12 remain blocked.
- C05--C08 remain provisional narrow claims.
- Patent freedom to operate is not determined.
- Measured deployment readiness is not authorized.
