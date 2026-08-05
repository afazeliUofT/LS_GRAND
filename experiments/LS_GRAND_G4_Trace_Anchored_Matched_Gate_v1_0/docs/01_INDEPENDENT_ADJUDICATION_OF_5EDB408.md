# Independent adjudication of precursor commit 5edb408

## Retained positive evidence

- structured eBCH and polar codes were made orbit-safe;
- the synthetic trigger detected 94--99% of forced slips with zero observed
  false alarms in 500 test frames per code;
- LS-FV matched the code-specific state-sweep error count in the precursor;
- median and p99 Python wall-time ratios were favorable against Chase-BCH and
  all-state polar SC-Flip;
- observed normal-frame overhead was small.

## Mandatory repairs

1. `event_state_sweep_osd` used a top-eight state limit; it was not all-state.
2. Zero false alarms in 500 frames does not establish a rare-event false-alarm
   probability below 1e-3.
3. Synthetic 1% mixture quantiles were reweighted from only 200 slip frames and
   had no tail-confidence statement.
4. The polar code was BEC-constructed and the code-specific decoder was SC-Flip,
   not a definitive CA-SCL implementation.
5. No BCJR/iterative, pilot-adjusted, differential, measured, or
   carrier-recovery-generated trace had been tested.

The independent action is one final trace-anchored matched gate with full, posterior-pruned, and state-marginal code-specific baselines, not publication
and not field-defining escalation.
