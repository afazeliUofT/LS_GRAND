# Decision policy

## `CONTINUE_TO_FINAL_PHYSICAL_MATCHED_GATE`

All G0 and C1–C7 checks pass. This authorizes one final measured or
standard-derived physical gate. It does not authorize a paper submission,
field-defining language, or patent claim.

## `CONTINUE_CONDITIONAL_ON_EXTERNAL_ALARM_ONLY`

Conditional recovery is promising, but the internal detector, normal-frame
neutrality, or synthetic mixture fails. Continue only if a defensible external
tracker alarm is available.

## `NARROW_TO_APPLICATION_ONLY`

The generic state-aware OSD baseline is beaten, but at least one code-specific
baseline is not. Retain only a narrow receiver/application result.

## `STOP_APPROXIMATE_ROUTE`

The approximate structured-code route does not survive matched baselines.

## `STOP_CODE_IDENTIFIABILITY`

Target structured codes cannot be made orbit-safe within the frozen budget.

## `STOP_CLAIM_CHAIN_VIOLATION`

The G0 blocked/allowed claim boundary was altered or lost.

## `STOP_REPRODUCIBILITY`

Tests, schemas, or hashes fail.

## `INCONCLUSIVE_SMOKE_ONLY`

Smoke mode validates software and policy only.
