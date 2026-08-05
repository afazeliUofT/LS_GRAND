# Preregistered gates

## G0 — Chain lock and claim discipline

Pass only if commit `0d2866b` is the required ancestor, C01/C02/C03/C04/C10/
C11/C12 remain blocked, and only the provisional C05–C08 combination is carried
forward. Failure is `STOP_CLAIM_CHAIN_VIOLATION`.

## C1 — Structured-code orbit safety

Both eBCH(64,45) and polar(64,48) must satisfy the frozen maximum one-slip affine
collision threshold within the bounded interleaver search.

## C2 — Event trigger

On independent test frames, false alarms must remain below the frozen limit and
early/uniform-slip detection must exceed their frozen minima.

## C3 — Generic matched frontier

Against event-triggered all-state OSD, a candidate must satisfy the paired-trial,
eligible-work, BLER, cap, median-wall, and p99-wall thresholds. Both structured
families must qualify.

## C4 — Code-specific baselines

The surviving candidate must also beat state-sweep Chase-BCH and state-sweep
polar SC-Flip by the frozen median-wall threshold without a material BLER loss.

## C5 — No-slip neutrality

False alarms and median/p99 no-slip wall-time overhead must remain below the
frozen limits in both structured code families.

## C6 — Synthetic rare-event mixture

At the frozen 1% screening prior, the same candidate that passed C3–C5 must have
no material unconditional BLER loss and at least the frozen p99.9 tail-latency
ratio against the code-specific state-sweep baseline. This is not final physical
validation.

## C7 — Reproducibility

Tests, schemas, hashes, seeds, caps, and result validation must pass.
