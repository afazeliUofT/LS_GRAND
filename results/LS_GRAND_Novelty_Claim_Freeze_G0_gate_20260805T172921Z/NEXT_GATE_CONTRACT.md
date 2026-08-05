# Next gate contract

**Gate:** `G3P_STRUCTURED_MATCHED_PHYSICAL_PRECURSOR`

**Objective:** Determine whether LS-FV/LS-A2 retains a tail-work or net-overhead advantage against optimized matched receivers on structured codes in a defensible residual-cycle-slip model.

## Algorithms

- LS-FV
- LS-A2 with frozen ambiguity trigger

## Matched baselines

- event-triggered all-state OSD with the identical cached state metric kernel
- C4 forward-backward/BCJR state inference followed by a strong code decoder
- decode-every-state strong code-specific decoder with complete candidate reranking
- ordinary no-slip decoder plus the same trigger
- pilot-assisted and differential alternatives at equal information bits, packet duration, bandwidth, and energy

## Pass conditions

- Across at least two structured code families, LS-FV or LS-A2 has no preregistered material reliability loss versus the strongest matched practical receiver.
- At least one physically defensible region shows approximately 10x or greater p99 normalized-work reduction, or a clear equal-net-rate pilot/energy advantage.
- No-slip p99 overhead remains small under the measured event rate and event trigger.
- Orbit screening removes exact ambiguity without unacceptable redundancy.
- The conclusion survives prior, phase-drift, and one-extra-jump mismatch.

## Stop conditions

- Optimized event-triggered BCJR or code-specific state sweep is within roughly 2x in p99 work at equivalent reliability.
- The physical event rate is too small for unconditional benefit after normal-frame overhead.
- Residual continuous phase uncertainty destroys concentration on a small state set.
- The narrowed approximate combination is directly anticipated by a newly found source or blocking patent claim.
- Structured target codes exhibit unavoidable orbit ambiguity.
