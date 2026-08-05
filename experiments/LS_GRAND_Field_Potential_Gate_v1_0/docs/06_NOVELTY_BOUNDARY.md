# Novelty boundary and claim discipline

The proposal must distinguish its contribution from four established research
families:

1. noise-guessing decoding for additive and reliability-ordered channels;
2. GRAND variants that learn or exploit correlated noise memory;
3. code-aided carrier-phase, cycle-slip, and synchronization recovery; and
4. joint hidden-state/channel decoding by trellis, message-passing, list, or OSD
   methods.

The broad statements “coding resolves cycle slips” and “hidden-state decoding is
new” are not defensible novelty claims.  The strongest claim worth testing is:

> For a membership-testable code and a finite, compactly enumerable non-additive
> latent channel, state-conditioned GRAND queues can be aggregated by codeword
> fibers to recover exact marginal-ML decisions, with a computable unseen-
> codeword certificate; in identified regimes this yields a favorable search
> scaling compared with matched state-aware list decoders.

Even this statement has two independent parts:

- a mathematical framework/exactness claim; and
- an empirical efficiency/scaling claim.

The first can survive if the second fails, but it would then support a narrower
theory paper rather than the full field-defining program.  Before publication, a
formal primary-source claim chart must be completed for every theorem and
algorithmic component.  The numerical gate cannot establish novelty by itself.
