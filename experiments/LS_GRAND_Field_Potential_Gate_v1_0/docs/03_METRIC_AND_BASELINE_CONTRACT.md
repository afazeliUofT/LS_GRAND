# Metric and baseline contract

## Operations that must remain separate

- `latent_hypotheses_available`: state paths represented by the model.
- `latent_queues_touched`: state queues from which at least one item was popped.
- `residual_patterns_generated`: queue items materialized.
- `membership_queries`: calls to the code-membership predicate.
- `valid_witnesses`: state/noise pairs whose binary word is in the code.
- `unique_codewords_seen`: distinct valid binary codewords encountered.
- `complete_marginal_scores`: codewords for which all modeled states were
  explicitly evaluated by a baseline.
- `state_codeword_likelihoods`: individual \((s,c)\) likelihood evaluations.
- `wall_seconds`: measured end-to-end decoder time in the same Python process.
- `cap_hit`: whether a finite resource cap censored the trial.

No result may call a reduction in one of these quantities a reduction in another.
In particular, fewer membership queries do not imply fewer likelihood operations
or lower wall-clock time.

## Required baselines

1. **Exhaustive marginal ML** on small instances.
2. **First valid latent witness**, using the same queue scheduler as LS-GRAND.
3. **Per-state GRAND sweep**, followed by comparison of the best state/codeword
   witness.
4. **Plain reliability-ordered GRAND** under the no-slip demapper.
5. **Latent-state OSD**, generated from reliability-selected information sets,
   deduplicated across state hypotheses, and scored with the complete marginal
   likelihood across every modeled state.
6. **Oracle-rank diagnostics**, which measure when the exact ML or transmitted
   codeword enters each candidate order without pretending that an oracle is a
   practical stopping rule.

## Fairness rules

- Common random numbers and the same transmitted frame are used across decoders.
- Every decoder sees the same channel parameters except in explicitly labeled
  mismatch experiments.
- Candidate caps and OSD orders are frozen in configuration files.
- A baseline that reaches the cap is not assigned its last observed work count;
  it is marked censored.
- BLER comparisons are paired and include uncertainty intervals.
- Wall-clock comparisons are secondary until algorithmic work counts are
  favorable and implementations are comparably optimized.
