# Metric and baseline contract

The campaign never treats the following as interchangeable:

- `components_generated`: state-word queue outputs or OSD reprocessings;
- `membership_queries`: distinct syndrome/code-membership tests;
- `complete_marginal_candidates`: codewords completely rescored over states;
- `state_word_metric_evals`: state-codeword likelihood terms;
- `bit_metric_accumulations`: binary-coordinate contributions used in cached
  marginal scoring;
- `osd_reprocessings`: MRB information patterns encoded;
- `latent_queues_touched`: distinct state queues popped or selected;
- `wall_seconds`: end-to-end Python execution on the same process; and
- cap, certification, BLER, and confidence intervals.

The state-aware OSD baseline uses exact cached QPSK flip metrics for complete
marginal rescoring.  Every state contributes OSD-0; by default every modeled
state contributes higher-order perturbations.  It is stronger and more
symmetrically implemented than v1.0, but it is not a substitute for an
optimized code-specific BCJR/iterative/pilot-adjusted receiver.

The first-valid LS mode is explicitly labeled approximate/max-witness.  Marginal
lists are also approximate unless an exact unseen-codeword certificate fires.
