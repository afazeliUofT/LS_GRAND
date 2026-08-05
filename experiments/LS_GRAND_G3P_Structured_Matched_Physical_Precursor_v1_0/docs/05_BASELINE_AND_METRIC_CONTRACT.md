# Baseline and metric contract

## Candidate receivers

- event-triggered LS-FV;
- event-triggered adaptive LS-A2;
- fixed L=2 only when explicitly enabled as a diagnostic.

## Baselines in this precursor

- ordinary no-slip code-specific receiver using the same trigger architecture;
- event-triggered state-pruned OSD with the same cached state metric kernel and
  complete all-state marginal reranking;
- event-triggered state-sweep Chase-BCH for eBCH(64,45);
- event-triggered state-sweep polar SC with frozen SC-Flip trials for polar(64,48).

## Accounting

Report separately:

- generated state-word components;
- distinct membership queries;
- valid and distinct candidates;
- complete state-word metric evaluations and bit-metric accumulations;
- OSD or code-specific reprocessings;
- detector and preprocessing work;
- cap/censoring rate;
- same-platform wall time, including p50, p90, and p99 where available;
- no-slip false-alarm and latency overhead;
- conditional and synthetic-unconditional BLER.

No membership query is equated with a Chase, SC-Flip, OSD, or future BCJR
execution.
