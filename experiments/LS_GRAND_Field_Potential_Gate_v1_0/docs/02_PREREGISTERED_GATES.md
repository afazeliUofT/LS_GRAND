# Pre-registered decision gates

Thresholds are evaluated from aggregate files produced by the campaign.  A cap
hit is treated as censored work, never as a successful low-complexity decode.

## G0 — Mathematical and implementation exactness (mandatory)

Pass only if all conditions hold:

- deterministic unit tests pass;
- certified LS-GRAND agrees with exhaustive marginal ML on every non-tied exact
  trial;
- reported certificates are independently rechecked from complete enumeration;
- all returned words pass the code membership test;
- zero silent queue-order violations occur;
- numerical log-domain and direct-probability scores agree on tiny instances.

Any reproducible mismatch gives `STOP_CURRENT_FORM` until repaired.

## G1 — Nontrivial marginal-fiber effect

At least one non-degenerate region must exhibit one of the following:

- a statistically resolved marginal-ML advantage over first-hit or joint-MAP;
- recurrent fibers with multiplicity greater than one that materially change
  posterior margins; or
- a certified reduction in complete codeword scoring relative to exhaustive
  state rescoring.

Failure does not make the decoder incorrect, but narrows the contribution to a
state-aware search method and prevents the strongest “fiber” claim.

## G2 — Structural and empirical search advantage (mandatory for a major claim)

Against the strongest matched baseline that attains statistically compatible
BLER, require:

- at least a 10x median reduction in the primary work metric in two independent
  code/channel families, and
- a favorable scaling trend over at least three blocklengths, with the paired
  bootstrap 95% interval excluding parity at the largest blocklength.

A 100x reduction across two families, without an offsetting explosion in another
work metric, is classified as strong evidence.

The primary metric is selected before looking at results and cannot combine
incommensurate operations.  Membership queries, generated latent paths,
generated residual patterns, complete marginal scores, and wall-clock time are
all reported separately.

## G3 — Certificate usefulness (mandatory for an efficient *exact* claim)

Pass if, in the target operating region:

- certified decoding completes within the frozen cap on at least 95% of frames;
- median query overhead relative to the same search stopped at the oracle winner
  is at most 10x;
- the median fraction of latent queues touched is below 0.50; and
- the 90th percentile fraction is below 0.90.

If search is good but certification fails, the correct verdict is
`PIVOT_THEORETICAL_OR_APPROXIMATE_ONLY`, not an exact-complexity success.

## G4 — Identifiability and robustness

Pass if the favorable region survives:

- dense and sparse systematic random linear codes;
- at least two rates;
- early, middle, and late slips;
- prior mismatch by the configured factors;
- noise-variance mismatch; and
- a small unmodeled two-slip component.

The exact ambiguity floor must be stated.  A pilot-aided success cannot be
reported as a pilotless result.

## G5 — Reproducibility and claim discipline

Pass only if the final report includes all caps, seeds, denominators, confidence
intervals, censored observations, software versions, and git identifiers.  The
verdict engine cannot upgrade `INCONCLUSIVE` to `CONTINUE` through prose.
