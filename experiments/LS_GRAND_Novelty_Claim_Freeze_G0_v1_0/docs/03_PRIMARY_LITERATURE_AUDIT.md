# Primary-literature audit

## Hidden-state marginalization

BCJR and the sum-product framework already establish exact posterior
marginalization over latent Markov states.  Phase-noise receivers have also
combined factor-graph inference with code constraints and explicit phase-slip
handling.  LS-GRAND may provide a different search implementation, but not a new
generic marginalization principle.

## Sorted-list aggregation and stopping

The Threshold Algorithm considers objects appearing in multiple sorted score
lists, obtains complete object scores by random access, and stops when the best
complete score exceeds the monotone aggregate of the current list thresholds.
For sum aggregation and top one, its threshold is the sum of current list heads.
Mapping state trajectories to lists and codewords to objects gives the core of
the proposal's original exact stopping rule.  Code membership adds a feasibility
filter but does not justify describing the generic threshold logic as new.

## GRAND and channels with memory

GRAND, ORBGRAND, finite-state additive guessing, SGRAND-ISI, Low-Pathwidth GRAND,
symbol/modulation-aware GRAND, and GRAND-assisted demodulation already cover a
wide range of code-agnostic likelihood-ordered candidate generation.  The safe
research question is therefore the precise sparse intra-packet automorphism
ordering and its tail-complexity benefit, not a broad first GRAND-with-memory
claim.

## Code-aided phase inference and list rescoring

Code-aided phase ambiguity resolution, iterative phase-noise decoding,
finite phase-candidate model selection, and candidate-list construction followed
by complete noncoherent rescoring are established.  The remaining novelty, if
any, must lie in the exact interface and sequence of operations of LS-FV/LS-A2.
