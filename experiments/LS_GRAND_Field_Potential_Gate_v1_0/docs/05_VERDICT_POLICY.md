# Verdict policy

## `CONTINUE_FIELD_DEFINING_CANDIDATE`

Requires G0, G2, G3, G4, and G5 to pass, plus a substantive G1 fiber effect.  The
advantage must survive the matched latent-OSD baseline and cannot be explained
only by comparing against a receiver that ignores the latent state.

## `CONTINUE_SIGNIFICANT_BUT_NARROW`

Used when correctness and a reproducible advantage are established, but the
advantage is confined to a clearly bounded class such as single persistent
cycle slips, high-rate short blocks, or pilot-aided operation.  This can still
support a strong paper, but broad field-defining language must be removed.

## `PIVOT_THEORETICAL_OR_APPROXIMATE_ONLY`

Used when fiber marginalization is correct and search finds good candidates, but
exact certification is computationally ineffective.  Recommended pivots include
tighter syndrome-aware upper bounds, admissible outer-state A* bounds, controlled
posterior truncation with error certificates, or an approximate LS-ORBGRAND
receiver with explicit performance/complexity guarantees.

## `STOP_CURRENT_FORM`

Used for a mathematical counterexample, failure against matched baselines,
pervasive non-identifiability, or loss of the claimed scaling advantage after all
work is counted.

## `INCONCLUSIVE_COMPUTE_OR_STATISTICS`

Used when caps, error counts, or sample sizes prevent a defensible conclusion.
The report must identify the exact unresolved gate; it cannot substitute a
positive narrative.
