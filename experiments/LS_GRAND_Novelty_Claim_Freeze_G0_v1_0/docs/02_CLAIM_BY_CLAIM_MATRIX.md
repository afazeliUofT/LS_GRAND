# Claim-by-claim novelty matrix

The machine-readable matrix is `data/claims.json`.  The gate produces a CSV copy
with the closest source IDs and final disposition.

Classification meanings:

- `ANTICIPATED_*`: the core principle or workflow is already disclosed; it may
  be used but not claimed as novel.
- `PARTIALLY_ANTICIPATED_COMBINATION`: ingredients are known and the exact
  combination was not found; only narrow combination wording is allowed.
- `NARROW_CANDIDATE_UNRESOLVED`: no identical disclosure was found in this
  bounded search, but the novelty position remains provisional.
- `USEFUL_SPECIALIZATION_NOT_FOUND_AS_IDENTICAL`: potentially publishable as a
  supporting mathematical result, not a foundational field-defining principle.
- `BLOCK`: not authorized by evidence.
