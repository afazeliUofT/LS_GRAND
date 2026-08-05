# Prior-art boundary carried forward from G0

The G0 audit at commit `0d2866b` is a chain constraint, not a gate to be rerun by
this numerical package.

Established elements include:

- posterior marginalization over hidden states (BCJR and sum-product);
- sorted-list top-k aggregation with threshold stopping;
- likelihood-ordered GRAND and correlated/channel-memory variants;
- code-aided phase-hypothesis selection and cycle-slip-aware FEC;
- adaptive lists and complete noncoherent/state-marginal rescoring; and
- event-triggered receiver activation.

The only provisional research object retained here is the exact frozen
combination of sparse intra-packet automorphism-component ordering,
arbitrary-code membership, LS-FV/adaptive-LS-A2 operation, and explicit orbit
screening. Patent freedom to operate remains undetermined.
