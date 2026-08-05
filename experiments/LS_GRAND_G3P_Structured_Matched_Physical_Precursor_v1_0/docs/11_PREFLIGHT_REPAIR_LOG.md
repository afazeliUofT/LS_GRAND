# G0 alignment and repair log

This package was derived from an internal structured-code precursor, then
repaired after the completed G0 novelty/claim-freeze at commit `0d2866b`.

Repairs:

- replaced the provisional broad exact-novelty gate with an immutable G0 chain
  lock;
- blocked hidden-state marginalization, Threshold-Algorithm stopping, generic
  code-aided phase selection, and list-rescoring as novelty claims;
- kept only LS-FV/adaptive-LS-A2 plus orbit screening as provisional;
- changed the positive verdict to authorize only a final physical/matched gate;
- removed independent broad novelty adjudication from automatic next steps;
- retained the exact decoder solely as oracle/reference;
- retained the eligible-work-pair threshold repair from commit `ed5a4b4` review;
- explicitly labeled synthetic p99.9 mixtures as a screen rather than measured
  physical evidence.
