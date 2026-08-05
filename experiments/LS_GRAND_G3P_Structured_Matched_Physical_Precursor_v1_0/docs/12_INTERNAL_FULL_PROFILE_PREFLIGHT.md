# Internal full-profile preflight

The final numerical engine was subjected to the following checks before delivery:

- all 13 unit tests passed;
- a full default `gate` profile completed;
- result schemas and SHA-256 manifests validated with no reported problem; and
- a separately extracted frozen ZIP completed a smoke-profile audit.

The internal full-profile classification was
`CONTINUE_TO_FINAL_PHYSICAL_MATCHED_GATE`.  This only demonstrates that the
package can execute its preregistered decision logic and produce a scientifically
interpretable result.  It is not evidence from the user's WSL environment and
cannot substitute for the pushed run.

The preflight remains subject to the same scientific limits as the user run:
synthetic residual cycle slips, interim Chase-BCH and polar SC-Flip baselines,
implementation-specific Python timing, no final C4 BCJR/iterative or CA-SCL
comparison, no equal-net-rate pilot/differential comparison, and no measured or
standard-derived physical trace.  No preflight result can authorize publication,
field-defining language, or patent clearance.

A compact record is retained in `preflight_results/` solely for package audit.
