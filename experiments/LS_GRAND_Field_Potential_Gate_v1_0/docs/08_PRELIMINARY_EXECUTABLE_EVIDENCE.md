# Preliminary executable evidence

The deterministic unit audit and end-to-end smoke campaign completed before packaging.  The smoke profile is an installation/correctness preflight, not the powered G0–G5 decision campaign.

| Quantity | Value |
|---|---:|
| `profile` | smoke |
| `unit_audit` | PASS |
| `exact_trials` | 8 |
| `non_tied_exact_trials` | 8 |
| `certified_marginal_ml_mismatches` | 0 |
| `certificate_verification_failures` | 0 |
| `marginal_vs_joint_disagreements` | 2 |
| `marginal_vs_first_disagreements` | 2 |
| `median_exact_query_fraction_of_cartesian` | 0.0022638494318181 |
| `median_exact_queue_touch_fraction` | 1.0 |
| `maximum_smoke_log2_coordinate_separation_lower_bound` | 53.74933339356458 |
| `maximum_mean_piecewise_collision_fraction` | 1.0 |
| `automated_smoke_verdict` | INCONCLUSIVE_COMPUTE_OR_STATISTICS |

The final verdict must come from `configs/gate.json` (the wrapper default) or the larger stress profile.  The preflight result is intentionally `INCONCLUSIVE_COMPUTE_OR_STATISTICS`.
