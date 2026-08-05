# Post-run review handoff

Return either the pushed commit hash or the new directory name under `results/`.
The minimum review set is:

1. `FINAL_DECISIVE_VERDICT.json`
2. `FINAL_DECISIVE_REPORT.md`
3. `gate_status.csv`
4. `exact_fiber_summary.json`
5. `rank_probe_aggregate.csv`
6. `orbit_aggregate.csv`
7. `performance_comparisons.csv`
8. `certificate_comparisons.csv`
9. `VALIDATION_REPORT.json`
10. `REPRODUCIBILITY_MANIFEST.json`

Raw per-frame and per-transform CSVs remain necessary for an independent audit
and are committed automatically.
