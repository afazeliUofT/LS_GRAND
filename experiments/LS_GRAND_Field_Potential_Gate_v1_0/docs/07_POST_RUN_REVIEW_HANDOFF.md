# Post-run review handoff

After the wrapper finishes and pushes, provide the latest git commit hash or the
name of the timestamped `results/LS_GRAND_Field_Potential_Gate_*` directory.
A rigorous review should read, in order:

1. `FINAL_GATE_VERDICT.json`
2. `FINAL_GATE_REPORT.md`
3. `gate_status.csv`
4. `exactness_trials.csv`
5. `performance_aggregate.csv`
6. `certificate_aggregate.csv`
7. `collision_aggregate.csv`
8. `rank_separation_aggregate.csv`
9. `mismatch_aggregate.csv`
10. `REPRODUCIBILITY_MANIFEST.json`

Raw frame-level files are retained locally under the run directory but are not
pushed by default.  They can be packaged later only if an aggregate anomaly
requires independent replay.
