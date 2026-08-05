# Nonblocking erratum for commit ed5a4b4

The v1.1 F3 adjudicator filtered cap-hit pairs before estimating work ratios but
applied its minimum-sample threshold to `paired_trials` rather than
`eligible_work_pairs`.

Required repair in the next numerical package:

```python
perf_cmp["eligible_work_pairs"] >= int(th["f3_min_eligible_work_pairs"])
```

This does not change the ed5a4b4 verdict because sufficient uncapped qualifying
first-valid/tiny-list cases remain in both dense and sparse families.  It must
nevertheless be fixed before the next scientific gate.
