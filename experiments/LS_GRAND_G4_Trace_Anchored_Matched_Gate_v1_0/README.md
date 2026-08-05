# LS-GRAND G4 Trace-Anchored Matched Gate v1.0

This is the final bounded numerical gate for the **narrow approximate** LS-GRAND
route.  It cannot revive the rejected exact-foundational or practical-certificate
claims.

The gate replaces forced phase slips by a standard-derived QPSK carrier-recovery
trace.  A causal fourth-power Viterbi--Viterbi estimator is driven by a Wiener
laser-phase process and AWGN.  One-slip packet windows arise endogenously from
carrier-recovery branch errors.  The receiver is tested on orbit-screened
extended BCH(64,45) and polar(64,48) codes.

Primary candidates:

- event-triggered LS-FV;
- event-triggered adaptive LS-A2.

Mandatory competitors:

- full-state OSD audit (not the top-eight precursor baseline);
- all-state code-specific BCH Chase or polar SC/SC-Flip;
- a faster code-aided posterior-pruned code-specific state sweep;
- an exact finite-hypothesis state-marginal (forward--backward-equivalent) demapper followed by a code-specific decoder;
- executable DQPSK control;
- an optimistic 1.56% pilot-overhead goodput bound.

The strongest positive classification is
`READY_FOR_FOCUSED_SUBSTANTIAL_PAPER_PROGRAM`.  Field-defining authorization is
hard-coded false.

Run through the supplied repository wrapper.  Direct use:

```bash
python -m lsgrand_g4.cli test --package-root .
python -m lsgrand_g4.cli run --package-root . --config configs/smoke.json \
  --output-root /tmp --run-name g4_smoke
```
