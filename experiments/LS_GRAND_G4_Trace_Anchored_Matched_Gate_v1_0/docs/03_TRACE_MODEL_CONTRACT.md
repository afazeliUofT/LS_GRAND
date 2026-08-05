# Standard-derived trace contract

The gate does not insert a discrete slip into the payload.  It generates random
QPSK, Wiener laser phase with innovation variance

\[\sigma_\phi^2=2\pi\Delta\nu/R_s,\]

AWGN, and a causal fourth-power Viterbi--Viterbi estimate.  The estimated phase
is branch-unwrapped by nearest continuation.  After removing the initial packet
quadrant, packet windows are classified by changes in the estimator's quadrant
offset relative to the simulated true laser phase.

The resulting complex post-CPE gain sequence is applied to independently coded
QPSK packets.  This preserves the residual phase/noise law while avoiding any
code-dependent construction of the carrier-recovery trace.

The primary physical anchor is 28-Gbaud QPSK and 2-MHz combined linewidth,
consistent with the OFC 2014 pilot-symbol phase-unwrapping experiment.  OFC 2012
real-time measurements reported cycle-slip probabilities spanning 1e-12 to
1e-3, establishing that the event is real but highly platform dependent.

This is standard-derived software emulation, not measured hardware data.
