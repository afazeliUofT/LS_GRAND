# Synthetic physical-model contract

The precursor uses QPSK packets after conventional coarse synchronization. The
normal residual contains AWGN, a small initial phase error, Wiener-like phase
innovation, and residual frequency offset. A rare event adds one persistent
±π/2 jump at an early or uniformly distributed payload location.

The model is deliberately narrower than generic phase noise. It represents a
residual discrete ambiguity transition after ordinary tracking.

The default gate evaluates conditional no-slip/early-slip/uniform-slip data and
forms synthetic unconditional mixtures for slip probabilities 10^-4, 10^-3,
and 10^-2. These mixtures are sensitivity screens, not measured event-rate
claims. A positive result still requires a measured, hardware-emulated, or
standard-derived physical anchor.
