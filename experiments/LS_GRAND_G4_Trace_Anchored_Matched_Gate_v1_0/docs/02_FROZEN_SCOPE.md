# Frozen scope

- QPSK, 28 Gbaud, 2 MHz combined linewidth primary point.
- Wiener laser phase plus AWGN.
- Causal fourth-power Viterbi--Viterbi carrier recovery.
- Known initial quadrant at the packet boundary.
- Packet payload: 32 QPSK symbols carrying 64 coded bits.
- Retain only trace windows with no slip, one persistent +/-pi/2 slip, or an
  explicitly reported out-of-model class.
- Structured codes: eBCH(64,45) and polar(64,48), orbit screened.
- Practical candidates: LS-FV and adaptive LS-A2.
- Exact marginal LS-GRAND remains an oracle only and is not rerun as a practical
  flagship.

- Full code-specific state sweeps run on every one-slip test frame; full-state OSD remains an audit subset because it is a generic rather than decisive competitor.
