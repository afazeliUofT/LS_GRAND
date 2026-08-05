# Frozen model and receiver scope

## Channel

- QPSK with a known initial phase state.
- AWGN after an ordinary synchronization/carrier-tracking front end.
- Zero or one persistent residual phase jump of +pi/2 or -pi/2 inside the
  payload.
- Continuous phase drift is treated only as a small residual mismatch term.

## Practical receiver

### LS-FV
Globally merge exact state-conditioned component queues and return the first
candidate satisfying the code-membership oracle.  This is a max-witness
approximation, not marginal ML.

### LS-A2
Start from LS-FV.  Invoke a two-unique-codeword list only when a frozen ambiguity
trigger fires.  Fully marginalize only those two codewords across the modeled
state family and return the better one.  This remains approximate because unseen
codewords are not certified away.

## Exact receiver

The exact marginal decoder remains only as:

- a tiny-instance correctness oracle;
- a benchmark for approximation loss; and
- a source of counterexamples to first-valid decoding.

The current exact stopping certificate is not the practical flagship.

## Code scope

The next gate is restricted to orbit-screened binary codes, initially with
redundancy 16 or 24.  Very-high-rate redundancy-8 codes are outside the initial
claim because the target-length audit found material orbit risks.
