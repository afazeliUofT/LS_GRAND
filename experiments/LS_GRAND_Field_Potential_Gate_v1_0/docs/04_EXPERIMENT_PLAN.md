# Experiment plan

## E0 — Deterministic mathematical audit

- GF(2) rank, inverse, systematic generators, and membership.
- QPSK rotation/label maps for the full \(C_4\) group.
- Monotone and duplicate-free subset-sum enumeration.
- Synthetic multi-queue stopping certificates, including adversarial examples.
- Direct and log-domain likelihood agreement.

## E1 — Exhaustive small-block oracle

Random small systematic codes are tested over uncertain one-slip QPSK-AWGN
channels.  Every codeword and every modeled state is enumerated.  Certified
LS-GRAND, first-hit, joint-MAP, and exhaustive marginal ML are compared frame by
frame.  Ties are recorded rather than forced into mismatches.

## E2 — Fiber and certificate anatomy

For each exact frame the campaign records fiber multiplicity, posterior mass by
state, certificate margin, number and fraction of queues touched, query overhead
beyond the first appearance of the ML codeword, and the fraction of the finite
Cartesian state/word space consumed.

## E3 — Affine orbit collisions

Global and piecewise QPSK rotations are applied to every codeword for small
codes.  The fraction remaining in the code, collision multiplicity, and
slip-location dependence are recorded.  Larger cases use reproducible sampling.

## E4 — Rank-separation experiment

Noiseless and high-SNR forced slips measure the uncompensated Hamming distance,
a combinatorial lower bound on hard-GRAND rank, latent-family size, and the rank
of the true state under the frozen state order.  This tests the proposal's
structural premise independently of decoder implementation.

## E5 — Matched performance/scaling campaign

Selected high-rate blocklength/rate/SNR points compare certified LS-GRAND,
uncertified first-hit LS search, state sweep, plain GRAND, and marginally rescored
latent OSD.  Aggregate BLER and every work metric are reported with cap rates.

## E6 — Model mismatch

The true channel includes prior mismatch, noise mismatch, and a configured
fraction of two-slip trajectories while the decoder retains its frozen one-slip
model.  The campaign tests whether the claimed gain is robust or a fragile model-
matching artifact.

## E7 — Automatic adjudication

The final JSON and Markdown reports evaluate G0–G5 mechanically.  The report also
lists the smallest next experiment capable of resolving any inconclusive gate.
