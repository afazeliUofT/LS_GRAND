# Executive scientific audit of the LS-GRAND proposal

## Bottom-line judgment before the numerical gate

The proposal contains a **genuinely strong research kernel**, but the broad
“field-defining” conclusion is not yet earned.  The high-value kernel is not the
generic observation that codes can help resolve phase slips.  It is the proposed
combination of:

1. a code-agnostic membership interface;
2. latent-state-conditioned noise ordering;
3. aggregation of all latent/noise witnesses mapping to the same codeword (a
   *fiber* score);
4. exact marginal rather than first-path or max-path decisions; and
5. a certificate that excludes every ungenerated codeword.

These components could define a useful extension of noise-guessing decoding to
non-additive channels with compact latent structure.  They also create three
serious failure modes that must be tested immediately.

## The three decisive risks

### R1 — Exactness may be mathematically valid but computationally vacuous

The generic unseen-codeword certificate is correct under a finite collection of
complete, non-increasing state-conditioned queues.  Its upper bound can still be
so loose that nearly every state head, or a very deep prefix of every queue, must
be consumed.  In that event the exact decoder is an existence result rather than
an efficient algorithm.

### R2 — A matched list decoder may erase the advantage

A comparison only with bit-domain hard GRAND is not scientifically sufficient.
A QPSK quarter-cycle slip can look like a dense bit error to that baseline, while
being easy for any receiver that explicitly lists slip hypotheses.  The gate
therefore compares LS-GRAND against latent-state OSD with complete marginal
likelihood scoring and against per-state GRAND.  LS-GRAND must win after counting
all state generation, membership queries, complete codeword scoring, and
wall-clock time separately.

### R3 — Group symmetries can make the state unidentifiable

QPSK rotations induce affine transformations of bit labels.  For some codes,
rates, slip positions, or global rotations, a transformed codeword can remain in
the code.  No decoder can remove a true statistical ambiguity.  The gate measures
code-orbit intersections and the corresponding exact small-block ambiguity.

## Why the idea is still worth an early gate

An early persistent quarter-cycle slip has a latent description consisting of a
slip time and a rotation increment.  Its description grows only logarithmically
with blocklength, while the uncompensated hard-decision difference can occupy a
linear fraction of the bits.  This creates a real possibility of a search-
coordinate separation: polynomially many latent hypotheses versus an
exponentially ranked apparent bit pattern.  That separation is the proposal's
best path to a major contribution.

## Current scientific status

- **Theory status:** the finite-queue fiber identity and stopping theorem are
  sound once all assumptions are made explicit.  The proposal must not extend
  the certificate to lazily unopened state trajectories without an outer-tail
  bound.
- **Novelty status:** potentially significant but narrower than the broad
  motivation.  The claim should be about code-agnostic exact fiber aggregation
  and certified marginal decoding, not about the general use of code constraints
  for synchronization or phase-slip resolution.
- **Practical status:** unknown until the gate measures certificate tightness and
  matched-baseline scaling.
- **Recommended action now:** continue only through this bounded gate.  Do not yet
  commit to a full field-defining program.
