# Independent scientific adjudication of commit 531fdae

## Bottom line

The numerical evidence supports the *existence of a useful latent search
coordinate*, but it rejects the present claim that the supplied exact
certificate is an efficient realization of marginal ML.  The automated label
`CONTINUE_SIGNIFICANT_BUT_NARROW` is not consistent with the package's own
preregistered decision policy.

## What is solid

- The QPSK C4 map, GF(2) code machinery, state-conditioned likelihood queues,
  exhaustive marginal oracle, and finite-queue certificate passed deterministic
  tests.
- Across 120 non-tied tiny-code frames, certified LS-GRAND had zero disagreement
  with exhaustive marginal ML and zero verified certificate failures.
- The sparse physical cause creates a severe mismatch for a no-slip reliability
  search.  Plain reliability GRAND failed badly in the forced-slip cases.
- The source/reproducibility handoff was disciplined: a frozen package hash,
  frozen configuration, raw per-frame data, aggregates, figures, and a bounded
  git allowlist were committed.

## What is not established

### Marginal fibers

Marginal ML disagreed with joint MAP on 14 frames and with first-valid on 13,
but had 57 errors versus 53 and 55.  This does not contradict ML optimality;
the sample is small.  It does mean that the campaign did not demonstrate a
practical benefit from state-path aggregation.

### Exact certificate

Every tested configuration had median and p90 latent-queue touch fraction equal
to one.  Median certificate-over-discovery overhead ranged from about 9.5 to
64, and the late-slip case had p90 overhead above 500.  This is a direct failure
of the preregistered exact-certificate usefulness gate.

### Matched baseline comparison

The large state-metric ratios arose against an OSD baseline that:

- perturbed only 16, 24, or 32 selected states;
- used OSD order two with a fixed reliability pool;
- recomputed complete symbol likelihoods in Python for every retained
  state-codeword pair; and
- was compared in several configurations where both decoders had BLER from
  roughly 0.38 to 0.53.

Those numbers are promising enough to motivate a repaired baseline, but they
are not a decisive matched-receiver advantage.

### Rank diagnostic

The reported “rank separation” was computed from the noiseless apparent Hamming
weight and `sum_{i<d} binom(n,i)`.  It did not locate the transmitted word in the
actual soft/reliability queue and did not locate the true state-word component
in the actual global latent enumeration.

### Orbit identifiability

The maximum mean piecewise collision fraction of one third came from enumerated
`[12,8]` and `[16,12]` random codes.  For QPSK rotations the label map is affine,
so target-length collision fractions can be computed exactly from GF(2) ranks.
Tiny-code maxima are a warning, not a valid target-length conclusion.

### Missing proposal gates

The proposal's G0 was novelty/claim freeze, whereas the package renamed G0 as
exactness.  The proposal's physical G4 was not run.  Synthetic prior/noise/two-
slip mismatch is not evidence that a relevant wireless or optical platform has
the assumed residual event at a useful rate.

## Correct current decision

1. Do not launch broad exact-decoding theory, hardware, code design, or a
   field-defining publication program.
2. Do not discard the sparse-cause search principle yet.
3. Execute one repaired, bounded salvage gate focused on actual noisy ranks,
   target-length identifiability, and the first-valid/small-list frontier against
   a cached matched baseline.
4. If that gate fails, stop the algorithmic route.  A theorem-only paper may
   remain possible if novelty survives independent review.
