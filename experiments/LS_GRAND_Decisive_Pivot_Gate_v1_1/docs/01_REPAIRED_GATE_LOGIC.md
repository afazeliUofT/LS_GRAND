# Repaired gate logic

## F0 — Exactness

Requires deterministic tests, a powered set of non-tied exhaustive-oracle
frames, zero certified decision disagreements, zero invalid certificate stops,
and zero direct-domain likelihood-check failures.

## F1 — Practical marginal-fiber relevance

Reports, rather than assumes, whether summing state paths changes decisions and
reduces paired word errors.  A theorem that first-path and marginal ML can
differ is retained even if this practical gate fails.

## F2 — Empirical noisy rank compression

Uses the actual likelihood queues.  A capped plain rank is a right-censored
observation and contributes only a lower bound to the rank ratio.  The gate must
pass across two code families and two blocklengths for early and middle slips.

## F3 — Approximate latent candidate frontier

The pre-registered frontier contains first-valid and marginal lists of sizes 2,
4, and 8.  Each is compared on paired frames with the same cached state-aware
OSD baseline.  A configuration qualifies only when:

- the LS error-rate excess confidence bound is small;
- the LS BLER confidence interval lies in a nontrivial operating region;
- an operation-class advantage is present;
- wall-clock advantage is also present; and
- cap-hit frames are excluded from work-ratio estimation but retained in error
  accounting.

A list size is not selected after seeing results; all are published.

## F4 — Exact certificate usefulness

Uses the same latent search stopped at first validity as the discovery baseline.
Failed F4 forces an approximate or theory-only pivot.  No failure of F5 or pass
of F2 can upgrade it to an exact-efficiency continue verdict.

## F5 — Target orbit identifiability

Uses target-length exact affine intersection fractions.  It distinguishes:

- unrestricted-code safety;
- code-screenable safety; and
- pervasive non-identifiability.

A continue verdict may be restricted to orbit-safe codes when screening is
shown to retain a sufficiently large ensemble fraction.

## F6 — Reproducibility

The post-run validator checks required files, schemas, nonempty tables, frozen
configuration, git context, result sizes, and SHA-256 values.  Mere file
existence is not a pass.

## Precedence

1. Failed F0: stop mathematical core.
2. Failed F2: stop sparse-cause search thesis.
3. Passed F2 but failed F3: theory-only pivot or stop algorithm.
4. Passed F3 but failed F5: only an explicit orbit-safe restriction can save the
   route.
5. Passed F3 and F5 but failed F4/F1: approximate latent search only.
6. Exact algorithmic continuation requires F0–F5, but remains subject to
   external novelty and physical gates.
