# LS-GRAND Decisive Pivot Gate v1.1

This is a falsification-first follow-up to commit
`531fdae2c27cb798edfe1bff6a8269f2fa341e29`.  It does **not** rerun the same
campaign with more trials.  It repairs the scientific questions that v1.0 did
not actually resolve.

## Independent adjudication of v1.0

The v1.0 run established that the finite-state exact decoder is internally
correct on its small exhaustive oracle.  It did not establish a field-defining
or even a secure “continue narrow” conclusion:

- exactness passed;
- the practical marginal-fiber effect failed;
- the exact stopping certificate touched every latent queue in every tested
  configuration and often added large query overhead;
- the apparent OSD advantage was measured against a truncated, uncached Python
  implementation and often at BLER near 0.5;
- the orbit-collision failure was extrapolated from tiny high-rate codes rather
  than computed at target lengths;
- the “rank separation” diagnostic was a noiseless Hamming combinatorial lower
  bound, not the actual noisy reliability-ordered rank requested by the
  proposal;
- novelty and physical relevance were not executed as numerical gates; and
- the verdict branch contradicted the preregistered policy by returning
  `CONTINUE_SIGNIFICANT_BUT_NARROW` even though certificate usefulness and
  identifiability failed.

The scientifically correct current classification is:

> **Pivot the present exact-efficiency claim, while allowing one bounded salvage
> gate for the low-description latent candidate-search principle.**

## What v1.1 measures

1. **Powered exact/fiber audit.** Exhaustive marginal ML, joint MAP, first-valid
   latent search, and certified LS-GRAND are compared on paired frames.  The
   test reports decision-switch frequency, paired error discordance, and the
   effective number of state paths supporting the winning codeword.
2. **Actual noisy rank probe.** The true state-codeword component is located in
   the global latent queues, and the transmitted word is located in the actual
   no-slip reliability queue.  Capped ranks are explicitly censored and used
   only as lower bounds.
3. **Target-length affine orbit audit.** For every one-slip C4 transformation,
   the exact fraction of codewords mapped back into the code is obtained from a
   GF(2) rank/consistency calculation.  No target-length codeword enumeration is
   required.
4. **Repaired practical frontier.** Pre-registered LS modes
   (`first-valid`, marginal lists of sizes 2, 4, and 8) are compared against a
   cached, vectorized, state-aware OSD union.  Candidate generation,
   membership queries, marginal scores, state-word metrics, bit-metric
   accumulations, wall time, BLER, caps, and certificate behavior remain
   separate.
5. **Correct verdict precedence.** Failed exact certification can no longer be
   upgraded to a continue verdict by unrelated passes.  Smoke mode can never
   issue a scientific classification.

## Verdict vocabulary

- `CONTINUE_EXACT_ALGORITHMIC_CANDIDATE_PHYSICAL_GATE_PENDING`
- `CONTINUE_APPROXIMATE_LATENT_SEARCH_CANDIDATE`
- `CONTINUE_ONLY_WITH_ORBIT_SAFE_CODE_RESTRICTION`
- `PIVOT_THEORY_ONLY_OR_STOP_ALGORITHM`
- `STOP_SPARSE_CAUSE_SEARCH_THESIS`
- `STOP_MATHEMATICAL_CORE`
- `STOP_CURRENT_FORM`
- `INCONCLUSIVE_SMOKE_ONLY`

No v1.1 outcome authorizes the phrase **field-defining**.  Formal novelty,
platform-specific physical evidence, and optimized code-specific matched
receivers remain mandatory external gates.

## Execution

The supplied WSL wrapper verifies the ZIP SHA-256, checks that commit `531fdae`
is an ancestor of the current branch, installs the package into the existing
repository, uses `/home/afazeli2006/LS_GRAND/.venv`, runs the frozen profile,
validates hashes and schemas, stages only reviewable artifacts, commits them,
and pushes the current branch.  It never executes a terminal-close command.

The default profile is `gate`.  `LSGRAND_V11_PROFILE=smoke` is an installation
check only.  `LSGRAND_V11_PROFILE=stress` increases the frozen statistical
campaign.
