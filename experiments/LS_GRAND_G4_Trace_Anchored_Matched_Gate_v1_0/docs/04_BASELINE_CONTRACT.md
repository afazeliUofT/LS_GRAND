# Baseline and accounting contract

The final gate separates:

- generated state-word components;
- membership queries;
- full state-word likelihood evaluations;
- bit-metric accumulations;
- OSD reprocessings;
- BCH decoding attempts;
- preprocessing state metrics;
- wall time.

Mandatory competitors are:

1. full-state OSD, with every modeled state reprocessed on the audit subset;
2. all-state BCH Chase or all-state polar SC/SC-Flip;
3. code-aided posterior-pruned BCH or polar state sweep using the normal decoded
   word to rank states;
4. exact finite-hypothesis state marginalization (the frozen forward--backward/BCJR-equivalent front end) followed by BCH Chase or polar SC/SC-Flip;
5. DQPSK with one reference symbol;
6. an optimistic pilot phase-unwrapping bound.

A positive result must beat both full and posterior-pruned code-specific
competitors.  The earlier top-eight OSD result is not used as a decisive claim.

A baseline comparison can qualify in either of two preregistered ways:

- **speed route:** statistically compatible reliability and at least the frozen median/p99 speed advantage;
- **reliability route:** a statistically resolved conditional-slip BLER improvement of at least the frozen amount while LS is no more than two times slower.

This prevents an extremely fast but materially unreliable front end from invalidating a useful reliability gain.
