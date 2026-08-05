# LS-GRAND G3P Structured Matched Precursor v1.0

This is the single bounded numerical step authorized by the novelty/claim-freeze
record at commit `0d2866b091576fe07521172378ded33da79ed545`. It does not revive the original exact
marginalization/stopping thesis as a novelty or practical claim.

## Frozen question

Can an event-triggered latent-cause receiver—first-valid LS-FV, with adaptively
invoked two-codeword marginal rescoring LS-A2—retain a meaningful tail-work
advantage on **structured short codes** against matched state-aware and
code-specific receivers, without harming ordinary no-slip packets?

## Default `gate` profile

The package:

1. verifies the G0 chain lock and blocked/allowed claim language;
2. constructs and orbit-screens extended BCH(64,45), polar(64,48), and a
   random-linear control, using bounded coordinate interleaver search;
3. calibrates an independent code-aided residual alarm on no-slip frames;
4. evaluates LS-FV and adaptive LS-A2 under AWGN plus residual Wiener phase and
   residual frequency offset;
5. compares them with event-triggered all-state OSD, state-sweep Chase-BCH, and
   state-sweep polar SC-Flip receivers;
6. reports BLER, caps, false alarms, generated components, membership work,
   code-specific work, median/p99 latency, and synthetic rare-event mixtures
   separately; and
7. enforces preregistered stop/continue rules.

## Scientific limit

This is a **precursor**, not final physical validation. A positive result can
only authorize a final measured or standard-derived residual-slip gate with
C4 forward-backward/BCJR, stronger code-specific decoding, and equal-net-rate
pilot/differential comparisons. No result from this package can authorize a
field-defining claim.

## Profiles

- `smoke`: installation, tests, schemas, and claim-chain validation only.
- `gate`: the default bounded precursor.
- `stress`: confirmation only after independent adjudication of `gate`.

Use the separately supplied `RUN_LS_GRAND_G3P_STRUCTURED_MATCHED_PHYSICAL_PRECURSOR.py`
from `/home/afazeli2006/LS_GRAND`.
