# Pilot and differential controls

The executable DQPSK control carries the same codeword through 32 differential
increments plus one reference symbol.  Differential observations are decoded by
the same normal code-specific decoder, and the extra symbol is included in net
rate.

The pilot control is deliberately optimistic: it assumes that the 1.56% pilot
overhead reported for a 28-Gbaud, 2-MHz QPSK experiment completely removes
cycle-slip losses and adds no decoder cost.  It is a goodput upper bound for the
pilot alternative, not a detailed pilot receiver simulation.  LS must at least
match this optimistic rate-accounted bound to pass.
