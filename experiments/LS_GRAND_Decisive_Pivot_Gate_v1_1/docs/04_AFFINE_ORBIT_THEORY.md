# Exact affine orbit-collision calculation

For sign-labelled QPSK, every C4 rotation acts affinely on each bit pair:

- `d=0`: identity;
- `d=1`: `(bI,bQ) -> (bQ+1,bI)`;
- `d=2`: `(bI,bQ) -> (bI+1,bQ+1)`;
- `d=3`: `(bI,bQ) -> (bQ,bI+1)`.

For a state path, write the block-diagonal transformation as

`T(x) = A x + b` over GF(2).

Let a binary linear code have generator `G` and parity-check `H`, with a
codeword in column convention `c=G^T u`.  The transformed word is also a
codeword exactly when

`H A G^T u = H b`.

If this system is inconsistent, the collision fraction is zero.  If it is
consistent with rank `rT`, exactly `2^(k-rT)` messages collide, and therefore

`|{c in C : T(c) in C}| / |C| = 2^(-rT)`.

This theorem makes target-length orbit auditing inexpensive and exact.  In the
special 180-degree suffix case, `A=I`; therefore the transformation is a pure
translation.  A linear code is then either completely invariant (`b in C`) or
has no collision at all.  This all-or-none structure explains why tiny,
low-redundancy code experiments can show dramatic collision fractions and why
redundancy/orbit screening must be assessed at the intended length.
