from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _u8(a: np.ndarray | Iterable[int]) -> np.ndarray:
    return np.asarray(a, dtype=np.uint8) & 1


def rref(a: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return GF(2) reduced row-echelon form and pivot columns."""
    m = _u8(a).copy()
    rows, cols = m.shape
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        candidates = np.flatnonzero(m[r:, c])
        if candidates.size == 0:
            continue
        p = r + int(candidates[0])
        if p != r:
            m[[r, p]] = m[[p, r]]
        for rr in range(rows):
            if rr != r and m[rr, c]:
                m[rr] ^= m[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return m, pivots


def rank(a: np.ndarray) -> int:
    return len(rref(a)[1])


def inverse(a: np.ndarray) -> np.ndarray:
    a = _u8(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("GF(2) inverse requires a square matrix")
    n = a.shape[0]
    aug = np.concatenate([a.copy(), np.eye(n, dtype=np.uint8)], axis=1)
    rr, pivots = rref(aug)
    if pivots[:n] != list(range(n)) or not np.array_equal(rr[:, :n], np.eye(n, dtype=np.uint8)):
        raise ValueError("matrix is singular over GF(2)")
    return rr[:, n:]


def nullspace(a: np.ndarray) -> np.ndarray:
    """Rows form a GF(2) basis for the nullspace of a."""
    rr, pivots = rref(a)
    n = rr.shape[1]
    free = [j for j in range(n) if j not in pivots]
    out = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        out[i, f] = 1
        for r, p in enumerate(pivots):
            out[i, p] = rr[r, f]
    return out


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for b in _u8(bits).reshape(-1):
        value = (value << 1) | int(b)
    return value


def int_to_bits(value: int, n: int) -> np.ndarray:
    if value < 0 or value >= (1 << n):
        raise ValueError("integer outside requested bit width")
    return np.fromiter(((value >> i) & 1 for i in range(n - 1, -1, -1)), dtype=np.uint8, count=n)


@dataclass(frozen=True)
class LinearCode:
    name: str
    G: np.ndarray
    H: np.ndarray
    family: str

    def __post_init__(self) -> None:
        g = _u8(self.G)
        h = _u8(self.H)
        if g.ndim != 2 or h.ndim != 2 or g.shape[1] != h.shape[1]:
            raise ValueError("incompatible generator/parity-check matrices")
        if rank(g) != g.shape[0]:
            raise ValueError("generator matrix lacks full row rank")
        if rank(h) != h.shape[0]:
            raise ValueError("parity-check matrix lacks full row rank")
        if np.any((g @ h.T) & 1):
            raise ValueError("G H^T is not zero over GF(2)")
        object.__setattr__(self, "G", g)
        object.__setattr__(self, "H", h)

    @property
    def k(self) -> int:
        return int(self.G.shape[0])

    @property
    def n(self) -> int:
        return int(self.G.shape[1])

    @property
    def rate(self) -> float:
        return self.k / self.n

    def encode(self, message: np.ndarray) -> np.ndarray:
        u = _u8(message).reshape(-1)
        if u.size != self.k:
            raise ValueError(f"expected {self.k} message bits")
        return (u @ self.G) & 1

    def syndrome(self, word: np.ndarray) -> np.ndarray:
        x = _u8(word).reshape(-1)
        if x.size != self.n:
            raise ValueError(f"expected {self.n} code bits")
        return (self.H @ x) & 1

    def contains(self, word: np.ndarray) -> bool:
        return not bool(np.any(self.syndrome(word)))

    def enumerate_codewords(self, max_k: int = 22) -> np.ndarray:
        if self.k > max_k:
            raise ValueError(f"enumeration disabled for k={self.k} > {max_k}")
        messages = np.arange(1 << self.k, dtype=np.uint64)
        shifts = np.arange(self.k - 1, -1, -1, dtype=np.uint64)
        u = ((messages[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
        return (u @ self.G) & 1


def systematic_random_code(
    n: int,
    k: int,
    rng: np.random.Generator,
    family: str = "dense",
    density: float | None = None,
) -> LinearCode:
    if not (1 <= k < n):
        raise ValueError("require 1 <= k < n")
    pcols = n - k
    if density is None:
        density = 0.5 if family == "dense" else min(0.22, max(0.06, 3.0 / max(k, 1)))
    p = (rng.random((k, pcols)) < density).astype(np.uint8)
    # Avoid structurally empty parity columns and information rows.
    for j in range(pcols):
        if not p[:, j].any():
            p[int(rng.integers(k)), j] = 1
    for i in range(k):
        if not p[i].any():
            p[i, int(rng.integers(pcols))] = 1
    g = np.concatenate([np.eye(k, dtype=np.uint8), p], axis=1)
    h = np.concatenate([p.T, np.eye(pcols, dtype=np.uint8)], axis=1)
    return LinearCode(name=f"random_{family}_{n}_{k}", G=g, H=h, family=family)


def hamming_code(m: int = 4) -> LinearCode:
    """Binary Hamming [2^m-1, 2^m-1-m] code."""
    if m < 2:
        raise ValueError("m must be at least 2")
    n = (1 << m) - 1
    cols = [int_to_bits(i, m) for i in range(1, n + 1)]
    h = np.stack(cols, axis=1)
    g = nullspace(h)
    return LinearCode(name=f"hamming_{n}_{n-m}", G=g, H=h, family="hamming")


def most_reliable_basis_generator(g: np.ndarray, reliabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return row-equivalent G whose selected most-reliable columns form I.

    Returns `(G_mrb, info_positions)`.  Greedy matroid selection produces a
    maximum-weight basis because columns form a linear matroid.
    """
    g = _u8(g)
    rel = np.asarray(reliabilities, dtype=float).reshape(-1)
    k, n = g.shape
    if rel.size != n:
        raise ValueError("reliability length mismatch")
    order = np.argsort(-rel, kind="stable")
    selected: list[int] = []
    current = np.zeros((k, 0), dtype=np.uint8)
    current_rank = 0
    for col in order:
        trial = np.concatenate([current, g[:, [int(col)]]], axis=1)
        r = rank(trial)
        if r > current_rank:
            selected.append(int(col))
            current = trial
            current_rank = r
            if len(selected) == k:
                break
    if len(selected) != k:
        raise ValueError("generator does not contain an information basis")
    gi = g[:, selected]
    u = inverse(gi)
    gmrb = (u @ g) & 1
    if not np.array_equal(gmrb[:, selected], np.eye(k, dtype=np.uint8)):
        raise AssertionError("MRB row transformation failed")
    return gmrb, np.asarray(selected, dtype=int)
