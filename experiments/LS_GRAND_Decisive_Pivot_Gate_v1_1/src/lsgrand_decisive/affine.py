from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .gf2 import LinearCode, rank


def qpsk_affine_pair(direction: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the GF(2) affine label map induced by a C4 rotation.

    Bits are column vectors ``[b_I,b_Q]^T`` for sign-labelled QPSK.  The map is
    ``b' = A b + d`` over GF(2).
    """
    d = int(direction) % 4
    if d == 0:
        return np.eye(2, dtype=np.uint8), np.zeros(2, dtype=np.uint8)
    if d == 1:
        return np.array([[0, 1], [1, 0]], dtype=np.uint8), np.array([1, 0], dtype=np.uint8)
    if d == 2:
        return np.eye(2, dtype=np.uint8), np.ones(2, dtype=np.uint8)
    return np.array([[0, 1], [1, 0]], dtype=np.uint8), np.array([0, 1], dtype=np.uint8)


def affine_map_for_path(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(path, dtype=np.int8).reshape(-1) % 4
    n = 2 * p.size
    a = np.zeros((n, n), dtype=np.uint8)
    b = np.zeros(n, dtype=np.uint8)
    for t, direction in enumerate(p):
        aa, bb = qpsk_affine_pair(int(direction))
        sl = slice(2 * t, 2 * t + 2)
        a[sl, sl] = aa
        b[sl] = bb
    return a, b


def apply_affine_bits(bits: np.ndarray, path: np.ndarray) -> np.ndarray:
    x = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    a, b = affine_map_for_path(path)
    if a.shape[1] != x.size:
        raise ValueError("path/word length mismatch")
    return ((a @ x) ^ b) & 1


def transformed_generator_rows(code: LinearCode, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rows spanning the linear part of T(C), and affine offset b."""
    a, b = affine_map_for_path(path)
    return ((a @ code.G.T) & 1).T, b


def gf2_consistent(a: np.ndarray, b: np.ndarray) -> bool:
    aa = np.asarray(a, dtype=np.uint8) & 1
    bb = np.asarray(b, dtype=np.uint8).reshape(-1, 1) & 1
    return rank(np.concatenate([aa, bb], axis=1)) == rank(aa)


@dataclass(frozen=True)
class OrbitIntersection:
    collision_fraction: float
    collision_log2_fraction: float
    system_rank: int
    consistent: bool
    colliding_codewords_log2: float | None


def linear_code_affine_collision(code: LinearCode, path: np.ndarray) -> OrbitIntersection:
    """Compute |C ∩ T^{-1}(C)|/|C| exactly without codeword enumeration.

    For ``c=G^T u`` (column convention), the requirement ``T(c)∈C`` is
    ``H A G^T u = H b``.  A consistent rank-r system has ``2^(k-r)``
    solutions, hence collision fraction ``2^-r``.
    """
    a, b = affine_map_for_path(path)
    m = (code.H @ a @ code.G.T) & 1
    rhs = (code.H @ b) & 1
    r = rank(m)
    consistent = gf2_consistent(m, rhs)
    if not consistent:
        return OrbitIntersection(0.0, float("-inf"), r, False, None)
    frac = float(2.0 ** (-r))
    return OrbitIntersection(frac, float(-r), r, True, float(code.k - r))


def all_one_slip_paths(symbols: int, directions: Iterable[int] = (1, 2, 3)) -> list[tuple[str, np.ndarray]]:
    if symbols < 2:
        return []
    out: list[tuple[str, np.ndarray]] = []
    for tau in range(1, symbols):
        for direction in directions:
            d = int(direction) % 4
            if d == 0:
                continue
            path = np.zeros(symbols, dtype=np.int8)
            path[tau:] = d
            out.append((f"tau{tau}_d{d}", path))
    return out


def global_rotation_paths(symbols: int) -> list[tuple[str, np.ndarray]]:
    return [(f"global_d{d}", np.full(symbols, d, dtype=np.int8)) for d in (1, 2, 3)]
