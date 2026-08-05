from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import fixed_slip_path
from .gf2 import LinearCode, permute_code, rank


def qpsk_affine_pair(direction: int) -> tuple[np.ndarray, np.ndarray]:
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


def gf2_consistent(a: np.ndarray, b: np.ndarray) -> bool:
    aa = np.asarray(a, dtype=np.uint8) & 1
    bb = (np.asarray(b, dtype=np.uint8).reshape(-1, 1) & 1)
    return rank(np.concatenate([aa, bb], axis=1)) == rank(aa)


@dataclass(frozen=True)
class OrbitIntersection:
    collision_fraction: float
    rank: int
    consistent: bool


def linear_code_affine_collision(code: LinearCode, path: np.ndarray) -> OrbitIntersection:
    a, b = affine_map_for_path(path)
    system = (code.H @ a @ code.G.T) & 1
    rhs = (code.H @ b) & 1
    r = rank(system)
    consistent = gf2_consistent(system, rhs)
    return OrbitIntersection(0.0 if not consistent else float(2.0 ** (-r)), r, consistent)


def one_slip_paths(symbols: int, directions: tuple[int, ...] = (1, 3)) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for tau in range(1, symbols):
        for direction in directions:
            d = int(direction) % 4
            if d:
                out.append((f"tau{tau}_d{d}", fixed_slip_path(symbols, tau, d)))
    return out


def orbit_summary(code: LinearCode, threshold: float = 1e-3) -> dict:
    records = []
    for label, path in one_slip_paths(code.n // 2, (1, 3)):
        result = linear_code_affine_collision(code, path)
        records.append((label, result.collision_fraction, result.rank, result.consistent))
    fractions = np.asarray([r[1] for r in records], dtype=float)
    max_idx = int(np.argmax(fractions)) if fractions.size else 0
    return {
        "code": code.name,
        "family": code.family,
        "n": code.n,
        "k": code.k,
        "redundancy": code.redundancy,
        "transform_count": len(records),
        "max_collision_fraction": float(fractions.max()) if fractions.size else 0.0,
        "mean_collision_fraction": float(fractions.mean()) if fractions.size else 0.0,
        "union_bound": float(min(1.0, fractions.sum())) if fractions.size else 0.0,
        "dangerous_transform_count": int(np.count_nonzero(fractions > float(threshold))),
        "worst_transform": records[max_idx][0] if records else None,
        "orbit_safe": bool(not fractions.size or fractions.max() <= float(threshold)),
        "records": [
            {"transform": label, "collision_fraction": frac, "system_rank": r, "consistent": cons}
            for label, frac, r, cons in records
        ],
    }


def find_orbit_safe_interleaver(
    code: LinearCode,
    rng: np.random.Generator,
    *,
    threshold: float = 1e-3,
    max_attempts: int = 200,
) -> tuple[LinearCode, dict]:
    baseline = orbit_summary(code, threshold)
    best_code = code
    best = baseline
    if baseline["orbit_safe"]:
        return code, {**baseline, "attempt": 0, "interleaved": False}
    for attempt in range(1, int(max_attempts) + 1):
        permutation = rng.permutation(code.n)
        candidate = permute_code(code, permutation, name_suffix=f"orbit{attempt}")
        summary = orbit_summary(candidate, threshold)
        if summary["max_collision_fraction"] < best["max_collision_fraction"]:
            best_code, best = candidate, summary
        if summary["orbit_safe"]:
            return candidate, {**summary, "attempt": attempt, "interleaved": True}
    return best_code, {**best, "attempt": int(max_attempts), "interleaved": best_code is not code}
