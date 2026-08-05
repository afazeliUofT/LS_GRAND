from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .channel import StateHypothesis, affine_transform_bits, fixed_slip_path
from .gf2 import LinearCode
from .search import word_key


def log2_binomial_prefix(n: int, max_weight: int) -> float:
    """log2(sum_{i=0}^{max_weight} binom(n,i)) without overflow."""
    if max_weight < 0:
        return float("-inf")
    max_weight = min(max_weight, n)
    logs = np.array(
        [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(max_weight + 1)],
        dtype=float,
    )
    m = float(logs.max())
    return float((m + np.log(np.exp(logs - m).sum())) / np.log(2.0))


def rank_separation_record(bits: np.ndarray, true_path: np.ndarray, latent_family_size: int) -> dict:
    transformed = affine_transform_bits(bits, true_path)
    d = int(np.count_nonzero(transformed ^ np.asarray(bits, dtype=np.uint8)))
    n = int(np.asarray(bits).size)
    return {
        "n": n,
        "symbols": n // 2,
        "apparent_hamming_weight": d,
        "apparent_error_fraction": d / n,
        "hard_grand_log2_rank_lower_bound": log2_binomial_prefix(n, d - 1),
        "latent_family_size": int(latent_family_size),
        "latent_family_log2_size": float(np.log2(max(latent_family_size, 1))),
        "log2_coordinate_separation_lower_bound": log2_binomial_prefix(n, d - 1)
        - float(np.log2(max(latent_family_size, 1))),
    }


def exact_orbit_collision_records(
    code: LinearCode,
    paths: Sequence[tuple[str, np.ndarray]],
    max_k: int = 20,
) -> list[dict]:
    cws = code.enumerate_codewords(max_k=max_k)
    code_set = {word_key(c) for c in cws}
    records: list[dict] = []
    for label, path in paths:
        hits = 0
        fixed = 0
        image_counts: dict[bytes, int] = {}
        for c in cws:
            t = affine_transform_bits(c, path)
            k = word_key(t)
            if k in code_set:
                hits += 1
            if np.array_equal(t, c):
                fixed += 1
            image_counts[k] = image_counts.get(k, 0) + 1
        records.append(
            {
                "code": code.name,
                "family": code.family,
                "n": code.n,
                "k": code.k,
                "rate": code.rate,
                "transform": label,
                "colliding_codewords": hits,
                "collision_fraction": hits / len(cws),
                "fixed_codewords": fixed,
                "fixed_fraction": fixed / len(cws),
                "max_image_multiplicity": max(image_counts.values()),
            }
        )
    return records


def canonical_collision_paths(n_bits: int) -> list[tuple[str, np.ndarray]]:
    if n_bits % 2:
        raise ValueError("QPSK collision paths require even n")
    m = n_bits // 2
    paths: list[tuple[str, np.ndarray]] = []
    for d in (1, 2, 3):
        paths.append((f"global_d{d}", np.full(m, d, dtype=np.int8)))
    if m >= 2:
        for frac, name in ((0.25, "early"), (0.5, "middle"), (0.75, "late")):
            tau = min(max(1, int(round(frac * m))), m - 1)
            for d in (1, 2, 3):
                paths.append((f"{name}_tau{tau}_d{d}", fixed_slip_path(m, tau, d)))
    return paths
