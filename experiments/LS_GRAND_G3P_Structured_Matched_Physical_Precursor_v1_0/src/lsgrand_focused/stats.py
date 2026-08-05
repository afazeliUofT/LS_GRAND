from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.stats import binomtest, norm


def wilson_interval(errors: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = errors / trials
    den = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / den
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / den
    return max(0.0, center - half), min(1.0, center + half)


def paired_error_difference_interval(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    rng: np.random.Generator,
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float, float, int]:
    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    estimate = float(d.mean())
    if d.size == 1:
        return estimate, estimate, estimate, 1
    boots = np.empty(int(resamples), dtype=float)
    for i in range(boots.size):
        idx = rng.integers(0, d.size, size=d.size)
        boots[i] = d[idx].mean()
    alpha = (1.0 - confidence) / 2.0
    return estimate, float(np.quantile(boots, alpha)), float(np.quantile(boots, 1.0 - alpha)), int(d.size)


def paired_bootstrap_median_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    rng: np.random.Generator,
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float, float, int]:
    a = np.asarray(numerator, dtype=float)
    b = np.asarray(denominator, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (a >= 0) & (b > 0)
    ratios = a[mask] / b[mask]
    if ratios.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    estimate = float(np.median(ratios))
    if ratios.size == 1:
        return estimate, estimate, estimate, 1
    boots = np.empty(int(resamples), dtype=float)
    for i in range(boots.size):
        boots[i] = np.median(ratios[rng.integers(0, ratios.size, size=ratios.size)])
    alpha = (1.0 - confidence) / 2.0
    return estimate, float(np.quantile(boots, alpha)), float(np.quantile(boots, 1.0 - alpha)), int(ratios.size)


def mcnemar_one_sided(errors_a: Iterable[bool], errors_b: Iterable[bool]) -> dict:
    a = np.asarray(list(errors_a), dtype=bool)
    b = np.asarray(list(errors_b), dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired vectors must have equal shape")
    a_only = int(np.count_nonzero(a & ~b))
    b_only = int(np.count_nonzero(~a & b))
    discordant = a_only + b_only
    p = 1.0 if discordant == 0 else float(binomtest(a_only, discordant, 0.5, alternative="less").pvalue)
    return {"a_only_errors": a_only, "b_only_errors": b_only, "discordant": discordant, "p_a_better": p}


def safe_quantile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v = v[mask]
    w = w[mask]
    if v.size == 0:
        return float("nan")
    order = np.argsort(v, kind="stable")
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return float(v[min(int(np.searchsorted(cdf, float(q), side="left")), v.size - 1)])
