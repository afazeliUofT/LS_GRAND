from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.stats import norm


def wilson_interval(errors: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = errors / trials
    den = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / den
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / den
    return max(0.0, center - half), min(1.0, center + half)


def rule_of_three_upper(zero_events_trials: int, confidence: float = 0.95) -> float:
    if zero_events_trials <= 0:
        return float("nan")
    return 1.0 - (1.0 - confidence) ** (1.0 / zero_events_trials)


def percentile(values: Iterable[float], q: float) -> float:
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.quantile(a, q)) if a.size else float("nan")


def paired_bootstrap_median_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    rng: np.random.Generator,
    resamples: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float, float, int]:
    a = np.asarray(numerator, dtype=float)
    b = np.asarray(denominator, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (b > 0) & (a >= 0)
    a = a[mask]
    b = b[mask]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    ratios = a / b
    estimate = float(np.median(ratios))
    if a.size == 1:
        return estimate, estimate, estimate, 1
    boots = np.empty(resamples, dtype=float)
    for i in range(resamples):
        idx = rng.integers(0, a.size, size=a.size)
        boots[i] = np.median(ratios[idx])
    alpha = (1.0 - confidence) / 2.0
    return estimate, float(np.quantile(boots, alpha)), float(np.quantile(boots, 1.0 - alpha)), int(a.size)


def paired_error_difference_interval(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    rng: np.random.Generator,
    resamples: int = 4000,
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
    boots = np.empty(resamples, dtype=float)
    for i in range(resamples):
        idx = rng.integers(0, d.size, size=d.size)
        boots[i] = d[idx].mean()
    alpha = (1.0 - confidence) / 2.0
    return estimate, float(np.quantile(boots, alpha)), float(np.quantile(boots, 1.0 - alpha)), int(d.size)
