from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np

from .channel import StateHypothesis, stream_parameters

NEG_INF = float("-inf")


def logsumexp(values: Iterable[float]) -> float:
    vals = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return NEG_INF
    m = float(vals.max())
    return m + float(np.log(np.exp(vals - m).sum()))


def word_key(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8).reshape(-1), bitorder="big").tobytes()


class SubsetSumEnumerator:
    """Exact lazy enumeration of subsets by nondecreasing sum of nonnegative costs."""

    def __init__(self, costs: np.ndarray):
        c = np.asarray(costs, dtype=float).reshape(-1)
        if np.any(~np.isfinite(c)) or np.any(c < 0):
            raise ValueError("costs must be finite and nonnegative")
        self.original_order = np.argsort(c, kind="stable")
        self.costs = c[self.original_order]
        self.heap: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]

    def pop(self) -> tuple[float, np.ndarray]:
        if not self.heap:
            raise StopIteration
        cost, subset = heapq.heappop(self.heap)
        n = self.costs.size
        if not subset:
            if n:
                heapq.heappush(self.heap, (float(self.costs[0]), (0,)))
        else:
            j = subset[-1]
            if j + 1 < n:
                heapq.heappush(self.heap, (cost + float(self.costs[j + 1]), subset + (j + 1,)))
                replacement = subset[:-1] + (j + 1,)
                heapq.heappush(
                    self.heap,
                    (cost - float(self.costs[j]) + float(self.costs[j + 1]), replacement),
                )
        if subset:
            idx = np.fromiter(subset, dtype=int, count=len(subset))
            original = self.original_order[idx]
        else:
            original = np.empty(0, dtype=int)
        return float(cost), original


class StateQueue:
    def __init__(self, y: np.ndarray, state: StateHypothesis, n0: float):
        self.state = state
        self.hard, self.costs, self.base_log_weight = stream_parameters(y, state, n0)
        self.enumerator = SubsetSumEnumerator(self.costs)
        self.current: tuple[float, np.ndarray] | None = None
        self.popped = 0
        self._advance()

    def _advance(self) -> None:
        try:
            self.current = self.enumerator.pop()
        except StopIteration:
            self.current = None

    @property
    def next_log_weight(self) -> float:
        if self.current is None:
            return NEG_INF
        return float(self.base_log_weight - self.current[0])

    def pop_word(self) -> tuple[np.ndarray, float, int]:
        if self.current is None:
            raise StopIteration
        cost, flips = self.current
        word = self.hard.copy()
        if flips.size:
            word[flips] ^= 1
        logw = float(self.base_log_weight - cost)
        rank = self.popped
        self.popped += 1
        self._advance()
        return word, logw, rank
