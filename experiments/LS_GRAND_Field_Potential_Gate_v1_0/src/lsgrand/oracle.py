from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .channel import StateHypothesis, qpsk_loglikelihood
from .gf2 import LinearCode, bits_to_int
from .search import NEG_INF, logsumexp, word_key


@dataclass
class OracleResult:
    marginal_winner_bits: list[int]
    marginal_winner_int: int
    marginal_logscore: float
    marginal_second_logscore: float
    marginal_margin_log: float
    marginal_tie: bool
    joint_winner_bits: list[int]
    joint_winner_int: int
    joint_logscore: float
    marginal_scores: dict[str, float]
    joint_scores: dict[str, float]
    state_codeword_likelihoods: int

    def to_dict(self) -> dict:
        return asdict(self)


def exhaustive_oracle(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    tie_tolerance: float = 1e-10,
    max_k: int = 20,
) -> OracleResult:
    codewords = code.enumerate_codewords(max_k=max_k)
    marginal: list[float] = []
    joint: list[float] = []
    for c in codewords:
        vals = [s.log_prior + qpsk_loglikelihood(y, c, s.path, n0) for s in states]
        marginal.append(logsumexp(vals))
        joint.append(max(vals) if vals else NEG_INF)
    marg = np.asarray(marginal, dtype=float)
    joi = np.asarray(joint, dtype=float)
    order = np.argsort(-marg, kind="stable")
    best = int(order[0])
    second = int(order[1]) if order.size > 1 else best
    jbest = int(np.argmax(joi))
    margin = float(marg[best] - marg[second]) if order.size > 1 else float("inf")
    ms = {word_key(c).hex(): float(v) for c, v in zip(codewords, marg)}
    js = {word_key(c).hex(): float(v) for c, v in zip(codewords, joi)}
    return OracleResult(
        marginal_winner_bits=[int(x) for x in codewords[best]],
        marginal_winner_int=bits_to_int(codewords[best]),
        marginal_logscore=float(marg[best]),
        marginal_second_logscore=float(marg[second]),
        marginal_margin_log=margin,
        marginal_tie=bool(abs(margin) <= tie_tolerance),
        joint_winner_bits=[int(x) for x in codewords[jbest]],
        joint_winner_int=bits_to_int(codewords[jbest]),
        joint_logscore=float(joi[jbest]),
        marginal_scores=ms,
        joint_scores=js,
        state_codeword_likelihoods=int(codewords.shape[0] * len(states)),
    )


def direct_probability_scores(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    max_k: int = 16,
) -> dict[int, float]:
    """Tiny-instance direct-domain check after a global log shift."""
    cws = code.enumerate_codewords(max_k=max_k)
    all_logs = []
    by_word: list[list[float]] = []
    for c in cws:
        vals = [s.log_prior + qpsk_loglikelihood(y, c, s.path, n0) for s in states]
        by_word.append(vals)
        all_logs.extend(vals)
    shift = max(all_logs)
    return {
        bits_to_int(c): float(sum(np.exp(v - shift) for v in vals))
        for c, vals in zip(cws, by_word)
    }
