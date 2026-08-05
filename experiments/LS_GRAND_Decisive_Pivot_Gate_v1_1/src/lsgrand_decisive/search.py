from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np

from .channel import StateHypothesis, qpsk_loglikelihood, stream_parameters
from .gf2 import LinearCode, bits_to_int, most_reliable_basis_generator


NEG_INF = float("-inf")


def logsumexp(values: Iterable[float]) -> float:
    vals = np.fromiter((float(v) for v in values if np.isfinite(v)), dtype=float)
    if vals.size == 0:
        return NEG_INF
    m = float(vals.max())
    return m + float(np.log(np.exp(vals - m).sum()))


def logaddexp(a: float, b: float) -> float:
    return float(np.logaddexp(a, b))


def word_key(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8).reshape(-1), bitorder="big").tobytes()


class SubsetSumEnumerator:
    """Enumerate all subsets by nondecreasing sum of nonnegative costs.

    Costs are internally sorted.  The two-child heap construction is exact,
    duplicate-free, and lazy.  It is suitable for reliability-ordered GRAND.
    """

    def __init__(self, costs: np.ndarray):
        c = np.asarray(costs, dtype=float).reshape(-1)
        if np.any(~np.isfinite(c)) or np.any(c < 0):
            raise ValueError("subset costs must be finite and nonnegative")
        self.original_order = np.argsort(c, kind="stable")
        self.costs = c[self.original_order]
        self.heap: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
        self.emitted = 0

    def __len__(self) -> int:
        return 1 << self.costs.size

    def pop(self) -> tuple[float, np.ndarray]:
        if not self.heap:
            raise StopIteration
        cost, subset = heapq.heappop(self.heap)
        self.emitted += 1
        n = self.costs.size
        if not subset:
            if n:
                heapq.heappush(self.heap, (float(self.costs[0]), (0,)))
        else:
            j = subset[-1]
            if j + 1 < n:
                # Add the next item while retaining the current maximum.
                s_add = subset + (j + 1,)
                heapq.heappush(self.heap, (cost + float(self.costs[j + 1]), s_add))
                # Replace the current maximum by the next item.
                s_rep = subset[:-1] + (j + 1,)
                heapq.heappush(
                    self.heap,
                    (cost - float(self.costs[j]) + float(self.costs[j + 1]), s_rep),
                )
        original = self.original_order[np.fromiter(subset, dtype=int, count=len(subset))] if subset else np.empty(0, dtype=int)
        return float(cost), original


class StateQueue:
    def __init__(self, y: np.ndarray, state: StateHypothesis, n0: float):
        self.state = state
        self.hard, self.costs, self.base_log_weight = stream_parameters(y, state, n0)
        self.enumerator = SubsetSumEnumerator(self.costs)
        self._current: tuple[float, np.ndarray] | None = None
        self.popped = 0
        self._advance()

    def _advance(self) -> None:
        try:
            self._current = self.enumerator.pop()
        except StopIteration:
            self._current = None

    @property
    def next_log_weight(self) -> float:
        if self._current is None:
            return NEG_INF
        return self.base_log_weight - self._current[0]

    def pop_word(self) -> tuple[np.ndarray, float, int]:
        if self._current is None:
            raise StopIteration
        cost, flips = self._current
        word = self.hard.copy()
        if flips.size:
            word[flips] ^= 1
        logw = self.base_log_weight - cost
        rank = self.popped
        self.popped += 1
        self._advance()
        return word, float(logw), rank


@dataclass
class DecodeResult:
    decoder: str
    decoded_bits: list[int] | None
    decoded_int: int | None
    success: bool
    certified: bool
    certificate_margin_log: float | None
    stop_reason: str
    membership_queries: int
    residual_patterns_generated: int
    latent_hypotheses_available: int
    latent_queues_touched: int
    valid_witnesses: int
    unique_codewords_seen: int
    complete_marginal_scores: int
    state_codeword_likelihoods: int
    cap_hit: bool
    wall_seconds: float
    winner_partial_logscore: float | None
    oracle_first_seen_query: int | None = None
    certificate_query_overhead: float | None = None
    queue_touch_fraction: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _make_result(
    decoder: str,
    bits: np.ndarray | None,
    certified: bool,
    margin: float | None,
    reason: str,
    queries: int,
    states: int,
    touched: set[int],
    valid: int,
    scores: dict[bytes, float],
    full_scores: int,
    state_evals: int,
    cap_hit: bool,
    elapsed: float,
    winner_log: float | None,
    oracle_first: int | None = None,
) -> DecodeResult:
    overhead = None
    if oracle_first is not None and oracle_first > 0:
        overhead = queries / oracle_first
    return DecodeResult(
        decoder=decoder,
        decoded_bits=None if bits is None else [int(x) for x in bits],
        decoded_int=None if bits is None else bits_to_int(bits),
        success=bits is not None,
        certified=certified,
        certificate_margin_log=margin,
        stop_reason=reason,
        membership_queries=queries,
        residual_patterns_generated=queries,
        latent_hypotheses_available=states,
        latent_queues_touched=len(touched),
        valid_witnesses=valid,
        unique_codewords_seen=len(scores),
        complete_marginal_scores=full_scores,
        state_codeword_likelihoods=state_evals,
        cap_hit=cap_hit,
        wall_seconds=elapsed,
        winner_partial_logscore=winner_log,
        oracle_first_seen_query=oracle_first,
        certificate_query_overhead=overhead,
        queue_touch_fraction=(len(touched) / states if states else None),
    )


def certified_lsgrand(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    query_cap: int,
    certificate_interval: int = 1,
    strict_tolerance: float = 1e-11,
    oracle_word: np.ndarray | None = None,
) -> DecodeResult:
    """Exact finite-state fiber decoder with a per-codeword head certificate."""
    start = time.perf_counter()
    queues = [StateQueue(y, s, n0) for s in states]
    heap: list[tuple[float, int, int]] = []
    serial = 0
    for i, q in enumerate(queues):
        if np.isfinite(q.next_log_weight):
            heapq.heappush(heap, (-q.next_log_weight, serial, i))
            serial += 1
    scores: dict[bytes, float] = {}
    words: dict[bytes, np.ndarray] = {}
    seen_states: dict[bytes, set[int]] = {}
    touched: set[int] = set()
    valid = 0
    queries = 0
    oracle_key = None if oracle_word is None else word_key(oracle_word)
    oracle_first: int | None = None
    first_seen_query: dict[bytes, int] = {}
    last_margin: float | None = None

    def certificate() -> tuple[bytes | None, float | None]:
        if not scores:
            return None, None
        next_heads = np.array([q.next_log_weight for q in queues], dtype=float)
        unseen_bound = logsumexp(next_heads)
        best_key = max(scores, key=scores.get)
        best_lower = scores[best_key]
        rival_upper = unseen_bound
        for key, partial in scores.items():
            if key == best_key:
                continue
            seen = seen_states[key]
            tail = logsumexp(next_heads[i] for i in range(len(queues)) if i not in seen)
            ub = logaddexp(partial, tail)
            rival_upper = max(rival_upper, ub)
        margin = best_lower - rival_upper
        if margin > strict_tolerance:
            return best_key, margin
        return None, margin

    while heap and queries < query_cap:
        _, _, si = heapq.heappop(heap)
        q = queues[si]
        word, logw, _ = q.pop_word()
        touched.add(si)
        queries += 1
        if np.isfinite(q.next_log_weight):
            heapq.heappush(heap, (-q.next_log_weight, serial, si))
            serial += 1
        is_valid = code.contains(word)
        if is_valid:
            valid += 1
            key = word_key(word)
            if key not in scores:
                scores[key] = logw
                words[key] = word.copy()
                seen_states[key] = {si}
                first_seen_query[key] = queries
            else:
                if si in seen_states[key]:
                    raise AssertionError("fixed-state enumeration emitted a duplicate codeword")
                scores[key] = logaddexp(scores[key], logw)
                seen_states[key].add(si)
            if oracle_key is not None and key == oracle_key and oracle_first is None:
                oracle_first = queries
        do_check = bool(scores) and (valid > 0) and (
            certificate_interval <= 1 or queries % certificate_interval == 0 or is_valid
        )
        if do_check:
            winner, last_margin = certificate()
            if winner is not None:
                elapsed = time.perf_counter() - start
                return _make_result(
                    "lsgrand_certified",
                    words[winner],
                    True,
                    last_margin,
                    "strict_unseen_codeword_certificate",
                    queries,
                    len(states),
                    touched,
                    valid,
                    scores,
                    0,
                    queries,
                    False,
                    elapsed,
                    scores[winner],
                    oracle_first if oracle_first is not None else first_seen_query.get(winner),
                )

    elapsed = time.perf_counter() - start
    if not heap and scores:
        winner = max(scores, key=scores.get)
        return _make_result(
            "lsgrand_certified",
            words[winner],
            True,
            math.inf,
            "all_finite_queues_exhausted",
            queries,
            len(states),
            touched,
            valid,
            scores,
            0,
            queries,
            False,
            elapsed,
            scores[winner],
            oracle_first if oracle_first is not None else first_seen_query.get(winner),
        )
    best = max(scores, key=scores.get) if scores else None
    return _make_result(
        "lsgrand_certified",
        None if best is None else words[best],
        False,
        last_margin,
        "query_cap_before_certificate",
        queries,
        len(states),
        touched,
        valid,
        scores,
        0,
        queries,
        True,
        elapsed,
        None if best is None else scores[best],
        oracle_first if oracle_first is not None else (None if best is None else first_seen_query.get(best)),
    )


def first_valid_latent(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    query_cap: int,
) -> DecodeResult:
    start = time.perf_counter()
    queues = [StateQueue(y, s, n0) for s in states]
    heap: list[tuple[float, int, int]] = []
    serial = 0
    for i, q in enumerate(queues):
        heapq.heappush(heap, (-q.next_log_weight, serial, i))
        serial += 1
    touched: set[int] = set()
    queries = 0
    while heap and queries < query_cap:
        _, _, si = heapq.heappop(heap)
        word, logw, _ = queues[si].pop_word()
        touched.add(si)
        queries += 1
        if np.isfinite(queues[si].next_log_weight):
            heapq.heappush(heap, (-queues[si].next_log_weight, serial, si))
            serial += 1
        if code.contains(word):
            elapsed = time.perf_counter() - start
            key = word_key(word)
            return _make_result(
                "lsgrand_first_valid", word, False, None, "first_valid_witness", queries,
                len(states), touched, 1, {key: logw}, 0, queries, False, elapsed, logw
            )
    elapsed = time.perf_counter() - start
    return _make_result(
        "lsgrand_first_valid", None, False, None, "query_cap_or_exhaustion", queries,
        len(states), touched, 0, {}, 0, queries, queries >= query_cap, elapsed, None
    )


def per_state_grand_sweep(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    per_state_cap: int,
    global_cap: int,
) -> DecodeResult:
    start = time.perf_counter()
    best_word: np.ndarray | None = None
    best_logw = NEG_INF
    touched: set[int] = set()
    total = 0
    valid = 0
    score_map: dict[bytes, float] = {}
    cap_hit = False
    for si, state in enumerate(states):
        if total >= global_cap:
            cap_hit = True
            break
        q = StateQueue(y, state, n0)
        touched.add(si)
        found = False
        for _ in range(per_state_cap):
            if total >= global_cap:
                cap_hit = True
                break
            try:
                word, logw, _ = q.pop_word()
            except StopIteration:
                break
            total += 1
            if code.contains(word):
                valid += 1
                found = True
                key = word_key(word)
                score_map[key] = max(score_map.get(key, NEG_INF), logw)
                if logw > best_logw:
                    best_logw = logw
                    best_word = word.copy()
                break
        if not found and q._current is not None:
            cap_hit = True
    elapsed = time.perf_counter() - start
    return _make_result(
        "per_state_grand_sweep", best_word, False, None,
        "best_fixed_state_codeword" if best_word is not None else "no_valid_word_within_cap",
        total, len(states), touched, valid, score_map, 0, total, cap_hit, elapsed,
        None if best_word is None else best_logw,
    )


def plain_grand(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    query_cap: int,
) -> DecodeResult:
    state = StateHypothesis(
        index=0,
        path=np.zeros(len(y), dtype=np.int8),
        log_prior=0.0,
        label="assumed_no_slip",
        slip_count=0,
        slip_locations=(),
        increments=(),
    )
    result = first_valid_latent(y, n0, code, [state], query_cap)
    result.decoder = "plain_reliability_grand"
    return result


def complete_marginal_score(
    y: np.ndarray,
    n0: float,
    word: np.ndarray,
    states: Sequence[StateHypothesis],
) -> tuple[float, int]:
    vals = [s.log_prior + qpsk_loglikelihood(y, word, s.path, n0) for s in states]
    return logsumexp(vals), len(vals)


def latent_osd(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    order: int = 2,
    pool_size: int = 16,
    state_limit: int = 24,
    candidate_cap: int = 100_000,
) -> DecodeResult:
    """State-aware OSD list with complete marginal rescoring.

    Every state first contributes an OSD-0 codeword.  States are selected by the
    fixed-state likelihood of that codeword, after which order-d patterns over
    the least reliable MRB information positions are generated.  The union is
    deduplicated and every retained codeword is scored across all modeled states.
    """
    start = time.perf_counter()
    state_data: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    generated = 0
    for si, state in enumerate(states):
        hard, rel, _ = stream_parameters(y, state, n0)
        gmrb, info = most_reliable_basis_generator(code.G, rel)
        u0 = hard[info].copy()
        c0 = (u0 @ gmrb) & 1
        fixed = state.log_prior + qpsk_loglikelihood(y, c0, state.path, n0)
        state_data.append((fixed, si, hard, rel, gmrb, info))
        generated += 1
    state_data.sort(key=lambda x: x[0], reverse=True)
    selected = state_data[: min(state_limit, len(state_data))]
    candidates: dict[bytes, np.ndarray] = {}
    touched: set[int] = set()
    cap_hit = False
    for _, si, hard, rel, gmrb, info in selected:
        touched.add(si)
        u0 = hard[info].copy()
        info_rel = rel[info]
        pool_local = np.argsort(info_rel, kind="stable")[: min(pool_size, code.k)]
        for d in range(order + 1):
            for combo in itertools.combinations(pool_local.tolist(), d):
                u = u0.copy()
                if combo:
                    u[np.fromiter(combo, dtype=int, count=len(combo))] ^= 1
                word = (u @ gmrb) & 1
                candidates.setdefault(word_key(word), word)
                generated += 1
                if len(candidates) >= candidate_cap:
                    cap_hit = True
                    break
            if cap_hit:
                break
        if cap_hit:
            break
    best_word: np.ndarray | None = None
    best_score = NEG_INF
    scores: dict[bytes, float] = {}
    state_evals = 0
    for key, word in candidates.items():
        score, evals = complete_marginal_score(y, n0, word, states)
        state_evals += evals
        scores[key] = score
        if score > best_score:
            best_score = score
            best_word = word.copy()
    elapsed = time.perf_counter() - start
    return _make_result(
        f"latent_osd_order{order}", best_word, False, None,
        "best_complete_marginal_score_in_osd_list" if best_word is not None else "empty_osd_list",
        0, len(states), touched, generated, scores, len(scores), state_evals, cap_hit,
        elapsed, None if best_word is None else best_score,
    )
