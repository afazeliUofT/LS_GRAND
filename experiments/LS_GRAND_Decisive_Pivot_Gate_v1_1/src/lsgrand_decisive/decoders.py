from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .channel import StateHypothesis, qpsk_loglikelihood, stream_parameters
from .gf2 import LinearCode, bits_to_int, most_reliable_basis_generator
from .search import StateQueue, certified_lsgrand, word_key


@dataclass
class DecoderAuditResult:
    decoder: str
    decoded_bits: list[int] | None
    decoded_int: int | None
    success: bool
    certified: bool
    cap_hit: bool
    stop_reason: str
    components_generated: int
    membership_queries: int
    latent_hypotheses_available: int
    latent_queues_touched: int
    valid_codewords: int
    unique_candidates: int
    complete_marginal_candidates: int
    state_word_metric_evals: int
    bit_metric_accumulations: int
    osd_reprocessings: int
    preprocessing_state_metrics: int
    wall_seconds: float

    @property
    def queue_touch_fraction(self) -> float:
        if self.latent_hypotheses_available <= 0:
            return float("nan")
        return self.latent_queues_touched / self.latent_hypotheses_available

    def to_dict(self) -> dict:
        out = asdict(self)
        out["queue_touch_fraction"] = self.queue_touch_fraction
        return out


def _bits_or_none(bits: np.ndarray | None) -> tuple[list[int] | None, int | None]:
    if bits is None:
        return None, None
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    return [int(x) for x in b], bits_to_int(b)


def _state_metric_tables(
    y: np.ndarray,
    n0: float,
    states: Sequence[StateHypothesis],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hard = []
    costs = []
    base = []
    for state in states:
        h, c, b = stream_parameters(y, state, n0)
        hard.append(h)
        costs.append(c)
        base.append(b)
    return np.asarray(hard, dtype=np.uint8), np.asarray(costs, dtype=float), np.asarray(base, dtype=float)


def batch_marginal_scores(
    y: np.ndarray,
    n0: float,
    candidates: np.ndarray,
    states: Sequence[StateHypothesis],
    *,
    batch_size: int = 512,
    tables: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, int, int, int]:
    """Exact marginal scores using cached fixed-state hard words and flip costs.

    This is algebraically identical to direct QPSK likelihood evaluation but
    avoids recomputing symbol distances for every state-candidate pair.
    Returns scores and transparent operation counts.
    """
    cands = np.asarray(candidates, dtype=np.uint8)
    if cands.ndim != 2:
        raise ValueError("candidates must be a 2-D binary array")
    if tables is None:
        hard, costs, base = _state_metric_tables(y, n0, states)
        pre = len(states)
    else:
        hard, costs, base = tables
        pre = 0
    if cands.shape[1] != hard.shape[1]:
        raise ValueError("candidate/state metric length mismatch")
    out = np.empty(cands.shape[0], dtype=float)
    kstates = hard.shape[0]
    for start in range(0, cands.shape[0], max(1, int(batch_size))):
        block = cands[start : start + batch_size]
        # Shape B x S x n.  Chunking keeps memory bounded.
        diff = np.bitwise_xor(block[:, None, :], hard[None, :, :]).astype(float, copy=False)
        penalties = np.einsum("bsn,sn->bs", diff, costs, optimize=True)
        vals = base[None, :] - penalties
        m = np.max(vals, axis=1)
        out[start : start + block.shape[0]] = m + np.log(np.exp(vals - m[:, None]).sum(axis=1))
    state_evals = int(cands.shape[0] * kstates)
    bit_accums = int(state_evals * cands.shape[1])
    return out, state_evals, bit_accums, pre


def latent_list_decode(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    *,
    list_size: int,
    query_cap: int,
    marginal_rescore: bool = True,
    score_batch_size: int = 512,
) -> DecoderAuditResult:
    """Globally ordered latent search, stopped after L unique valid codewords.

    ``L=1`` is the practical max-witness/first-valid decoder.  For ``L>1``, the
    discovered codewords are completely marginalized across all states and the
    best list member is returned.  This is implementable but approximate because
    unseen codewords are not certified away.
    """
    if list_size < 1:
        raise ValueError("list_size must be positive")
    start = time.perf_counter()
    queues = [StateQueue(y, s, n0) for s in states]
    heap: list[tuple[float, int, int]] = []
    serial = 0
    for si, q in enumerate(queues):
        if np.isfinite(q.next_log_weight):
            heapq.heappush(heap, (-q.next_log_weight, serial, si))
            serial += 1
    touched: set[int] = set()
    membership_cache: dict[bytes, bool] = {}
    valid_words: dict[bytes, np.ndarray] = {}
    valid_witnesses = 0
    components = 0
    membership_queries = 0
    while heap and components < query_cap and len(valid_words) < list_size:
        _, _, si = heapq.heappop(heap)
        word, _, _ = queues[si].pop_word()
        components += 1
        touched.add(si)
        if np.isfinite(queues[si].next_log_weight):
            heapq.heappush(heap, (-queues[si].next_log_weight, serial, si))
            serial += 1
        key = word_key(word)
        is_valid = membership_cache.get(key)
        if is_valid is None:
            is_valid = code.contains(word)
            membership_cache[key] = is_valid
            membership_queries += 1
        if is_valid:
            valid_witnesses += 1
            valid_words.setdefault(key, word.copy())

    cap_hit = components >= query_cap and len(valid_words) < list_size
    winner: np.ndarray | None = None
    full = 0
    state_evals = 0
    bit_accums = 0
    pre = 0
    if valid_words:
        cands = np.stack(list(valid_words.values()))
        if marginal_rescore:
            scores, state_evals, bit_accums, pre = batch_marginal_scores(
                y, n0, cands, states, batch_size=score_batch_size
            )
            winner = cands[int(np.argmax(scores))]
            full = cands.shape[0]
        else:
            # The first inserted valid codeword is the highest individual witness.
            winner = cands[0]
    elapsed = time.perf_counter() - start
    bits, value = _bits_or_none(winner)
    return DecoderAuditResult(
        decoder=(f"ls_list_marginal_L{list_size}" if marginal_rescore else "ls_first_valid"),
        decoded_bits=bits,
        decoded_int=value,
        success=winner is not None,
        certified=False,
        cap_hit=cap_hit,
        stop_reason=("valid_list_complete" if winner is not None and not cap_hit else "query_cap_before_list"),
        components_generated=components,
        membership_queries=membership_queries,
        latent_hypotheses_available=len(states),
        latent_queues_touched=len(touched),
        valid_codewords=valid_witnesses,
        unique_candidates=len(valid_words),
        complete_marginal_candidates=full,
        state_word_metric_evals=state_evals,
        bit_metric_accumulations=bit_accums,
        osd_reprocessings=0,
        preprocessing_state_metrics=len(states) + pre,
        wall_seconds=elapsed,
    )


def latent_osd_batch(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    *,
    order: int = 2,
    pool_size: int = 14,
    state_limit: int | None = None,
    candidate_cap: int = 100_000,
    score_batch_size: int = 512,
) -> DecoderAuditResult:
    """Matched state-aware OSD union with cached exact marginal rescoring.

    All states contribute OSD-0 candidates.  The selected states additionally
    contribute order-1..order reprocessings.  By default every modeled state is
    selected, making this substantially stronger and fairer than the v1.0
    truncated Python baseline.
    """
    start = time.perf_counter()
    hard_table, cost_table, base_table = _state_metric_tables(y, n0, states)
    state_data: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    candidates: dict[bytes, np.ndarray] = {}
    generated = 0
    reprocessings = 0
    for si, _state in enumerate(states):
        rel = cost_table[si]
        gmrb, info = most_reliable_basis_generator(code.G, rel)
        u0 = hard_table[si, info].copy()
        c0 = (u0 @ gmrb) & 1
        key = word_key(c0)
        candidates.setdefault(key, c0)
        fixed = float(base_table[si] - rel[np.flatnonzero(c0 ^ hard_table[si])].sum())
        state_data.append((fixed, si, gmrb, info))
        generated += 1
        reprocessings += 1
    state_data.sort(key=lambda z: z[0], reverse=True)
    limit = len(state_data) if state_limit is None else min(max(1, int(state_limit)), len(state_data))
    selected = state_data[:limit]
    touched: set[int] = set()
    cap_hit = False
    for _, si, gmrb, info in selected:
        touched.add(si)
        u0 = hard_table[si, info].copy()
        info_rel = cost_table[si, info]
        pool = np.argsort(info_rel, kind="stable")[: min(pool_size, code.k)]
        # OSD-0 was inserted above.  Add only positive orders here.
        for d in range(1, order + 1):
            for combo in itertools.combinations(pool.tolist(), d):
                u = u0.copy()
                u[np.fromiter(combo, dtype=int, count=len(combo))] ^= 1
                word = (u @ gmrb) & 1
                candidates.setdefault(word_key(word), word)
                generated += 1
                reprocessings += 1
                if len(candidates) >= candidate_cap:
                    cap_hit = True
                    break
            if cap_hit:
                break
        if cap_hit:
            break
    winner: np.ndarray | None = None
    state_evals = 0
    bit_accums = 0
    if candidates:
        cands = np.stack(list(candidates.values()))
        scores, state_evals, bit_accums, _ = batch_marginal_scores(
            y, n0, cands, states,
            batch_size=score_batch_size,
            tables=(hard_table, cost_table, base_table),
        )
        winner = cands[int(np.argmax(scores))]
    elapsed = time.perf_counter() - start
    bits, value = _bits_or_none(winner)
    return DecoderAuditResult(
        decoder=f"latent_osd_batch_o{order}_s{limit}",
        decoded_bits=bits,
        decoded_int=value,
        success=winner is not None,
        certified=False,
        cap_hit=cap_hit,
        stop_reason="best_marginal_in_osd_union" if winner is not None else "empty_osd_union",
        components_generated=generated,
        membership_queries=0,
        latent_hypotheses_available=len(states),
        latent_queues_touched=len(touched),
        valid_codewords=len(candidates),
        unique_candidates=len(candidates),
        complete_marginal_candidates=len(candidates),
        state_word_metric_evals=state_evals,
        bit_metric_accumulations=bit_accums,
        osd_reprocessings=reprocessings,
        preprocessing_state_metrics=len(states),
        wall_seconds=elapsed,
    )


def certified_lsgrand_audit(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    *,
    query_cap: int,
    certificate_interval: int = 16,
) -> DecoderAuditResult:
    start = time.perf_counter()
    r = certified_lsgrand(
        y, n0, code, states,
        query_cap=query_cap,
        certificate_interval=certificate_interval,
    )
    # Use the decoder's own elapsed time, but include adapter overhead only if
    # it somehow exceeds it materially.
    elapsed = max(r.wall_seconds, time.perf_counter() - start)
    bits = r.decoded_bits
    return DecoderAuditResult(
        decoder="lsgrand_certified_v1_bound",
        decoded_bits=bits,
        decoded_int=r.decoded_int,
        success=r.success,
        certified=r.certified,
        cap_hit=r.cap_hit,
        stop_reason=r.stop_reason,
        components_generated=r.residual_patterns_generated,
        membership_queries=r.membership_queries,
        latent_hypotheses_available=r.latent_hypotheses_available,
        latent_queues_touched=r.latent_queues_touched,
        valid_codewords=r.valid_witnesses,
        unique_candidates=r.unique_codewords_seen,
        complete_marginal_candidates=0,
        state_word_metric_evals=r.state_codeword_likelihoods,
        bit_metric_accumulations=r.state_codeword_likelihoods * code.n,
        osd_reprocessings=0,
        preprocessing_state_metrics=len(states),
        wall_seconds=elapsed,
    )


@dataclass(frozen=True)
class RankProbeResult:
    latent_true_component_rank: int | None
    latent_cap_hit: bool
    plain_true_word_rank: int | None
    plain_cap_hit: bool
    rank_ratio: float | None
    rank_ratio_lower_bound: float | None
    latent_components_generated: int
    plain_components_generated: int
    latent_queues_touched: int
    latent_hypotheses_available: int
    wall_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def empirical_true_ranks(
    y: np.ndarray,
    n0: float,
    transmitted_word: np.ndarray,
    states: Sequence[StateHypothesis],
    true_path: np.ndarray,
    *,
    latent_cap: int,
    plain_cap: int,
) -> RankProbeResult:
    """Measure noisy enumeration ranks without using code membership stopping."""
    start = time.perf_counter()
    target = np.asarray(transmitted_word, dtype=np.uint8)
    true_indices = [i for i, s in enumerate(states) if np.array_equal(s.path, true_path)]
    if len(true_indices) != 1:
        latent_rank = None
        latent_used = 0
        latent_cap_hit = True
        touched: set[int] = set()
    else:
        true_si = true_indices[0]
        queues = [StateQueue(y, s, n0) for s in states]
        heap: list[tuple[float, int, int]] = []
        serial = 0
        for si, q in enumerate(queues):
            heapq.heappush(heap, (-q.next_log_weight, serial, si))
            serial += 1
        latent_rank = None
        latent_used = 0
        touched = set()
        while heap and latent_used < latent_cap:
            _, _, si = heapq.heappop(heap)
            word, _, _ = queues[si].pop_word()
            latent_used += 1
            touched.add(si)
            if si == true_si and np.array_equal(word, target):
                latent_rank = latent_used
                break
            if np.isfinite(queues[si].next_log_weight):
                heapq.heappush(heap, (-queues[si].next_log_weight, serial, si))
                serial += 1
        latent_cap_hit = latent_rank is None

    # Plain no-slip stream.  This is the actual reliability-ordered rank under
    # the mismatched no-slip observation, not a Hamming-weight surrogate.
    no_slip = StateHypothesis(
        index=0,
        path=np.zeros(len(y), dtype=np.int8),
        log_prior=0.0,
        label="plain_no_slip",
        slip_count=0,
        slip_locations=(),
        increments=(),
    )
    q0 = StateQueue(y, no_slip, n0)
    plain_rank = None
    plain_used = 0
    while plain_used < plain_cap:
        try:
            word, _, _ = q0.pop_word()
        except StopIteration:
            break
        plain_used += 1
        if np.array_equal(word, target):
            plain_rank = plain_used
            break
    plain_cap_hit = plain_rank is None

    ratio = None
    lower = None
    if latent_rank is not None:
        if plain_rank is not None:
            ratio = plain_rank / latent_rank
            lower = ratio
        else:
            lower = plain_cap / latent_rank
    return RankProbeResult(
        latent_true_component_rank=latent_rank,
        latent_cap_hit=latent_cap_hit,
        plain_true_word_rank=plain_rank,
        plain_cap_hit=plain_cap_hit,
        rank_ratio=ratio,
        rank_ratio_lower_bound=lower,
        latent_components_generated=latent_used,
        plain_components_generated=plain_used,
        latent_queues_touched=len(touched),
        latent_hypotheses_available=len(states),
        wall_seconds=time.perf_counter() - start,
    )
