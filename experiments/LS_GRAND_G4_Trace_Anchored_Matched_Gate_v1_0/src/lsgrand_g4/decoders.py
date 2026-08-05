from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Sequence

import numpy as np

from .channel import StateHypothesis, bits_to_qpsk, compensate_state, qpsk_bit_llrs, qpsk_loglikelihood, stream_parameters
from .gf2 import LinearCode, bits_to_int, most_reliable_basis_generator
from .search import StateQueue, logsumexp, word_key
from .structured_codes import ExtendedBCHDecoder, PolarSCDecoder

_DECODER_CACHE: dict[tuple[str, int, int, bytes], object] = {}


def _cached_decoder(code: LinearCode, kind: str):
    key = (kind, code.n, code.k, code.G.tobytes())
    decoder = _DECODER_CACHE.get(key)
    if decoder is None:
        decoder = ExtendedBCHDecoder(code) if kind == "bch" else PolarSCDecoder(code)
        _DECODER_CACHE[key] = decoder
    return decoder



@dataclass
class DecoderResult:
    decoder: str
    decoded_bits: list[int] | None
    decoded_int: int | None
    success: bool
    cap_hit: bool
    stop_reason: str
    alarm: bool
    trigger_metric: float | None
    components_generated: int
    membership_queries: int
    latent_hypotheses_available: int
    latent_queues_touched: int
    valid_witnesses: int
    unique_candidates: int
    complete_marginal_candidates: int
    state_word_metric_evals: int
    bit_metric_accumulations: int
    osd_reprocessings: int
    bch_decode_attempts: int
    preprocessing_state_metrics: int
    wall_seconds: float
    first_component_gap: float | None = None

    @property
    def queue_touch_fraction(self) -> float:
        if self.latent_hypotheses_available <= 0:
            return float("nan")
        return self.latent_queues_touched / self.latent_hypotheses_available

    def to_dict(self) -> dict:
        out = asdict(self)
        out["queue_touch_fraction"] = self.queue_touch_fraction
        return out


def _result_bits(bits: np.ndarray | None) -> tuple[list[int] | None, int | None]:
    if bits is None:
        return None, None
    b = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    return [int(x) for x in b], bits_to_int(b)


def state_metric_tables(
    y: np.ndarray,
    n0: float,
    states: Sequence[StateHypothesis],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hard, costs, base = [], [], []
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
    batch_size: int = 256,
    tables: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, int, int, int]:
    cands = np.asarray(candidates, dtype=np.uint8)
    if cands.ndim != 2:
        raise ValueError("candidates must be a two-dimensional binary array")
    if tables is None:
        hard, costs, base = state_metric_tables(y, n0, states)
        preprocessing = len(states)
    else:
        hard, costs, base = tables
        preprocessing = 0
    if cands.shape[1] != hard.shape[1]:
        raise ValueError("candidate/state length mismatch")
    scores = np.empty(cands.shape[0], dtype=float)
    for start in range(0, cands.shape[0], max(1, int(batch_size))):
        block = cands[start : start + int(batch_size)]
        diff = np.bitwise_xor(block[:, None, :], hard[None, :, :]).astype(float, copy=False)
        penalties = np.einsum("bsn,sn->bs", diff, costs, optimize=True)
        vals = base[None, :] - penalties
        maxima = np.max(vals, axis=1)
        scores[start : start + block.shape[0]] = maxima + np.log(
            np.exp(vals - maxima[:, None]).sum(axis=1)
        )
    state_evals = int(cands.shape[0] * len(states))
    bit_accums = int(state_evals * cands.shape[1])
    return scores, state_evals, bit_accums, preprocessing


def latent_list_decode(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    *,
    list_size: int,
    query_cap: int,
    marginal_rescore: bool,
    score_batch_size: int = 256,
) -> DecoderResult:
    if list_size < 1:
        raise ValueError("list_size must be positive")
    start = time.perf_counter()
    queues = [StateQueue(y, state, n0) for state in states]
    tables = (
        np.asarray([q.hard for q in queues], dtype=np.uint8),
        np.asarray([q.costs for q in queues], dtype=float),
        np.asarray([q.base_log_weight for q in queues], dtype=float),
    )
    heap: list[tuple[float, int, int]] = []
    serial = 0
    for si, queue in enumerate(queues):
        if np.isfinite(queue.next_log_weight):
            heapq.heappush(heap, (-queue.next_log_weight, serial, si))
            serial += 1
    touched: set[int] = set()
    membership_cache: dict[bytes, bool] = {}
    valid: dict[bytes, np.ndarray] = {}
    components = 0
    membership_queries = 0
    valid_witnesses = 0
    first_logw: float | None = None
    first_gap: float | None = None
    while heap and components < int(query_cap) and len(valid) < int(list_size):
        neg, _, si = heapq.heappop(heap)
        queue = queues[si]
        word, logw, _ = queue.pop_word()
        components += 1
        touched.add(si)
        if np.isfinite(queue.next_log_weight):
            heapq.heappush(heap, (-queue.next_log_weight, serial, si))
            serial += 1
        key = word_key(word)
        is_valid = membership_cache.get(key)
        if is_valid is None:
            is_valid = code.contains(word)
            membership_cache[key] = is_valid
            membership_queries += 1
        if is_valid:
            valid_witnesses += 1
            if key not in valid:
                valid[key] = word.copy()
                if first_logw is None:
                    first_logw = logw
                    next_head = -heap[0][0] if heap else float("-inf")
                    first_gap = float(logw - next_head) if np.isfinite(next_head) else float("inf")
    cap_hit = components >= int(query_cap) and len(valid) < int(list_size)
    winner: np.ndarray | None = None
    full = state_evals = bit_accums = extra_pre = 0
    if valid:
        cands = np.stack(list(valid.values()))
        if marginal_rescore:
            scores, state_evals, bit_accums, extra_pre = batch_marginal_scores(
                y, n0, cands, states, batch_size=score_batch_size, tables=tables
            )
            winner = cands[int(np.argmax(scores))]
            full = int(cands.shape[0])
        else:
            winner = cands[0]
    elapsed = time.perf_counter() - start
    bits, value = _result_bits(winner)
    return DecoderResult(
        decoder=(f"ls_list_marginal_L{list_size}" if marginal_rescore else "ls_first_valid"),
        decoded_bits=bits,
        decoded_int=value,
        success=winner is not None,
        cap_hit=cap_hit,
        stop_reason=("list_complete" if winner is not None and not cap_hit else "cap_or_exhaustion"),
        alarm=True,
        trigger_metric=None,
        components_generated=components,
        membership_queries=membership_queries,
        latent_hypotheses_available=len(states),
        latent_queues_touched=len(touched),
        valid_witnesses=valid_witnesses,
        unique_candidates=len(valid),
        complete_marginal_candidates=full,
        state_word_metric_evals=state_evals,
        bit_metric_accumulations=bit_accums,
        osd_reprocessings=0,
        bch_decode_attempts=0,
        preprocessing_state_metrics=len(states),
        wall_seconds=elapsed,
        first_component_gap=first_gap,
    )


def latent_adaptive_l2(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    *,
    logsumexp_gain_threshold: float,
    query_cap: int,
    score_batch_size: int = 256,
) -> DecoderResult:
    """First-valid search with L=2 only when the first word has material multi-state support.

    The trigger is ``log Lambda(c1) - log q_max(c1)``.  It is zero when one
    state component entirely dominates and equals the logarithm of an effective
    path multiplicity otherwise.  This directly targets the weak fiber effect
    observed in v1.1 without forcing an expensive second-codeword search on
    every alarmed frame.
    """
    start = time.perf_counter()
    queues = [StateQueue(y, state, n0) for state in states]
    tables = (
        np.asarray([q.hard for q in queues], dtype=np.uint8),
        np.asarray([q.costs for q in queues], dtype=float),
        np.asarray([q.base_log_weight for q in queues], dtype=float),
    )
    heap: list[tuple[float, int, int]] = []
    serial = 0
    for si, q in enumerate(queues):
        heapq.heappush(heap, (-q.next_log_weight, serial, si))
        serial += 1
    touched: set[int] = set()
    membership_cache: dict[bytes, bool] = {}
    valid: dict[bytes, np.ndarray] = {}
    components = membership_queries = valid_witnesses = 0
    gain: float | None = None
    first_score: float | None = None
    invoked_l2 = False
    trigger_state_evals = trigger_bit_accums = trigger_pre = 0
    while heap and components < int(query_cap):
        _, _, si = heapq.heappop(heap)
        q = queues[si]
        word, logw, _ = q.pop_word()
        components += 1
        touched.add(si)
        if np.isfinite(q.next_log_weight):
            heapq.heappush(heap, (-q.next_log_weight, serial, si))
            serial += 1
        key = word_key(word)
        valid_flag = membership_cache.get(key)
        if valid_flag is None:
            valid_flag = code.contains(word)
            membership_cache[key] = valid_flag
            membership_queries += 1
        if not valid_flag:
            continue
        valid_witnesses += 1
        if key in valid:
            continue
        valid[key] = word.copy()
        if len(valid) == 1:
            score, trigger_state_evals, trigger_bit_accums, trigger_pre = batch_marginal_scores(
                y, n0, word.reshape(1, -1), states,
                batch_size=score_batch_size, tables=tables,
            )
            first_score = float(score[0])
            gain = float(first_score - logw)
            if gain <= float(logsumexp_gain_threshold):
                elapsed = time.perf_counter() - start
                bits, value = _result_bits(word)
                return DecoderResult(
                    decoder="ls_adaptive_L2_pathgain",
                    decoded_bits=bits,
                    decoded_int=value,
                    success=True,
                    cap_hit=False,
                    stop_reason="dominant_path_for_first_word",
                    alarm=True,
                    trigger_metric=None,
                    components_generated=components,
                    membership_queries=membership_queries,
                    latent_hypotheses_available=len(states),
                    latent_queues_touched=len(touched),
                    valid_witnesses=valid_witnesses,
                    unique_candidates=1,
                    complete_marginal_candidates=1,
                    state_word_metric_evals=trigger_state_evals,
                    bit_metric_accumulations=trigger_bit_accums,
                    osd_reprocessings=0,
                    bch_decode_attempts=0,
                    preprocessing_state_metrics=len(states),
                    wall_seconds=elapsed,
                    first_component_gap=gain,
                )
            invoked_l2 = True
        if invoked_l2 and len(valid) >= 2:
            break
    cap_hit = components >= int(query_cap) and len(valid) < (2 if invoked_l2 else 1)
    winner: np.ndarray | None = None
    full = state_evals = bit_accums = extra_pre = 0
    if valid:
        cands = np.stack(list(valid.values()))
        if invoked_l2 and cands.shape[0] >= 2:
            second_score, second_evals, second_accums, extra_pre = batch_marginal_scores(
                y, n0, cands[1:2], states,
                batch_size=score_batch_size, tables=tables,
            )
            state_evals = trigger_state_evals + second_evals
            bit_accums = trigger_bit_accums + second_accums
            winner = cands[0] if first_score is not None and first_score >= float(second_score[0]) else cands[1]
            full = 2
        else:
            winner = cands[0]
            state_evals = trigger_state_evals
            bit_accums = trigger_bit_accums
            extra_pre = trigger_pre
            full = 1
    elapsed = time.perf_counter() - start
    bits, value = _result_bits(winner)
    return DecoderResult(
        decoder="ls_adaptive_L2_pathgain",
        decoded_bits=bits,
        decoded_int=value,
        success=winner is not None,
        cap_hit=cap_hit,
        stop_reason=("adaptive_l2_complete" if winner is not None and not cap_hit else "cap_or_exhaustion"),
        alarm=True,
        trigger_metric=None,
        components_generated=components,
        membership_queries=membership_queries,
        latent_hypotheses_available=len(states),
        latent_queues_touched=len(touched),
        valid_witnesses=valid_witnesses,
        unique_candidates=len(valid),
        complete_marginal_candidates=full,
        state_word_metric_evals=state_evals,
        bit_metric_accumulations=bit_accums,
        osd_reprocessings=0,
        bch_decode_attempts=0,
        preprocessing_state_metrics=len(states),
        wall_seconds=elapsed,
        first_component_gap=gain,
    )

def latent_osd_batch(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    *,
    order: int = 2,
    pool_size: int = 10,
    state_limit: int | None = None,
    candidate_cap: int = 100_000,
    score_batch_size: int = 256,
) -> DecoderResult:
    """State-aware OSD with explicit state pruning and full marginal rescoring.

    States are ranked by the likelihood of their unconstrained hard word.  Only
    the frozen top-K states are reprocessed, but every retained candidate is
    scored across *all* modeled states.  ``state_limit=None`` recovers the full
    all-state union used in v1.1.
    """
    start = time.perf_counter()
    hard_table, cost_table, base_table = state_metric_tables(y, n0, states)
    limit = len(states) if state_limit is None else min(max(1, int(state_limit)), len(states))
    selected_indices = np.argsort(-base_table, kind="stable")[:limit]
    candidates: dict[bytes, np.ndarray] = {}
    generated = reprocessings = 0
    touched: set[int] = set()
    cap_hit = False
    for si in selected_indices:
        si = int(si)
        touched.add(si)
        rel = cost_table[si]
        gmrb, info = most_reliable_basis_generator(code.G, rel)
        u0 = hard_table[si, info].copy()
        c0 = (u0 @ gmrb) & 1
        candidates.setdefault(word_key(c0), c0)
        generated += 1
        reprocessings += 1
        pool = np.argsort(rel[info], kind="stable")[: min(int(pool_size), code.k)]
        for d in range(1, int(order) + 1):
            for combo in itertools.combinations(pool.tolist(), d):
                u = u0.copy()
                u[np.fromiter(combo, dtype=int, count=len(combo))] ^= 1
                word = (u @ gmrb) & 1
                candidates.setdefault(word_key(word), word)
                generated += 1
                reprocessings += 1
                if len(candidates) >= int(candidate_cap):
                    cap_hit = True
                    break
            if cap_hit:
                break
        if cap_hit:
            break
    winner = None
    state_evals = bit_accums = 0
    if candidates:
        cands = np.stack(list(candidates.values()))
        scores, state_evals, bit_accums, _ = batch_marginal_scores(
            y, n0, cands, states, batch_size=score_batch_size,
            tables=(hard_table, cost_table, base_table),
        )
        winner = cands[int(np.argmax(scores))]
    elapsed = time.perf_counter() - start
    bits, value = _result_bits(winner)
    return DecoderResult(
        decoder=f"state_pruned_osd_o{order}_s{limit}",
        decoded_bits=bits,
        decoded_int=value,
        success=winner is not None,
        cap_hit=cap_hit,
        stop_reason="best_marginal_in_pruned_state_osd_union" if winner is not None else "empty_union",
        alarm=True,
        trigger_metric=None,
        components_generated=generated,
        membership_queries=0,
        latent_hypotheses_available=len(states),
        latent_queues_touched=len(touched),
        valid_witnesses=len(candidates),
        unique_candidates=len(candidates),
        complete_marginal_candidates=len(candidates),
        state_word_metric_evals=state_evals,
        bit_metric_accumulations=bit_accums,
        osd_reprocessings=reprocessings,
        bch_decode_attempts=0,
        preprocessing_state_metrics=len(states),
        wall_seconds=elapsed,
    )

def bch_chase_state_sweep(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    *,
    chase_order: int = 2,
    chase_pool: int = 6,
    chase_state_limit: int = 8,
    candidate_cap: int = 100_000,
    score_batch_size: int = 256,
) -> DecoderResult:
    """Code-specific state sweep with algebraic BCH decoding.

    Every state receives one hard algebraic decode.  Positive-order Chase
    perturbations are applied only to the frozen top-K states ranked by their
    unconstrained hard-word likelihood.  All deduplicated outputs are scored
    across the complete state family.
    """
    start = time.perf_counter()
    decoder = _cached_decoder(code, "bch")
    hard_table, cost_table, base_table = state_metric_tables(y, n0, states)
    candidates: dict[bytes, np.ndarray] = {}
    attempts = generated = 0
    touched: set[int] = set(range(len(states)))
    cap_hit = False
    selected = set(int(x) for x in np.argsort(-base_table, kind="stable")[: min(int(chase_state_limit), len(states))])
    for si in range(len(states)):
        hard = hard_table[si]
        # Order zero for every state.
        candidate = decoder.decode(hard)
        attempts += 1
        generated += 1
        if candidate is not None:
            candidates.setdefault(word_key(candidate), candidate)
        if si not in selected:
            continue
        pool = np.argsort(cost_table[si], kind="stable")[: min(int(chase_pool), code.n)]
        for d in range(1, int(chase_order) + 1):
            for combo in itertools.combinations(pool.tolist(), d):
                trial = hard.copy()
                trial[np.fromiter(combo, dtype=int, count=len(combo))] ^= 1
                candidate = decoder.decode(trial)
                attempts += 1
                generated += 1
                if candidate is not None:
                    candidates.setdefault(word_key(candidate), candidate)
                if len(candidates) >= int(candidate_cap):
                    cap_hit = True
                    break
            if cap_hit:
                break
        if cap_hit:
            break
    winner = None
    state_evals = bit_accums = 0
    if candidates:
        cands = np.stack(list(candidates.values()))
        scores, state_evals, bit_accums, _ = batch_marginal_scores(
            y, n0, cands, states, batch_size=score_batch_size,
            tables=(hard_table, cost_table, base_table),
        )
        winner = cands[int(np.argmax(scores))]
    elapsed = time.perf_counter() - start
    bits, value = _result_bits(winner)
    return DecoderResult(
        decoder=f"state_sweep_chase_bch_o{chase_order}_p{chase_pool}_s{len(selected)}",
        decoded_bits=bits,
        decoded_int=value,
        success=winner is not None,
        cap_hit=cap_hit,
        stop_reason="best_marginal_chase_candidate" if winner is not None else "no_chase_candidate",
        alarm=True,
        trigger_metric=None,
        components_generated=generated,
        membership_queries=0,
        latent_hypotheses_available=len(states),
        latent_queues_touched=len(touched),
        valid_witnesses=len(candidates),
        unique_candidates=len(candidates),
        complete_marginal_candidates=len(candidates),
        state_word_metric_evals=state_evals,
        bit_metric_accumulations=bit_accums,
        osd_reprocessings=0,
        bch_decode_attempts=attempts,
        preprocessing_state_metrics=len(states),
        wall_seconds=elapsed,
    )


def polar_scflip_state_sweep(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    *,
    flip_trials: int = 4,
    score_batch_size: int = 256,
) -> DecoderResult:
    """Decode every state with polar SC plus frozen first-order SC-Flip trials."""
    start = time.perf_counter()
    decoder = _cached_decoder(code, "polar")
    hard_table, cost_table, base_table = state_metric_tables(y, n0, states)
    candidates: dict[bytes, np.ndarray] = {}
    attempts = 0
    for state in states:
        llr = qpsk_bit_llrs(compensate_state(y, state.path), n0)
        local = decoder.decode_candidates(llr, flip_trials=int(flip_trials))
        attempts += 1 + int(flip_trials)
        for candidate in local:
            candidates.setdefault(word_key(candidate), candidate)
    winner = None
    state_evals = bit_accums = 0
    if candidates:
        cands = np.stack(list(candidates.values()))
        scores, state_evals, bit_accums, _ = batch_marginal_scores(
            y, n0, cands, states, batch_size=score_batch_size,
            tables=(hard_table, cost_table, base_table),
        )
        winner = cands[int(np.argmax(scores))]
    elapsed = time.perf_counter() - start
    bits, value = _result_bits(winner)
    return DecoderResult(
        decoder=f"state_sweep_polar_scflip_t{flip_trials}",
        decoded_bits=bits,
        decoded_int=value,
        success=winner is not None,
        cap_hit=False,
        stop_reason="best_marginal_polar_scflip_candidate" if winner is not None else "no_polar_candidate",
        alarm=True,
        trigger_metric=None,
        components_generated=attempts,
        membership_queries=0,
        latent_hypotheses_available=len(states),
        latent_queues_touched=len(states),
        valid_witnesses=len(candidates),
        unique_candidates=len(candidates),
        complete_marginal_candidates=len(candidates),
        state_word_metric_evals=state_evals,
        bit_metric_accumulations=bit_accums,
        osd_reprocessings=0,
        bch_decode_attempts=0,
        preprocessing_state_metrics=len(states),
        wall_seconds=elapsed,
    )

def normalized_no_slip_residual(y: np.ndarray, n0: float, decoded_bits: np.ndarray) -> float:
    mean = bits_to_qpsk(decoded_bits)
    return float(np.sum(np.abs(np.asarray(y) - mean) ** 2) / (len(mean) * float(n0)))


def combine_event_result(
    no_slip: DecoderResult,
    recovery: DecoderResult | None,
    *,
    decoder_name: str,
    alarm: bool,
    trigger_metric: float,
) -> DecoderResult:
    chosen = recovery if alarm and recovery is not None else no_slip
    if chosen.decoded_bits is None:
        bits = value = None
    else:
        bits = list(chosen.decoded_bits)
        value = chosen.decoded_int
    if alarm and recovery is not None:
        return DecoderResult(
            decoder=decoder_name,
            decoded_bits=bits,
            decoded_int=value,
            success=chosen.success,
            cap_hit=chosen.cap_hit,
            stop_reason=f"alarm_then_{chosen.stop_reason}",
            alarm=True,
            trigger_metric=float(trigger_metric),
            components_generated=no_slip.components_generated + chosen.components_generated,
            membership_queries=no_slip.membership_queries + chosen.membership_queries,
            latent_hypotheses_available=chosen.latent_hypotheses_available,
            latent_queues_touched=chosen.latent_queues_touched,
            valid_witnesses=no_slip.valid_witnesses + chosen.valid_witnesses,
            unique_candidates=no_slip.unique_candidates + chosen.unique_candidates,
            complete_marginal_candidates=no_slip.complete_marginal_candidates + chosen.complete_marginal_candidates,
            state_word_metric_evals=no_slip.state_word_metric_evals + chosen.state_word_metric_evals,
            bit_metric_accumulations=no_slip.bit_metric_accumulations + chosen.bit_metric_accumulations,
            osd_reprocessings=no_slip.osd_reprocessings + chosen.osd_reprocessings,
            bch_decode_attempts=no_slip.bch_decode_attempts + chosen.bch_decode_attempts,
            preprocessing_state_metrics=no_slip.preprocessing_state_metrics + chosen.preprocessing_state_metrics,
            wall_seconds=no_slip.wall_seconds + chosen.wall_seconds,
            first_component_gap=chosen.first_component_gap,
        )
    return replace(
        no_slip,
        decoder=decoder_name,
        alarm=False,
        trigger_metric=float(trigger_metric),
        stop_reason="no_alarm_accept_no_slip",
    )


def _posterior_ranked_state_indices(
    y: np.ndarray,
    n0: float,
    anchor_bits: np.ndarray,
    states: Sequence[StateHypothesis],
    limit: int,
) -> np.ndarray:
    """Rank frozen one-slip hypotheses using one decoded anchor codeword."""
    anchor = np.asarray(anchor_bits, dtype=np.uint8).reshape(-1)
    scores = np.asarray([
        state.log_prior + qpsk_loglikelihood(y, anchor, state.path, n0)
        for state in states
    ], dtype=float)
    return np.argsort(-scores, kind="stable")[: min(max(1, int(limit)), len(states))]


def posterior_pruned_bch_sweep(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    anchor_bits: np.ndarray,
    *,
    state_limit: int = 4,
    chase_order: int = 2,
    chase_pool: int = 6,
    candidate_cap: int = 100_000,
    score_batch_size: int = 256,
) -> DecoderResult:
    """Dangerous optimized competitor: decode only posterior-leading states.

    The state ranking uses the ordinary decoded word as a code-aided anchor.
    Candidate generation is restricted to the frozen top-K states, but every
    deduplicated candidate is rescored over the complete one-slip state family.
    """
    start = time.perf_counter()
    decoder = _cached_decoder(code, "bch")
    hard_table, cost_table, base_table = state_metric_tables(y, n0, states)
    selected = _posterior_ranked_state_indices(y, n0, anchor_bits, states, state_limit)
    candidates: dict[bytes, np.ndarray] = {}
    attempts = generated = 0
    cap_hit = False
    for si0 in selected:
        si = int(si0)
        hard = hard_table[si]
        candidate = decoder.decode(hard)
        attempts += 1
        generated += 1
        if candidate is not None:
            candidates.setdefault(word_key(candidate), candidate)
        pool = np.argsort(cost_table[si], kind="stable")[: min(int(chase_pool), code.n)]
        for d in range(1, int(chase_order) + 1):
            for combo in itertools.combinations(pool.tolist(), d):
                trial = hard.copy()
                trial[np.fromiter(combo, dtype=int, count=len(combo))] ^= 1
                candidate = decoder.decode(trial)
                attempts += 1
                generated += 1
                if candidate is not None:
                    candidates.setdefault(word_key(candidate), candidate)
                if len(candidates) >= int(candidate_cap):
                    cap_hit = True
                    break
            if cap_hit:
                break
        if cap_hit:
            break
    winner = None
    state_evals = bit_accums = 0
    if candidates:
        cands = np.stack(list(candidates.values()))
        scores, state_evals, bit_accums, _ = batch_marginal_scores(
            y, n0, cands, states, batch_size=score_batch_size,
            tables=(hard_table, cost_table, base_table),
        )
        winner = cands[int(np.argmax(scores))]
    bits, value = _result_bits(winner)
    return DecoderResult(
        decoder=f"posterior_pruned_chase_bch_k{len(selected)}",
        decoded_bits=bits,
        decoded_int=value,
        success=winner is not None,
        cap_hit=cap_hit,
        stop_reason="best_marginal_posterior_pruned_chase" if winner is not None else "no_candidate",
        alarm=True,
        trigger_metric=None,
        components_generated=generated,
        membership_queries=0,
        latent_hypotheses_available=len(states),
        latent_queues_touched=len(selected),
        valid_witnesses=len(candidates),
        unique_candidates=len(candidates),
        complete_marginal_candidates=len(candidates),
        state_word_metric_evals=state_evals,
        bit_metric_accumulations=bit_accums,
        osd_reprocessings=0,
        bch_decode_attempts=attempts,
        preprocessing_state_metrics=len(states),
        wall_seconds=time.perf_counter() - start,
    )


def posterior_pruned_polar_sweep(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    anchor_bits: np.ndarray,
    *,
    state_limit: int = 4,
    flip_trials: int = 8,
    score_batch_size: int = 256,
) -> DecoderResult:
    """Code-aided posterior-pruned polar SC/SC-Flip competitor."""
    start = time.perf_counter()
    decoder = _cached_decoder(code, "polar")
    hard_table, cost_table, base_table = state_metric_tables(y, n0, states)
    selected = _posterior_ranked_state_indices(y, n0, anchor_bits, states, state_limit)
    candidates: dict[bytes, np.ndarray] = {}
    attempts = 0
    for si0 in selected:
        si = int(si0)
        state = states[si]
        llr = qpsk_bit_llrs(compensate_state(y, state.path), n0)
        local = decoder.decode_candidates(llr, flip_trials=int(flip_trials))
        attempts += 1 + int(flip_trials)
        for candidate in local:
            candidates.setdefault(word_key(candidate), candidate)
    winner = None
    state_evals = bit_accums = 0
    if candidates:
        cands = np.stack(list(candidates.values()))
        scores, state_evals, bit_accums, _ = batch_marginal_scores(
            y, n0, cands, states, batch_size=score_batch_size,
            tables=(hard_table, cost_table, base_table),
        )
        winner = cands[int(np.argmax(scores))]
    bits, value = _result_bits(winner)
    return DecoderResult(
        decoder=f"posterior_pruned_polar_scflip_k{len(selected)}",
        decoded_bits=bits,
        decoded_int=value,
        success=winner is not None,
        cap_hit=False,
        stop_reason="best_marginal_posterior_pruned_polar" if winner is not None else "no_candidate",
        alarm=True,
        trigger_metric=None,
        components_generated=attempts,
        membership_queries=0,
        latent_hypotheses_available=len(states),
        latent_queues_touched=len(selected),
        valid_witnesses=len(candidates),
        unique_candidates=len(candidates),
        complete_marginal_candidates=len(candidates),
        state_word_metric_evals=state_evals,
        bit_metric_accumulations=bit_accums,
        osd_reprocessings=0,
        bch_decode_attempts=0,
        preprocessing_state_metrics=len(states),
        wall_seconds=time.perf_counter() - start,
    )


def dqpsk_encode(bits: np.ndarray) -> np.ndarray:
    """Differentially encode sign-labelled QPSK labels with one reference."""
    data = bits_to_qpsk(bits) * np.exp(-1j * np.pi / 4.0)
    out = np.empty(data.size + 1, dtype=np.complex128)
    out[0] = np.exp(1j * np.pi / 4.0)
    for t, inc in enumerate(data):
        out[t + 1] = out[t] * inc
    return out


def dqpsk_observations(y: np.ndarray) -> np.ndarray:
    """Return sign-labelled QPSK-like differential observations."""
    yy = np.asarray(y, dtype=np.complex128).reshape(-1)
    if yy.size < 2:
        raise ValueError("DQPSK sequence requires a reference and data")
    z = yy[1:] * np.conjugate(yy[:-1])
    return z * np.exp(1j * np.pi / 4.0)


def exact_state_marginal_bit_llrs(
    y: np.ndarray,
    n0: float,
    states: Sequence[StateHypothesis],
) -> tuple[np.ndarray, int]:
    """Exact uncoded bit LLRs after marginalizing the frozen state family.

    This is the finite-hypothesis counterpart of a forward--backward/BCJR
    state-aware demapper.  QPSK labels are treated as independent and uniform
    during front-end state inference; code constraints enter only in the
    subsequent code-specific decoder.
    """
    yy = np.asarray(y, dtype=np.complex128).reshape(-1)
    if n0 <= 0:
        raise ValueError("n0 must be positive")
    labels = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    symbols = np.asarray([bits_to_qpsk(label)[0] for label in labels], dtype=np.complex128)
    s_count = len(states)
    t_count = yy.size
    # S x T x 4 matched symbol log metrics; constants cancel in LLRs.
    metrics = np.empty((s_count, t_count, 4), dtype=float)
    for si, state in enumerate(states):
        rot = np.power(1j, np.asarray(state.path, dtype=np.int8).reshape(-1) % 4)
        means = rot[:, None] * symbols[None, :]
        metrics[si] = -np.abs(yy[:, None] - means) ** 2 / float(n0)

    def lse(a: np.ndarray, axis: int) -> np.ndarray:
        m = np.max(a, axis=axis, keepdims=True)
        return np.squeeze(m, axis=axis) + np.log(np.exp(a - m).sum(axis=axis))

    log_z = lse(metrics, axis=2) - np.log(4.0)  # S x T
    path_total = np.asarray([s.log_prior for s in states], dtype=float) + log_z.sum(axis=1)
    llrs = np.empty(2 * t_count, dtype=float)
    for t in range(t_count):
        base = path_total - log_z[:, t]
        for bit in range(2):
            m0 = labels[:, bit] == 0
            m1 = ~m0
            z0 = lse(metrics[:, t, :][:, m0], axis=1) - np.log(4.0)
            z1 = lse(metrics[:, t, :][:, m1], axis=1) - np.log(4.0)
            a0 = base + z0
            a1 = base + z1
            llrs[2 * t + bit] = float(lse(a0[:, None], axis=0)[0] - lse(a1[:, None], axis=0)[0])
    branch_metrics = int(s_count * t_count * 4)
    return llrs, branch_metrics


def state_marginal_code_specific(
    y: np.ndarray,
    n0: float,
    code: LinearCode,
    states: Sequence[StateHypothesis],
    *,
    bch_chase_order: int = 2,
    bch_chase_pool: int = 8,
    polar_flip_trials: int = 16,
    candidate_cap: int = 100_000,
    score_batch_size: int = 256,
) -> DecoderResult:
    """State-marginal demapper followed by one strong code-specific decode.

    This baseline is intentionally cheap and dangerous for LS-GRAND: the state
    family is marginalized before decoding, after which BCH Chase or polar
    SC/SC-Flip produces a small codeword list.  All candidates are reranked by
    the same complete state-marginal metric used elsewhere.
    """
    start = time.perf_counter()
    llr, branch_metrics = exact_state_marginal_bit_llrs(y, n0, states)
    candidates: dict[bytes, np.ndarray] = {}
    attempts = 0
    cap_hit = False
    if code.family == "extended_bch":
        decoder = _cached_decoder(code, "bch")
        hard = (llr < 0).astype(np.uint8)
        pool = np.argsort(np.abs(llr), kind="stable")[: min(int(bch_chase_pool), code.n)]
        for d in range(int(bch_chase_order) + 1):
            for combo in itertools.combinations(pool.tolist(), d):
                trial = hard.copy()
                if combo:
                    trial[np.fromiter(combo, dtype=int, count=len(combo))] ^= 1
                candidate = decoder.decode(trial)
                attempts += 1
                if candidate is not None:
                    candidates.setdefault(word_key(candidate), candidate)
                if len(candidates) >= int(candidate_cap):
                    cap_hit = True
                    break
            if cap_hit:
                break
    elif code.family == "polar":
        decoder = _cached_decoder(code, "polar")
        local = decoder.decode_candidates(llr, flip_trials=int(polar_flip_trials))
        attempts = 1 + int(polar_flip_trials)
        for candidate in local:
            candidates.setdefault(word_key(candidate), candidate)
    else:
        raise ValueError("state-marginal code-specific baseline requires BCH or polar code")

    winner = None
    state_evals = bit_accums = 0
    if candidates:
        cands = np.stack(list(candidates.values()))
        scores, state_evals, bit_accums, _ = batch_marginal_scores(
            y, n0, cands, states, batch_size=score_batch_size
        )
        winner = cands[int(np.argmax(scores))]
    bits, value = _result_bits(winner)
    return DecoderResult(
        decoder="state_marginal_code_specific",
        decoded_bits=bits,
        decoded_int=value,
        success=winner is not None,
        cap_hit=cap_hit,
        stop_reason="best_complete_score_after_state_marginal_demapper" if winner is not None else "no_candidate",
        alarm=True,
        trigger_metric=None,
        components_generated=attempts,
        membership_queries=0,
        latent_hypotheses_available=len(states),
        latent_queues_touched=len(states),
        valid_witnesses=len(candidates),
        unique_candidates=len(candidates),
        complete_marginal_candidates=len(candidates),
        state_word_metric_evals=state_evals,
        bit_metric_accumulations=bit_accums + branch_metrics,
        osd_reprocessings=0,
        bch_decode_attempts=attempts if code.family == "extended_bch" else 0,
        preprocessing_state_metrics=branch_metrics,
        wall_seconds=time.perf_counter() - start,
    )
