from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .affine import find_orbit_safe_interleaver
from .channel import bits_to_qpsk, one_slip_hypotheses, qpsk_hard_bits
from .decoders import (
    DecoderResult,
    bch_chase_state_sweep,
    combine_event_result,
    dqpsk_encode,
    dqpsk_observations,
    latent_adaptive_l2,
    latent_list_decode,
    latent_osd_batch,
    normalized_no_slip_residual,
    polar_scflip_state_sweep,
    posterior_pruned_bch_sweep,
    posterior_pruned_polar_sweep,
    state_marginal_code_specific,
)
from .gf2 import LinearCode, bits_to_int
from .stats import (
    clopper_pearson_interval,
    mcnemar_one_sided,
    paired_bootstrap_median_ratio,
    paired_error_difference_interval,
    safe_quantile,
    weighted_quantile,
    wilson_interval,
)
from .structured_codes import build_frozen_code
from .trace import TracePool, collect_trace_pool


EXPECTED_G0_COMMIT = "0d2866b091576fe07521172378ded33da79ed545"
EXPECTED_G3P_COMMIT = "5edb4083ec30d44061f0020a2701fd189f87df23"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def seeded_rng(master: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(master), *(int(x) for x in parts)]))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    required = {
        "profile", "master_seed", "codes", "trace", "detector", "decoders",
        "trials", "controls", "thresholds",
    }
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"configuration missing keys: {sorted(missing)}")
    return cfg


def load_chain(package_root: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    g0 = json.loads((package_root / "data" / "G0_CHAIN_LOCK.json").read_text())
    g3p = json.loads((package_root / "data" / "G3P_CHAIN_LOCK.json").read_text())
    claims = pd.read_csv(package_root / "docs" / "02_CLAIM_BY_CLAIM_NOVELTY_MATRIX.csv")
    return g0, g3p, claims


def build_screened_codes(cfg: dict[str, Any], run_dir: Path) -> tuple[dict[str, LinearCode], pd.DataFrame, pd.DataFrame]:
    master = int(cfg["master_seed"])
    code_dir = run_dir / "codes"
    code_dir.mkdir(exist_ok=True)
    codes: dict[str, LinearCode] = {}
    summaries: list[dict[str, Any]] = []
    transforms: list[dict[str, Any]] = []
    for ci, spec in enumerate(cfg["codes"]):
        base = build_frozen_code(str(spec["spec"]), seeded_rng(master, 100, ci))
        screened, summary = find_orbit_safe_interleaver(
            base,
            seeded_rng(master, 101, ci),
            threshold=float(spec.get("orbit_threshold", cfg["thresholds"]["max_orbit_collision_fraction"])),
            max_attempts=int(spec.get("max_interleaver_attempts", 500)),
        )
        code_id = str(spec["id"])
        codes[code_id] = screened
        np.savez_compressed(
            code_dir / f"{code_id}.npz",
            G=screened.G,
            H=screened.H,
            permutation=np.asarray(screened.metadata.get("permutation", list(range(screened.n))), dtype=int),
            inverse_permutation=np.asarray(screened.metadata.get("inverse_permutation", list(range(screened.n))), dtype=int),
        )
        summaries.append({
            "code_id": code_id,
            "base_name": base.name,
            "screened_name": screened.name,
            "family": screened.family,
            "structured": bool(spec.get("structured", True)),
            "n": screened.n,
            "k": screened.k,
            "rate": screened.rate,
            "redundancy": screened.redundancy,
            "interleaved": bool(summary["interleaved"]),
            "interleaver_attempt": int(summary["attempt"]),
            "orbit_safe": bool(summary["orbit_safe"]),
            "max_collision_fraction": float(summary["max_collision_fraction"]),
            "mean_collision_fraction": float(summary["mean_collision_fraction"]),
            "union_bound": float(summary["union_bound"]),
            "dangerous_transform_count": int(summary["dangerous_transform_count"]),
            "worst_transform": summary["worst_transform"],
        })
        for record in summary["records"]:
            transforms.append({"code_id": code_id, **record})
    summary_df = pd.DataFrame(summaries)
    transform_df = pd.DataFrame(transforms)
    atomic_csv(summary_df, run_dir / "code_orbit_summary.csv")
    atomic_csv(transform_df, run_dir / "code_orbit_transforms.csv")
    return codes, summary_df, transform_df


def normal_code_decoder(y: np.ndarray, n0: float, code: LinearCode, dcfg: dict[str, Any]) -> DecoderResult:
    states = one_slip_hypotheses(code.n // 2, 0.0, (1, 3))
    if code.family == "extended_bch":
        result = bch_chase_state_sweep(
            y, n0, code, states,
            chase_order=int(dcfg["normal_bch_chase_order"]),
            chase_pool=int(dcfg["normal_bch_chase_pool"]),
            chase_state_limit=1,
            candidate_cap=int(dcfg["candidate_cap"]),
            score_batch_size=int(dcfg["score_batch_size"]),
        )
    elif code.family == "polar":
        result = polar_scflip_state_sweep(
            y, n0, code, states,
            flip_trials=int(dcfg["normal_polar_flip_trials"]),
            score_batch_size=int(dcfg["score_batch_size"]),
        )
    else:
        result = latent_osd_batch(
            y, n0, code, states,
            order=int(dcfg["normal_osd_order"]),
            pool_size=int(dcfg["normal_osd_pool"]),
            state_limit=1,
            candidate_cap=int(dcfg["candidate_cap"]),
            score_batch_size=int(dcfg["score_batch_size"]),
        )
    return replace(result, decoder="normal_code_decoder", alarm=False)


def _detector_metric(y: np.ndarray, n0: float, normal: DecoderResult, code: LinearCode) -> float:
    if normal.decoded_bits is None:
        return float("inf")
    return normalized_no_slip_residual(y, n0, np.asarray(normal.decoded_bits, dtype=np.uint8))


def _gain_to_observation(codeword: np.ndarray, gain: np.ndarray) -> np.ndarray:
    x = bits_to_qpsk(codeword)
    if gain.size < x.size:
        raise ValueError("trace is shorter than codeword")
    return x * gain[: x.size]


def calibrate_detector(
    cfg: dict[str, Any],
    codes: dict[str, LinearCode],
    calibration_gain: np.ndarray,
    run_dir: Path,
) -> tuple[dict[str, float], pd.DataFrame]:
    master = int(cfg["master_seed"])
    dcfg = cfg["decoders"]
    target_fa = float(cfg["detector"]["target_false_alarm_rate"])
    count = min(int(cfg["detector"]["calibration_frames"]), calibration_gain.shape[0])
    rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    for ci, (code_id, code) in enumerate(codes.items()):
        metrics = np.empty(count, dtype=float)
        for t in range(count):
            rng = seeded_rng(master, 300, ci, t)
            word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
            n0 = float(cfg["trace"]["n0"])
            y = _gain_to_observation(word, calibration_gain[t])
            start = time.perf_counter()
            normal = normal_code_decoder(y, n0, code, dcfg)
            metric = _detector_metric(y, n0, normal, code)
            detector_wall = time.perf_counter() - start - normal.wall_seconds
            metrics[t] = metric
            rows.append({
                "code_id": code_id,
                "trial": t,
                "trigger_metric": metric,
                "normal_decoded_correct": normal.decoded_int == bits_to_int(word),
                "normal_wall_seconds": normal.wall_seconds,
                "detector_extra_wall_seconds": max(0.0, detector_wall),
            })
        thresholds[code_id] = float(np.quantile(metrics, 1.0 - target_fa, method="higher"))
    df = pd.DataFrame(rows)
    atomic_csv(df, run_dir / "detector_calibration.csv")
    write_json(run_dir / "detector_thresholds.json", thresholds)
    return thresholds, df


def _decoder_row(
    result: DecoderResult,
    *,
    frame_uid: str,
    code_id: str,
    code: LinearCode,
    condition: str,
    tx_int: int,
    alarm: bool,
    metric: float,
    slip_location: int,
    slip_direction: int,
    total_symbols: int,
) -> dict[str, Any]:
    row = result.to_dict()
    row.update({
        "frame_uid": frame_uid,
        "code_id": code_id,
        "family": code.family,
        "n": code.n,
        "k": code.k,
        "rate": code.rate,
        "condition": condition,
        "transmitted_int": tx_int,
        "decoded_correct": result.decoded_int == tx_int,
        "alarm": bool(alarm),
        "trigger_metric": float(metric),
        "slip_location": int(slip_location),
        "slip_direction": int(slip_direction),
        "total_transmitted_symbols": int(total_symbols),
        "net_information_bits_per_symbol": float(code.k / total_symbols),
    })
    row.pop("decoded_bits", None)
    return row


def _combine_event_with_detector(
    normal: DecoderResult,
    recovery: DecoderResult | None,
    *,
    decoder_name: str,
    alarm: bool,
    metric: float,
    detector_wall: float,
) -> DecoderResult:
    normal_with_detector = replace(normal, wall_seconds=float(normal.wall_seconds + detector_wall))
    return combine_event_result(
        normal_with_detector,
        recovery,
        decoder_name=decoder_name,
        alarm=alarm,
        trigger_metric=metric,
    )


def _dqpsk_control(
    gain: np.ndarray,
    word: np.ndarray,
    n0: float,
    code: LinearCode,
    dcfg: dict[str, Any],
) -> DecoderResult:
    start = time.perf_counter()
    tx = dqpsk_encode(word)
    if gain.size < tx.size:
        raise ValueError("trace is shorter than differential packet")
    y = tx * gain[: tx.size]
    obs = dqpsk_observations(y)
    decoded = normal_code_decoder(obs, float(dcfg.get("dqpsk_n0_factor", 2.0)) * n0, code, dcfg)
    return replace(
        decoded,
        decoder="dqpsk_control",
        wall_seconds=float(time.perf_counter() - start),
        components_generated=decoded.components_generated + code.n // 2,
    )


def run_performance(
    cfg: dict[str, Any],
    codes: dict[str, LinearCode],
    thresholds: dict[str, float],
    test_no_gain: np.ndarray,
    test_one_gain: np.ndarray,
    one_locations: np.ndarray,
    one_directions: np.ndarray,
    test_other_gain: np.ndarray,
    run_dir: Path,
) -> pd.DataFrame:
    master = int(cfg["master_seed"])
    dcfg = cfg["decoders"]
    n0 = float(cfg["trace"]["n0"])
    rows: list[dict[str, Any]] = []
    frame_counter = 0

    conditions: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        ("trace_no_slip", test_no_gain, np.full(test_no_gain.shape[0], -1), np.zeros(test_no_gain.shape[0])),
        ("trace_one_slip", test_one_gain, one_locations, one_directions),
    ]
    if int(cfg["trials"].get("other_trace_frames", 0)) > 0 and test_other_gain.size:
        m = min(int(cfg["trials"]["other_trace_frames"]), test_other_gain.shape[0])
        conditions.append(("trace_other", test_other_gain[:m], np.full(m, -1), np.zeros(m)))

    for ci, (code_id, code) in enumerate(codes.items()):
        states = one_slip_hypotheses(
            code.n // 2,
            float(dcfg["conditional_slip_prior_after_alarm"]),
            (1, 3),
        )
        for condition_index, (condition, gains, locations, directions) in enumerate(conditions):
            if condition == "trace_no_slip":
                target = min(int(cfg["trials"]["test_no_slip_frames"]), gains.shape[0])
            elif condition == "trace_one_slip":
                target = min(int(cfg["trials"]["one_slip_frames"]), gains.shape[0])
            else:
                target = gains.shape[0]
            for t in range(target):
                rng = seeded_rng(master, 400, ci, condition_index, t)
                word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
                tx_int = bits_to_int(word)
                gain = gains[t]
                y = _gain_to_observation(word, gain)
                normal = normal_code_decoder(y, n0, code, dcfg)
                detector_start = time.perf_counter()
                metric = _detector_metric(y, n0, normal, code)
                detector_wall = time.perf_counter() - detector_start
                alarm = bool(metric > thresholds[code_id])
                uid = f"{code_id}_{condition}_{t}"

                rows.append(_decoder_row(
                    normal,
                    frame_uid=uid,
                    code_id=code_id,
                    code=code,
                    condition=condition,
                    tx_int=tx_int,
                    alarm=False,
                    metric=metric,
                    slip_location=int(locations[t]),
                    slip_direction=int(directions[t]),
                    total_symbols=code.n // 2,
                ))

                first = adaptive = full_osd = full_specific = pruned_specific = state_marginal = None
                if alarm:
                    first = latent_list_decode(
                        y, n0, code, states,
                        list_size=1,
                        query_cap=int(dcfg["ls_query_cap"]),
                        marginal_rescore=False,
                        score_batch_size=int(dcfg["score_batch_size"]),
                    )
                    adaptive = latent_adaptive_l2(
                        y, n0, code, states,
                        logsumexp_gain_threshold=float(dcfg["adaptive_logsumexp_gain_threshold"]),
                        query_cap=int(dcfg["adaptive_query_cap"]),
                        score_batch_size=int(dcfg["score_batch_size"]),
                    )
                    state_marginal = state_marginal_code_specific(
                        y, n0, code, states,
                        bch_chase_order=int(dcfg["state_marginal_bch_chase_order"]),
                        bch_chase_pool=int(dcfg["state_marginal_bch_chase_pool"]),
                        polar_flip_trials=int(dcfg["state_marginal_polar_flip_trials"]),
                        candidate_cap=int(dcfg["candidate_cap"]),
                        score_batch_size=int(dcfg["score_batch_size"]),
                    )
                    if condition != "trace_no_slip" and t < int(dcfg["full_osd_audit_frames"]):
                        full_osd = latent_osd_batch(
                            y, n0, code, states,
                            order=int(dcfg["full_osd_order"]),
                            pool_size=int(dcfg["full_osd_pool"]),
                            state_limit=None,
                            candidate_cap=int(dcfg["candidate_cap"]),
                            score_batch_size=int(dcfg["score_batch_size"]),
                        )
                    anchor = (
                        np.asarray(normal.decoded_bits, dtype=np.uint8)
                        if normal.decoded_bits is not None
                        else qpsk_hard_bits(y)
                    )
                    run_full_specific = condition != "trace_no_slip" and t < int(dcfg["full_specific_audit_frames"])
                    if code.family == "extended_bch":
                        if run_full_specific:
                            full_specific = bch_chase_state_sweep(
                                y, n0, code, states,
                                chase_order=int(dcfg["full_bch_chase_order"]),
                                chase_pool=int(dcfg["full_bch_chase_pool"]),
                                chase_state_limit=len(states),
                                candidate_cap=int(dcfg["candidate_cap"]),
                                score_batch_size=int(dcfg["score_batch_size"]),
                            )
                        pruned_specific = posterior_pruned_bch_sweep(
                            y, n0, code, states, anchor,
                            state_limit=int(dcfg["posterior_state_limit"]),
                            chase_order=int(dcfg["pruned_bch_chase_order"]),
                            chase_pool=int(dcfg["pruned_bch_chase_pool"]),
                            candidate_cap=int(dcfg["candidate_cap"]),
                            score_batch_size=int(dcfg["score_batch_size"]),
                        )
                    elif code.family == "polar":
                        if run_full_specific:
                            full_specific = polar_scflip_state_sweep(
                                y, n0, code, states,
                                flip_trials=int(dcfg["full_polar_flip_trials"]),
                                score_batch_size=int(dcfg["score_batch_size"]),
                            )
                        pruned_specific = posterior_pruned_polar_sweep(
                            y, n0, code, states, anchor,
                            state_limit=int(dcfg["posterior_state_limit"]),
                            flip_trials=int(dcfg["pruned_polar_flip_trials"]),
                            score_batch_size=int(dcfg["score_batch_size"]),
                        )

                event_results = [
                    _combine_event_with_detector(normal, first, decoder_name="event_ls_first", alarm=alarm, metric=metric, detector_wall=detector_wall),
                    _combine_event_with_detector(normal, adaptive, decoder_name="event_ls_adaptive_l2", alarm=alarm, metric=metric, detector_wall=detector_wall),
                    _combine_event_with_detector(normal, pruned_specific, decoder_name="event_posterior_pruned_code_specific", alarm=alarm, metric=metric, detector_wall=detector_wall),
                    _combine_event_with_detector(normal, state_marginal, decoder_name="event_state_marginal_code_specific", alarm=alarm, metric=metric, detector_wall=detector_wall),
                ]
                if condition == "trace_no_slip" or full_specific is not None:
                    event_results.append(
                        _combine_event_with_detector(normal, full_specific, decoder_name="event_full_code_specific", alarm=alarm, metric=metric, detector_wall=detector_wall)
                    )
                if full_osd is not None or (condition == "trace_no_slip" and not alarm):
                    event_results.append(
                        _combine_event_with_detector(normal, full_osd, decoder_name="event_full_state_osd", alarm=alarm, metric=metric, detector_wall=detector_wall)
                    )
                for result in event_results:
                    rows.append(_decoder_row(
                        result,
                        frame_uid=uid,
                        code_id=code_id,
                        code=code,
                        condition=condition,
                        tx_int=tx_int,
                        alarm=alarm,
                        metric=metric,
                        slip_location=int(locations[t]),
                        slip_direction=int(directions[t]),
                        total_symbols=code.n // 2,
                    ))

                if condition != "trace_other" and bool(cfg["controls"].get("run_dqpsk", True)):
                    dq = _dqpsk_control(gain, word, n0, code, dcfg)
                    rows.append(_decoder_row(
                        dq,
                        frame_uid=uid,
                        code_id=code_id,
                        code=code,
                        condition=condition,
                        tx_int=tx_int,
                        alarm=False,
                        metric=float("nan"),
                        slip_location=int(locations[t]),
                        slip_direction=int(directions[t]),
                        total_symbols=code.n // 2 + 1,
                    ))

                frame_counter += 1
                if frame_counter % int(cfg["trials"].get("checkpoint_every", 100)) == 0:
                    atomic_csv(pd.DataFrame(rows), run_dir / "performance_trials.csv")
    df = pd.DataFrame(rows)
    atomic_csv(df, run_dir / "performance_trials.csv")
    return df


def aggregate_performance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["code_id", "family", "n", "k", "rate", "condition", "decoder", "total_transmitted_symbols"]
    metrics = [
        "components_generated", "membership_queries", "latent_queues_touched", "queue_touch_fraction",
        "valid_witnesses", "unique_candidates", "complete_marginal_candidates", "state_word_metric_evals",
        "bit_metric_accumulations", "osd_reprocessings", "bch_decode_attempts", "preprocessing_state_metrics",
        "wall_seconds", "first_component_gap",
    ]
    for key, g in df.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, key))
        n = len(g)
        errors = int((~g["decoded_correct"].astype(bool)).sum())
        lo, hi = wilson_interval(errors, n)
        row.update({
            "trials": n,
            "errors": errors,
            "bler": errors / n,
            "bler_low": lo,
            "bler_high": hi,
            "alarm_rate": float(g["alarm"].astype(float).mean()),
            "cap_rate": float(g["cap_hit"].astype(float).mean()),
            "success_rate": float(g["success"].astype(float).mean()),
            "net_information_bits_per_symbol": float(g["net_information_bits_per_symbol"].iloc[0]),
        })
        for col in metrics:
            vals = pd.to_numeric(g[col], errors="coerce")
            vals = vals[np.isfinite(vals)]
            row[f"median_{col}"] = safe_quantile(vals, 0.5)
            row[f"p90_{col}"] = safe_quantile(vals, 0.9)
            row[f"p99_{col}"] = safe_quantile(vals, 0.99)
            row[f"mean_{col}"] = float(vals.mean()) if len(vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def detector_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["decoder"] == "event_ls_first"]
    rows: list[dict[str, Any]] = []
    for code_id, g in base.groupby("code_id", sort=True):
        row: dict[str, Any] = {"code_id": code_id}
        for condition in ["trace_no_slip", "trace_one_slip", "trace_other"]:
            h = g[g["condition"] == condition]
            if h.empty:
                continue
            events = int(h["alarm"].astype(bool).sum())
            lo, hi = clopper_pearson_interval(events, len(h))
            row[f"alarm_rate_{condition}"] = events / len(h)
            row[f"alarm_low_{condition}"] = lo
            row[f"alarm_high_{condition}"] = hi
            row[f"trials_{condition}"] = len(h)
        rows.append(row)
    return pd.DataFrame(rows)


def no_slip_overhead(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    no = df[df["condition"] == "trace_no_slip"]
    for code_id, g in no.groupby("code_id"):
        base = g[g["decoder"] == "normal_code_decoder"]
        if base.empty:
            continue
        bmed = safe_quantile(base["wall_seconds"], 0.5)
        bp99 = safe_quantile(base["wall_seconds"], 0.99)
        for decoder in [
            "event_ls_first", "event_ls_adaptive_l2", "event_full_code_specific",
            "event_posterior_pruned_code_specific", "event_state_marginal_code_specific",
            "event_full_state_osd", "dqpsk_control",
        ]:
            d = g[g["decoder"] == decoder]
            if d.empty:
                continue
            events = int(d["alarm"].astype(bool).sum()) if decoder.startswith("event_") else 0
            fa_lo, fa_hi = clopper_pearson_interval(events, len(d))
            rows.append({
                "code_id": code_id,
                "decoder": decoder,
                "trials": len(d),
                "false_alarm_rate": events / len(d),
                "false_alarm_low": fa_lo,
                "false_alarm_high": fa_hi,
                "median_wall_overhead_ratio": safe_quantile(d["wall_seconds"], 0.5) / max(bmed, np.finfo(float).tiny),
                "p99_wall_overhead_ratio": safe_quantile(d["wall_seconds"], 0.99) / max(bp99, np.finfo(float).tiny),
                "bler": float((~d["decoded_correct"].astype(bool)).mean()),
            })
    return pd.DataFrame(rows)


def _pair(df: pd.DataFrame, code_id: str, condition: str, a: str, b: str) -> pd.DataFrame:
    g = df[(df["code_id"] == code_id) & (df["condition"] == condition)]
    aa = g[g["decoder"] == a].set_index("frame_uid")
    bb = g[g["decoder"] == b].set_index("frame_uid")
    shared = aa.index.intersection(bb.index)
    if not len(shared):
        return pd.DataFrame()
    cols = ["decoded_correct", "cap_hit", "wall_seconds", "components_generated", "bit_metric_accumulations"]
    out = pd.DataFrame(index=shared)
    for col in cols:
        out[f"a_{col}"] = aa.loc[shared, col]
        out[f"b_{col}"] = bb.loc[shared, col]
    return out.reset_index()


def performance_comparisons(cfg: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    rng = seeded_rng(int(cfg["master_seed"]), 900)
    rows = []
    candidates = ["event_ls_first", "event_ls_adaptive_l2"]
    baselines = [
        "event_full_code_specific", "event_posterior_pruned_code_specific",
        "event_state_marginal_code_specific", "event_full_state_osd", "dqpsk_control"
    ]
    for code_id in sorted(df["code_id"].unique()):
        family = str(df[df["code_id"] == code_id]["family"].iloc[0])
        for baseline in baselines:
            for candidate in candidates:
                pair = _pair(df, code_id, "trace_one_slip", candidate, baseline)
                if pair.empty:
                    continue
                eligible = pair[~pair["a_cap_hit"].astype(bool) & ~pair["b_cap_hit"].astype(bool)]
                diff, dlo, dhi, dn = paired_error_difference_interval(
                    (~pair["a_decoded_correct"].astype(bool)).astype(int).to_numpy(),
                    (~pair["b_decoded_correct"].astype(bool)).astype(int).to_numpy(),
                    rng,
                    resamples=int(cfg["thresholds"]["bootstrap_resamples"]),
                )
                wall, wlo, whi, wn = paired_bootstrap_median_ratio(
                    pd.to_numeric(eligible["b_wall_seconds"], errors="coerce").to_numpy(),
                    pd.to_numeric(eligible["a_wall_seconds"], errors="coerce").to_numpy(),
                    rng,
                    resamples=int(cfg["thresholds"]["bootstrap_resamples"]),
                )
                comp, clo, chi, cn = paired_bootstrap_median_ratio(
                    pd.to_numeric(eligible["b_components_generated"], errors="coerce").to_numpy(),
                    np.maximum(1.0, pd.to_numeric(eligible["a_components_generated"], errors="coerce").to_numpy()),
                    rng,
                    resamples=int(cfg["thresholds"]["bootstrap_resamples"]),
                )
                aerr_vec = (~pair["a_decoded_correct"].astype(bool)).astype(int)
                berr_vec = (~pair["b_decoded_correct"].astype(bool)).astype(int)
                aerr = int(aerr_vec.sum())
                alo, ahi = wilson_interval(aerr, len(pair))
                ap99 = safe_quantile(pd.to_numeric(pair["a_wall_seconds"], errors="coerce"), 0.99)
                bp99 = safe_quantile(pd.to_numeric(pair["b_wall_seconds"], errors="coerce"), 0.99)
                rows.append({
                    "code_id": code_id,
                    "family": family,
                    "candidate": candidate,
                    "baseline": baseline,
                    "paired_trials": dn,
                    "eligible_work_pairs": min(wn, cn),
                    "candidate_bler": aerr / len(pair),
                    "candidate_bler_low": alo,
                    "candidate_bler_high": ahi,
                    "candidate_cap_rate": float(pair["a_cap_hit"].astype(float).mean()),
                    "candidate_minus_baseline_error_rate": diff,
                    "error_diff_low": dlo,
                    "error_diff_high": dhi,
                    "baseline_to_candidate_wall_median_ratio": wall,
                    "wall_ratio_low": wlo,
                    "wall_ratio_high": whi,
                    "baseline_to_candidate_component_median_ratio": comp,
                    "component_ratio_low": clo,
                    "component_ratio_high": chi,
                    "baseline_to_candidate_p99_wall_ratio": bp99 / max(ap99, np.finfo(float).tiny),
                    "mcnemar": json.dumps(mcnemar_one_sided(aerr_vec.astype(bool), berr_vec.astype(bool)), sort_keys=True),
                })
    return pd.DataFrame(rows)


def observed_physical_mixtures(cfg: dict[str, Any], df: pd.DataFrame, observed_p: float) -> pd.DataFrame:
    rows = []
    for code_id in sorted(df["code_id"].unique()):
        for decoder in sorted(df[df["code_id"] == code_id]["decoder"].unique()):
            base = df[(df["code_id"] == code_id) & (df["decoder"] == decoder)]
            no = base[base["condition"] == "trace_no_slip"]
            slip = base[base["condition"] == "trace_one_slip"]
            if no.empty or slip.empty:
                continue
            values_wall = np.concatenate([
                pd.to_numeric(no["wall_seconds"], errors="coerce").to_numpy(),
                pd.to_numeric(slip["wall_seconds"], errors="coerce").to_numpy(),
            ])
            values_comp = np.concatenate([
                pd.to_numeric(no["components_generated"], errors="coerce").to_numpy(),
                pd.to_numeric(slip["components_generated"], errors="coerce").to_numpy(),
            ])
            values_err = np.concatenate([
                (~no["decoded_correct"].astype(bool)).astype(float).to_numpy(),
                (~slip["decoded_correct"].astype(bool)).astype(float).to_numpy(),
            ])
            weights = np.concatenate([
                np.full(len(no), (1.0 - observed_p) / len(no)),
                np.full(len(slip), observed_p / len(slip)),
            ])
            rows.append({
                "code_id": code_id,
                "decoder": decoder,
                "observed_one_slip_probability": observed_p,
                "unconditional_bler": float(np.dot(values_err, weights) / weights.sum()),
                "mean_wall_seconds": float(np.dot(values_wall, weights) / weights.sum()),
                "p99_wall_seconds": weighted_quantile(values_wall, weights, 0.99),
                "p999_wall_seconds": weighted_quantile(values_wall, weights, 0.999),
                "p9999_wall_seconds": weighted_quantile(values_wall, weights, 0.9999),
                "conditional_slip_bler": float((~slip["decoded_correct"].astype(bool)).mean()),
                "conditional_no_slip_bler": float((~no["decoded_correct"].astype(bool)).mean()),
                "conditional_slip_p99_wall_seconds": safe_quantile(slip["wall_seconds"], 0.99),
                "mean_components_generated": float(np.dot(values_comp, weights) / weights.sum()),
            })
    return pd.DataFrame(rows)


def control_summary(cfg: dict[str, Any], perf: pd.DataFrame, mixtures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    overhead = float(cfg["controls"]["ideal_pilot_overhead_fraction"])
    for code_id in sorted(perf["code_id"].unique()):
        code_rows = perf[perf["code_id"] == code_id]
        k = int(code_rows["k"].iloc[0])
        data_symbols = int(code_rows["n"].iloc[0] // 2)
        no = code_rows[(code_rows["condition"] == "trace_no_slip") & (code_rows["decoder"] == "normal_code_decoder")]
        no_bler = float((~no["decoded_correct"].astype(bool)).mean()) if len(no) else float("nan")
        pilot_rate = k * (1.0 - overhead) / data_symbols
        pilot_goodput = pilot_rate * (1.0 - no_bler)
        rows.append({
            "code_id": code_id,
            "control": "ideal_pilot_phase_unwrap_bound",
            "overhead_fraction": overhead,
            "net_information_bits_per_symbol": pilot_rate,
            "unconditional_bler": no_bler,
            "goodput_information_bits_per_symbol": pilot_goodput,
            "note": "optimistic bound: assumes pilots eliminate branch slips with no extra decoder cost",
        })
        for decoder in ["event_ls_first", "event_ls_adaptive_l2", "dqpsk_control"]:
            mix = mixtures[(mixtures["code_id"] == code_id) & (mixtures["decoder"] == decoder)]
            if mix.empty:
                continue
            total_symbols = data_symbols + (1 if decoder == "dqpsk_control" else 0)
            rate = k / total_symbols
            bler = float(mix.iloc[0]["unconditional_bler"])
            rows.append({
                "code_id": code_id,
                "control": decoder,
                "overhead_fraction": 1.0 - data_symbols / total_symbols,
                "net_information_bits_per_symbol": rate,
                "unconditional_bler": bler,
                "goodput_information_bits_per_symbol": rate * (1.0 - bler),
                "note": "trace-weighted executable receiver",
            })
    return pd.DataFrame(rows)


def adjudicate(
    cfg: dict[str, Any],
    g0: dict[str, Any],
    g3p: dict[str, Any],
    claims: pd.DataFrame,
    orbit: pd.DataFrame,
    trace_summary: dict[str, Any],
    detector: pd.DataFrame,
    comparisons: pd.DataFrame,
    overhead: pd.DataFrame,
    mixtures: pd.DataFrame,
    controls: pd.DataFrame,
    unit_tests_passed: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    th = cfg["thresholds"]
    gates: list[dict[str, Any]] = []
    disposition = {str(r["claim_id"]): str(r["disposition"]) for _, r in claims.iterrows()}
    blocked = {"C01", "C02", "C03", "C04", "C10", "C11", "C12"}
    provisional = {"C05", "C06", "C07", "C08"}
    chain_ok = bool(
        g0.get("source_commit") == EXPECTED_G0_COMMIT
        and g3p.get("source_commit") == EXPECTED_G3P_COMMIT
        and g3p.get("source_verdict") == "CONTINUE_TO_FINAL_PHYSICAL_MATCHED_GATE"
        and all(disposition.get(x, "").startswith("BLOCK") for x in blocked)
        and all(disposition.get(x, "").startswith("ALLOW") for x in provisional)
    )
    gates.append({
        "gate": "G0_CHAIN_LOCK",
        "pass": chain_ok,
        "g0_commit": g0.get("source_commit"),
        "g3p_commit": g3p.get("source_commit"),
        "field_defining_authorized": False,
        "exact_route_status": "STOPPED_AS_PRIMARY_ROUTE",
    })

    observed_one = int(trace_summary.get("observed_one_slip_frames", 0))
    total_generated = int(trace_summary.get("total_frames_generated", 0))
    rate = float(trace_summary.get("one_slip_rate_point_estimate", float("nan")))
    rlo, rhi = clopper_pearson_interval(observed_one, total_generated)
    trace_pass = bool(
        observed_one >= int(th["min_observed_trace_slips"])
        and int(trace_summary.get("one_slip_collected", 0)) >= int(th["min_one_slip_test_frames"])
        and float(th["min_observed_slip_rate"]) <= rate <= float(th["max_observed_slip_rate"])
    )
    gates.append({
        "gate": "P1_STANDARD_DERIVED_TRACE",
        "pass": trace_pass,
        "measured_hardware_trace": False,
        "standard_derived_viterbi_viterbi_trace": True,
        "observed_one_slip_frames": observed_one,
        "total_frames_generated": total_generated,
        "one_slip_rate": rate,
        "one_slip_rate_low": rlo,
        "one_slip_rate_high": rhi,
        "trace_parameters": trace_summary,
    })

    structured = [str(x["id"]) for x in cfg["codes"] if bool(x.get("structured", True))]
    orbit_target = orbit[orbit["code_id"].isin(structured)]
    orbit_pass = bool(len(orbit_target) == len(structured) and orbit_target["orbit_safe"].astype(bool).all())
    gates.append({
        "gate": "P2_ORBIT_SAFETY",
        "pass": orbit_pass,
        "structured_codes": structured,
        "unsafe": orbit_target[~orbit_target["orbit_safe"].astype(bool)]["code_id"].astype(str).tolist(),
    })

    detector_details = []
    detector_good: set[str] = set()
    if not detector.empty:
        det = detector.set_index("code_id")
        for code_id in structured:
            if code_id not in det.index:
                continue
            row = det.loc[code_id]
            fa_hi = float(row.get("alarm_high_trace_no_slip", 1.0))
            det_lo = float(row.get("alarm_low_trace_one_slip", 0.0))
            qualifies = bool(
                fa_hi <= float(th["max_false_alarm_upper"])
                and det_lo >= float(th["min_slip_detection_lower"])
            )
            if qualifies:
                detector_good.add(code_id)
            detector_details.append({
                "code_id": code_id,
                "false_alarm_rate": float(row.get("alarm_rate_trace_no_slip", np.nan)),
                "false_alarm_upper": fa_hi,
                "one_slip_detection_rate": float(row.get("alarm_rate_trace_one_slip", np.nan)),
                "one_slip_detection_lower": det_lo,
                "qualifies": qualifies,
            })
    detector_pass = detector_good == set(structured)
    gates.append({"gate": "P3_EVENT_TRIGGER", "pass": detector_pass, "details": detector_details})

    candidate_modes = ["event_ls_first", "event_ls_adaptive_l2"]
    baseline_modes = [
        "event_full_code_specific",
        "event_posterior_pruned_code_specific",
        "event_state_marginal_code_specific",
    ]
    recovery_details = []
    recovery_modes: dict[str, list[str]] = {}
    recovery_routes: dict[tuple[str, str, str], str] = {}
    for code_id in structured:
        good_modes: list[str] = []
        for candidate in candidate_modes:
            rows = comparisons[
                (comparisons["code_id"] == code_id)
                & (comparisons["candidate"] == candidate)
                & comparisons["baseline"].isin(baseline_modes)
            ]
            if len(rows) < len(baseline_modes):
                continue
            all_ok = True
            per_baseline = []
            for _, row in rows.iterrows():
                enough = bool(
                    int(row["paired_trials"]) >= int(th["min_paired_slip_trials"])
                    and int(row["eligible_work_pairs"]) >= int(th["min_eligible_work_pairs"])
                    and float(row["candidate_bler_high"]) <= float(th["max_candidate_bler_upper"])
                    and float(row["candidate_cap_rate"]) <= float(th["max_candidate_cap_rate"])
                )
                speed_win = bool(
                    float(row["error_diff_high"]) <= float(th["max_candidate_bler_excess"])
                    and float(row["baseline_to_candidate_wall_median_ratio"]) >= float(th["min_median_wall_ratio"])
                    and float(row["wall_ratio_low"]) > 1.0
                    and float(row["baseline_to_candidate_p99_wall_ratio"]) >= float(th["min_p99_wall_ratio"])
                )
                reliability_win = bool(
                    float(row["error_diff_high"]) <= -float(th["min_material_bler_improvement"])
                    and float(row["baseline_to_candidate_wall_median_ratio"]) >= 1.0 / float(th["max_wall_slowdown_for_reliability_win"])
                )
                route = "speed" if enough and speed_win else ("reliability" if enough and reliability_win else "none")
                qualifies = route != "none"
                recovery_routes[(code_id, candidate, str(row["baseline"]))] = route
                all_ok = all_ok and qualifies
                per_baseline.append({**row.to_dict(), "qualifies": qualifies, "qualification_route": route})
            recovery_details.extend(per_baseline)
            if all_ok:
                good_modes.append(candidate)
        recovery_modes[code_id] = good_modes
    recovery_pass_codes = [k for k, v in recovery_modes.items() if v]
    recovery_pass = set(recovery_pass_codes) == set(structured)
    gates.append({
        "gate": "P4_STRONG_MATCHED_RECOVERY",
        "pass": recovery_pass,
        "qualifying_codes": recovery_pass_codes,
        "qualifying_modes_by_code": recovery_modes,
        "details": recovery_details,
    })

    no_slip_details = []
    no_slip_modes: dict[str, list[str]] = {}
    for code_id in structured:
        good: list[str] = []
        for candidate in candidate_modes:
            rows = overhead[(overhead["code_id"] == code_id) & (overhead["decoder"] == candidate)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            qualifies = bool(
                float(row["false_alarm_high"]) <= float(th["max_false_alarm_upper"])
                and float(row["median_wall_overhead_ratio"]) <= float(th["max_no_slip_median_overhead_ratio"])
                and float(row["p99_wall_overhead_ratio"]) <= float(th["max_no_slip_p99_overhead_ratio"])
            )
            no_slip_details.append({**row.to_dict(), "qualifies": qualifies})
            if qualifies:
                good.append(candidate)
        no_slip_modes[code_id] = good
    no_slip_pass = all(no_slip_modes.get(c) for c in structured)
    gates.append({
        "gate": "P5_NO_SLIP_NEUTRALITY",
        "pass": no_slip_pass,
        "qualifying_modes_by_code": no_slip_modes,
        "details": no_slip_details,
    })

    physical_details = []
    physical_modes: dict[str, list[str]] = {}
    for code_id in structured:
        candidates = set(recovery_modes.get(code_id, [])) & set(no_slip_modes.get(code_id, []))
        good: list[str] = []
        for candidate in sorted(candidates):
            cand = mixtures[(mixtures["code_id"] == code_id) & (mixtures["decoder"] == candidate)]
            if cand.empty:
                continue
            c = cand.iloc[0]
            baseline_checks = []
            all_ok = True
            for baseline in baseline_modes:
                b = mixtures[(mixtures["code_id"] == code_id) & (mixtures["decoder"] == baseline)]
                if b.empty:
                    all_ok = False
                    continue
                bb = b.iloc[0]
                ratio = float(bb["conditional_slip_p99_wall_seconds"]) / max(
                    float(c["conditional_slip_p99_wall_seconds"]), np.finfo(float).tiny
                )
                route = recovery_routes.get((code_id, candidate, baseline), "none")
                if route == "speed":
                    qualifies = bool(
                        ratio >= float(th["min_physical_conditional_p99_ratio"])
                        and float(c["unconditional_bler"] - bb["unconditional_bler"]) <= float(th["max_candidate_bler_excess"])
                    )
                elif route == "reliability":
                    qualifies = bool(
                        float(c["conditional_slip_bler"]) + float(th["min_material_bler_improvement"])
                        <= float(bb["conditional_slip_bler"])
                        and ratio >= 1.0 / float(th["max_wall_slowdown_for_reliability_win"])
                    )
                else:
                    qualifies = False
                all_ok = all_ok and qualifies
                baseline_checks.append({
                    "baseline": baseline,
                    "qualification_route": route,
                    "conditional_p99_wall_ratio": ratio,
                    "candidate_conditional_slip_bler": float(c["conditional_slip_bler"]),
                    "baseline_conditional_slip_bler": float(bb["conditional_slip_bler"]),
                    "qualifies": qualifies,
                })
            physical_details.append({
                "code_id": code_id,
                "candidate": candidate,
                "observed_one_slip_probability": float(c["observed_one_slip_probability"]),
                "unconditional_bler": float(c["unconditional_bler"]),
                "mean_wall_seconds": float(c["mean_wall_seconds"]),
                "baseline_checks": baseline_checks,
                "qualifies": all_ok,
            })
            if all_ok:
                good.append(candidate)
        physical_modes[code_id] = good
    physical_pass = all(physical_modes.get(c) for c in structured)
    gates.append({
        "gate": "P6_OBSERVED_TRACE_MIXTURE",
        "pass": physical_pass,
        "measured_hardware_trace": False,
        "qualifying_modes_by_code": physical_modes,
        "details": physical_details,
    })

    control_details = []
    control_pass_codes = []
    for code_id in structured:
        pilot = controls[(controls["code_id"] == code_id) & (controls["control"] == "ideal_pilot_phase_unwrap_bound")]
        candidate_rows = controls[(controls["code_id"] == code_id) & controls["control"].isin(physical_modes.get(code_id, []))]
        if pilot.empty or candidate_rows.empty:
            continue
        pgood = float(pilot.iloc[0]["goodput_information_bits_per_symbol"])
        best = candidate_rows.sort_values("goodput_information_bits_per_symbol", ascending=False).iloc[0]
        advantage = float(best["goodput_information_bits_per_symbol"] / max(pgood, np.finfo(float).tiny))
        qualifies = advantage >= float(th["min_goodput_ratio_vs_ideal_pilot"])
        control_details.append({
            "code_id": code_id,
            "candidate": str(best["control"]),
            "candidate_goodput": float(best["goodput_information_bits_per_symbol"]),
            "ideal_pilot_goodput": pgood,
            "goodput_ratio": advantage,
            "qualifies": qualifies,
        })
        if qualifies:
            control_pass_codes.append(code_id)
    controls_pass = set(control_pass_codes) == set(structured)
    gates.append({
        "gate": "P7_RATE_ACCOUNTED_CONTROLS",
        "pass": controls_pass,
        "ideal_pilot_is_optimistic_bound": True,
        "qualifying_codes": control_pass_codes,
        "details": control_details,
    })

    reproducible = bool(unit_tests_passed)
    gates.append({"gate": "P8_REPRODUCIBILITY", "pass": reproducible, "validator_required": True})

    profile = cfg["profile"]
    if profile == "smoke":
        verdict = "INCONCLUSIVE_SMOKE_ONLY"
        rationale = "Smoke mode validates software and schemas only."
    elif not chain_ok:
        verdict = "STOP_CLAIM_CHAIN_VIOLATION"
        rationale = "The narrowed claim chain was altered or the required commits are missing."
    elif not trace_pass:
        verdict = "INCONCLUSIVE_TRACE_EVENT_COUNT"
        rationale = "The standard-derived carrier-recovery trace did not produce enough defensible one-slip events."
    elif not orbit_pass:
        verdict = "STOP_CODE_IDENTIFIABILITY"
        rationale = "One or more structured codes failed the frozen orbit-safety screen."
    elif recovery_pass and no_slip_pass and physical_pass and controls_pass:
        verdict = "READY_FOR_FOCUSED_SUBSTANTIAL_PAPER_PROGRAM"
        rationale = "The narrowed event-triggered LS route survived a standard-derived Viterbi--Viterbi trace, strengthened code-specific and posterior-pruned baselines, normal-frame overhead, and rate-accounted controls."
    elif recovery_pass and physical_pass and not detector_pass:
        verdict = "CONTINUE_WITH_EXTERNAL_ALARM_ONLY"
        rationale = "Conditional recovery remains favorable, but the internal code-residual trigger is not sufficiently controlled."
    elif any(recovery_modes.get(c) for c in structured):
        verdict = "NARROW_TO_ONE_STRUCTURED_CODE_PAPER"
        rationale = "Only a subset of structured-code/application cells survived the final matched gate."
    else:
        verdict = "STOP_APPROXIMATE_ROUTE"
        rationale = "The narrowed LS-FV/LS-A2 receiver did not survive the strongest corrected matched baselines."

    result = {
        "verdict": verdict,
        "rationale": rationale,
        "profile": profile,
        "field_defining_verdict_authorized": False,
        "exact_foundational_route": "STOPPED",
        "practical_exact_certificate_route": "STOPPED",
        "focused_substantial_paper_authorized": verdict == "READY_FOR_FOCUSED_SUBSTANTIAL_PAPER_PROGRAM",
        "hardware_or_measured_trace_claim_authorized": False,
        "gates": gates,
        "next_if_ready": [
            "write a narrowly scoped paper around LS-FV/LS-A2, actual noisy latent rank, orbit screening, and trace-anchored tail complexity",
            "run one independent external reproduction with the frozen package",
            "obtain an experimental/hardware trace before claiming deployment readiness",
            "retain C05-C08 provisional novelty language and do not revive the exact-foundation claim",
        ],
    }
    return result, pd.DataFrame(gates)


def write_figures(run_dir: Path, trace_summary: dict[str, Any], detector: pd.DataFrame, comparisons: pd.DataFrame, overhead: pd.DataFrame, controls: pd.DataFrame) -> None:
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    if not detector.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        x = np.arange(len(detector))
        ax.bar(x - .18, detector.get("alarm_rate_trace_no_slip", 0), width=.36, label="no slip")
        ax.bar(x + .18, detector.get("alarm_rate_trace_one_slip", 0), width=.36, label="one slip")
        ax.set_xticks(x)
        ax.set_xticklabels(detector["code_id"].astype(str))
        ax.set_ylabel("Alarm rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "trace_event_trigger.png", dpi=180)
        plt.close(fig)
    if not comparisons.empty:
        d = comparisons[comparisons["baseline"].isin([
            "event_full_code_specific", "event_posterior_pruned_code_specific",
            "event_state_marginal_code_specific",
        ])]
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        ax.scatter(d["candidate_minus_baseline_error_rate"], d["baseline_to_candidate_p99_wall_ratio"])
        ax.axvline(0.0, linewidth=.8)
        ax.axhline(1.0, linewidth=.8)
        ax.set_yscale("log")
        ax.set_xlabel("LS error rate minus baseline error rate")
        ax.set_ylabel("Baseline / LS conditional p99 wall time")
        for _, r in d.iterrows():
            ax.annotate(f"{r['code_id']}:{r['candidate']}:{r['baseline']}", (r["candidate_minus_baseline_error_rate"], r["baseline_to_candidate_p99_wall_ratio"]), fontsize=6)
        fig.tight_layout()
        fig.savefig(figures / "strong_matched_frontier.png", dpi=180)
        plt.close(fig)
    if not overhead.empty:
        d = overhead[overhead["decoder"].isin(["event_ls_first", "event_ls_adaptive_l2"])]
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        x = np.arange(len(d))
        ax.bar(x, d["p99_wall_overhead_ratio"])
        ax.set_xticks(x)
        ax.set_xticklabels((d["code_id"] + ":" + d["decoder"]).tolist(), rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("No-slip p99 wall overhead ratio")
        fig.tight_layout()
        fig.savefig(figures / "no_slip_tail_overhead.png", dpi=180)
        plt.close(fig)
    if not controls.empty:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        d = controls.copy()
        labels = (d["code_id"].astype(str) + ":" + d["control"].astype(str)).tolist()
        x = np.arange(len(d))
        ax.bar(x, d["goodput_information_bits_per_symbol"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
        ax.set_ylabel("Trace-weighted goodput (information bits/symbol)")
        fig.tight_layout()
        fig.savefig(figures / "rate_accounted_controls.png", dpi=180)
        plt.close(fig)
    write_json(figures / "TRACE_SUMMARY_COPY.json", trace_summary)


def write_report(run_dir: Path, verdict: dict[str, Any], trace_summary: dict[str, Any]) -> None:
    gates = "\n".join(f"- **{g['gate']}**: {'PASS' if g['pass'] else 'FAIL'}" for g in verdict["gates"])
    text = f"""# LS-GRAND G4 trace-anchored matched-gate report

## Classification

**{verdict['verdict']}**

{verdict['rationale']}

This gate cannot revive the rejected exact-foundational route and cannot authorize
a field-defining claim.  Its strongest possible positive result is authorization
to develop a narrowly scoped substantial paper around event-triggered LS-FV/LS-A2.

## Gate status

{gates}

## Physical provenance

The executable trace is standard-derived rather than measured hardware data.  It
uses a QPSK Wiener laser-phase model and causal fourth-power Viterbi--Viterbi
carrier recovery.  One-slip frames are produced endogenously by the carrier
recovery; no discrete slip is forced into the final trace.

- Total carrier-recovery frames generated: {trace_summary.get('total_frames_generated')}
- Observed one-slip frames: {trace_summary.get('observed_one_slip_frames')}
- Observed one-slip rate: {trace_summary.get('one_slip_rate_point_estimate')}
- Symbol rate: {trace_summary.get('symbol_rate_baud')}
- Combined linewidth: {trace_summary.get('combined_linewidth_hz')}
- VV window: {trace_summary.get('vv_window')}

## Corrected baseline status

The earlier precursor's top-eight state OSD result is not treated as an all-state
baseline.  This gate includes full-state OSD audits, all-state code-specific
recovery, and a faster code-aided posterior-pruned code-specific competitor.  A
positive verdict requires LS to survive both full and pruned code-specific
baselines in both structured code families.

## Rate controls

The gate includes executable differential-QPSK control and an optimistic
literature-anchored pilot-overhead bound.  The pilot bound assumes perfect cycle
slip removal and is deliberately favorable to the conventional alternative.

## Claim discipline

- C01/C02/C03/C04/C10/C11/C12 remain blocked.
- C05--C08 remain provisional narrow claims.
- Patent freedom to operate is not determined.
- Measured deployment readiness is not authorized.
"""
    (run_dir / "FINAL_G4_REPORT.md").write_text(text)


def environment_manifest(package_root: Path, run_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    def capture(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=package_root, text=True, stderr=subprocess.STDOUT).strip()
        except Exception:
            return None
    return {
        "created_utc": utc_stamp(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_commit_at_run_start": capture(["git", "rev-parse", "HEAD"]),
        "git_branch": capture(["git", "branch", "--show-current"]),
        "git_remote": capture(["git", "remote", "get-url", "origin"]),
        "package_root": str(package_root.resolve()),
        "run_dir": str(run_dir.resolve()),
        "config": cfg,
    }


def run_campaign(package_root: Path, config_path: Path, output_root: Path, run_name: str, *, unit_tests_passed: bool) -> Path:
    cfg = load_config(config_path)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "RUN_STATE.json", {"status": "RUNNING", "profile": cfg["profile"], "run_name": run_name})
    write_json(run_dir / "FROZEN_CONFIG.json", cfg)
    g0, g3p, claims = load_chain(package_root)
    write_json(run_dir / "G0_CHAIN_LOCK.json", g0)
    write_json(run_dir / "G3P_CHAIN_LOCK.json", g3p)
    atomic_csv(claims, run_dir / "G0_CLAIM_MATRIX_FROZEN.csv")

    codes, orbit, _ = build_screened_codes(cfg, run_dir)

    trace_cfg = cfg["trace"]
    calibration_target = int(cfg["detector"]["calibration_frames"])
    test_no_target = int(cfg["trials"]["test_no_slip_frames"])
    one_target = int(cfg["trials"]["one_slip_frames"])
    other_target = int(cfg["trials"].get("other_trace_frames", 0))
    pool: TracePool = collect_trace_pool(
        seeded_rng(int(cfg["master_seed"]), 200),
        no_slip_target=calibration_target + test_no_target,
        one_slip_target=one_target,
        other_target=other_target,
        batch_frames=int(trace_cfg["batch_frames"]),
        max_total_frames=int(trace_cfg["max_total_frames"]),
        warmup_symbols=int(trace_cfg["warmup_symbols"]),
        payload_symbols=int(trace_cfg["payload_symbols"]),
        symbol_rate_baud=float(trace_cfg["symbol_rate_baud"]),
        combined_linewidth_hz=float(trace_cfg["combined_linewidth_hz"]),
        snr_db=float(trace_cfg["cpe_snr_db"]),
        vv_window=int(trace_cfg["vv_window"]),
    )
    rate_lo, rate_hi = clopper_pearson_interval(
        int(pool.summary["observed_one_slip_frames"]), int(pool.summary["total_frames_generated"])
    )
    trace_summary = {**pool.summary, "one_slip_rate_low": rate_lo, "one_slip_rate_high": rate_hi,
                     "source_anchor": cfg["trace"].get("source_anchor")}
    write_json(run_dir / "TRACE_PROVENANCE_AND_SUMMARY.json", trace_summary)

    calibration_gain = pool.gain_no_slip[:calibration_target]
    test_no_gain = pool.gain_no_slip[calibration_target : calibration_target + test_no_target]
    thresholds, _ = calibrate_detector(cfg, codes, calibration_gain, run_dir)
    perf = run_performance(
        cfg, codes, thresholds,
        test_no_gain,
        pool.gain_one_slip[:one_target],
        pool.one_slip_locations[:one_target],
        pool.one_slip_directions[:one_target],
        pool.gain_other[:other_target],
        run_dir,
    )
    agg = aggregate_performance(perf)
    det = detector_summary(perf)
    overhead = no_slip_overhead(perf)
    cmp = performance_comparisons(cfg, perf)
    mixtures = observed_physical_mixtures(cfg, perf, float(trace_summary["one_slip_rate_point_estimate"]))
    controls = control_summary(cfg, perf, mixtures)
    atomic_csv(agg, run_dir / "performance_aggregate.csv")
    atomic_csv(det, run_dir / "detector_summary.csv")
    atomic_csv(overhead, run_dir / "no_slip_overhead.csv")
    atomic_csv(cmp, run_dir / "performance_comparisons.csv")
    atomic_csv(mixtures, run_dir / "observed_physical_mixture.csv")
    atomic_csv(controls, run_dir / "rate_accounted_controls.csv")

    verdict, gate_df = adjudicate(
        cfg, g0, g3p, claims, orbit, trace_summary, det, cmp, overhead, mixtures, controls, unit_tests_passed
    )
    write_json(run_dir / "FINAL_G4_VERDICT.json", verdict)
    atomic_csv(gate_df, run_dir / "gate_status.csv")
    write_report(run_dir, verdict, trace_summary)
    write_figures(run_dir, trace_summary, det, cmp, overhead, controls)
    write_json(run_dir / "REPRODUCIBILITY_MANIFEST.json", environment_manifest(package_root, run_dir, cfg))
    write_json(run_dir / "RUN_STATE.json", {"status": "COMPLETE", "profile": cfg["profile"], "run_name": run_name, "verdict": verdict["verdict"]})

    manifest = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in {"RESULT_SHA256_MANIFEST.json", "VALIDATION_REPORT.json"}:
            rel = str(path.relative_to(run_dir))
            manifest[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    write_json(run_dir / "RESULT_SHA256_MANIFEST.json", manifest)
    return run_dir
