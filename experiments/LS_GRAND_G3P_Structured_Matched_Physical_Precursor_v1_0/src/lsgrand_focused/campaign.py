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
from .channel import (
    one_slip_hypotheses,
    sample_slip_path,
    simulate_residual_slip_qpsk,
)
from .decoders import (
    DecoderResult,
    bch_chase_state_sweep,
    combine_event_result,
    latent_adaptive_l2,
    latent_list_decode,
    latent_osd_batch,
    normalized_no_slip_residual,
    polar_scflip_state_sweep,
)
from .gf2 import LinearCode, bits_to_int
from .stats import (
    mcnemar_one_sided,
    paired_bootstrap_median_ratio,
    paired_error_difference_interval,
    safe_quantile,
    weighted_quantile,
    wilson_interval,
)
from .structured_codes import build_frozen_code


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
    required = {"profile", "master_seed", "codes", "channel", "detector", "decoders", "trials", "physical_priors", "thresholds"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"configuration missing keys: {sorted(missing)}")
    return cfg


def load_novelty_matrix(package_root: Path) -> pd.DataFrame:
    path = package_root / "docs" / "02_CLAIM_BY_CLAIM_NOVELTY_MATRIX.csv"
    df = pd.read_csv(path)
    required = {"claim_id", "claim", "closest_primary_source", "classification", "disposition", "scope_consequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"claim matrix missing columns: {sorted(missing)}")
    return df


def load_g0_chain_lock(package_root: Path) -> dict[str, Any]:
    path = package_root / "data" / "G0_CHAIN_LOCK.json"
    if not path.is_file():
        raise ValueError(f"missing G0 chain lock: {path}")
    payload = json.loads(path.read_text())
    required = {
        "source_commit", "source_verdict", "foundational_exact_route",
        "current_exact_certificate_route", "narrow_approximate_route",
        "field_defining_program", "frozen_practical_focus",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"G0 chain lock missing keys: {sorted(missing)}")
    return payload


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
            max_attempts=int(spec.get("max_interleaver_attempts", 200)),
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
        row = {
            "code_id": code_id,
            "base_name": base.name,
            "screened_name": screened.name,
            "family": screened.family,
            "structured": bool(spec.get("structured", True)),
            "n": screened.n,
            "k": screened.k,
            "rate": screened.rate,
            "redundancy": screened.redundancy,
            "snr_db": float(spec["snr_db"]),
            "interleaved": bool(summary["interleaved"]),
            "interleaver_attempt": int(summary["attempt"]),
            "orbit_safe": bool(summary["orbit_safe"]),
            "max_collision_fraction": float(summary["max_collision_fraction"]),
            "mean_collision_fraction": float(summary["mean_collision_fraction"]),
            "union_bound": float(summary["union_bound"]),
            "dangerous_transform_count": int(summary["dangerous_transform_count"]),
            "worst_transform": summary["worst_transform"],
        }
        summaries.append(row)
        for record in summary["records"]:
            transforms.append({"code_id": code_id, **record})
    summary_df = pd.DataFrame(summaries)
    transform_df = pd.DataFrame(transforms)
    atomic_csv(summary_df, run_dir / "code_orbit_summary.csv")
    atomic_csv(transform_df, run_dir / "code_orbit_transforms.csv")
    return codes, summary_df, transform_df


def _channel_sample(
    cfg: dict[str, Any],
    code: LinearCode,
    snr_db: float,
    rng: np.random.Generator,
    condition: str,
) -> tuple[np.ndarray, Any, str]:
    word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
    if condition == "no_slip":
        path, label = sample_slip_path(code.n // 2, rng, slip=False)
    elif condition == "slip_early":
        path, label = sample_slip_path(code.n // 2, rng, slip=True, location_mode="early")
    elif condition == "slip_uniform":
        path, label = sample_slip_path(code.n // 2, rng, slip=True, location_mode="uniform")
    elif condition == "slip_late":
        path, label = sample_slip_path(code.n // 2, rng, slip=True, location_mode="late")
    else:
        raise ValueError(f"unknown condition: {condition}")
    ch = cfg["channel"]
    sample = simulate_residual_slip_qpsk(
        word,
        path,
        snr_db,
        rng,
        initial_phase_std_deg=float(ch["initial_phase_std_deg"]),
        innovation_phase_std_deg=float(ch["innovation_phase_std_deg"]),
        frequency_offset_std_deg_per_symbol=float(ch.get("frequency_offset_std_deg_per_symbol", 0.0)),
        label=label,
    )
    return word, sample, label


def no_slip_osd(y: np.ndarray, n0: float, code: LinearCode, decoder_cfg: dict[str, Any]) -> DecoderResult:
    states = one_slip_hypotheses(code.n // 2, 0.0, (1, 3))
    result = latent_osd_batch(
        y,
        n0,
        code,
        states,
        order=int(decoder_cfg["osd_order"]),
        pool_size=int(decoder_cfg["osd_pool_size"]),
        state_limit=1,
        candidate_cap=int(decoder_cfg["osd_candidate_cap"]),
        score_batch_size=int(decoder_cfg["score_batch_size"]),
    )
    return replace(result, decoder="no_slip_osd", alarm=False)


def normal_code_decoder(y: np.ndarray, n0: float, code: LinearCode, decoder_cfg: dict[str, Any]) -> DecoderResult:
    """Frozen normal-path decoder used by every event architecture.

    Structured codes receive their code-specific decoder even on ordinary
    no-slip frames.  The random-linear control uses one-state OSD.  Sharing this
    normal path prevents the latent receiver from gaining by comparing against a
    needlessly expensive or mismatched conventional front end.
    """
    states = one_slip_hypotheses(code.n // 2, 0.0, (1, 3))
    if code.family == "extended_bch":
        result = bch_chase_state_sweep(
            y, n0, code, states,
            chase_order=int(decoder_cfg["bch_chase_order"]),
            chase_pool=int(decoder_cfg["bch_chase_pool"]),
            chase_state_limit=1,
            candidate_cap=int(decoder_cfg["osd_candidate_cap"]),
            score_batch_size=int(decoder_cfg["score_batch_size"]),
        )
    elif code.family == "polar":
        result = polar_scflip_state_sweep(
            y, n0, code, states,
            flip_trials=int(decoder_cfg["polar_flip_trials"]),
            score_batch_size=int(decoder_cfg["score_batch_size"]),
        )
    else:
        result = no_slip_osd(y, n0, code, decoder_cfg)
    return replace(result, decoder="normal_code_decoder", alarm=False)


def calibrate_detector(
    cfg: dict[str, Any],
    codes: dict[str, LinearCode],
    orbit_df: pd.DataFrame,
    run_dir: Path,
) -> tuple[dict[str, float], pd.DataFrame]:
    master = int(cfg["master_seed"])
    rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    trials = int(cfg["detector"]["calibration_no_slip_frames"])
    target_fa = float(cfg["detector"]["target_false_alarm_rate"])
    decoder_cfg = cfg["decoders"]
    spec_map = {str(x["id"]): x for x in cfg["codes"]}
    for ci, (code_id, code) in enumerate(codes.items()):
        snr_db = float(spec_map[code_id]["snr_db"])
        metrics: list[float] = []
        for t in range(trials):
            rng = seeded_rng(master, 200, ci, t)
            word, sample, _ = _channel_sample(cfg, code, snr_db, rng, "no_slip")
            decoded = normal_code_decoder(sample.y, sample.n0, code, decoder_cfg)
            decoded_bits = np.asarray(decoded.decoded_bits, dtype=np.uint8) if decoded.decoded_bits is not None else np.zeros(code.n, dtype=np.uint8)
            metric = normalized_no_slip_residual(sample.y, sample.n0, decoded_bits)
            metrics.append(metric)
            rows.append({
                "code_id": code_id,
                "trial": t,
                "snr_db": snr_db,
                "trigger_metric": metric,
                "decoded_correct": decoded.decoded_int == bits_to_int(word),
                "wall_seconds": decoded.wall_seconds,
            })
        threshold = float(np.quantile(np.asarray(metrics), 1.0 - target_fa, method="higher"))
        thresholds[code_id] = threshold
    df = pd.DataFrame(rows)
    atomic_csv(df, run_dir / "detector_calibration.csv")
    write_json(run_dir / "detector_thresholds.json", thresholds)
    return thresholds, df


def _add_detector_cost(result: DecoderResult, detector_wall: float) -> DecoderResult:
    return replace(result, wall_seconds=float(result.wall_seconds + detector_wall))


def decoder_row(
    result: DecoderResult,
    *,
    frame_uid: str,
    code_id: str,
    code: LinearCode,
    condition: str,
    tx_int: int,
    snr_db: float,
    true_label: str,
    threshold: float,
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
        "snr_db": snr_db,
        "true_label": true_label,
        "transmitted_int": tx_int,
        "decoded_correct": result.decoded_int == tx_int,
        "trigger_threshold": threshold,
    })
    row.pop("decoded_bits", None)
    return row


def run_performance(
    cfg: dict[str, Any],
    codes: dict[str, LinearCode],
    thresholds: dict[str, float],
    run_dir: Path,
) -> pd.DataFrame:
    master = int(cfg["master_seed"])
    rows: list[dict[str, Any]] = []
    spec_map = {str(x["id"]): x for x in cfg["codes"]}
    decoder_cfg = cfg["decoders"]
    conditions = list(cfg["trials"]["conditions"])
    frame_counter = 0
    for ci, (code_id, code) in enumerate(codes.items()):
        snr_db = float(spec_map[code_id]["snr_db"])
        recovery_states = one_slip_hypotheses(
            code.n // 2,
            float(decoder_cfg["conditional_slip_prior_after_alarm"]),
            tuple(int(x) for x in cfg["channel"]["directions"]),
        )
        for condition_index, condition in enumerate(conditions):
            frames_by_condition = cfg["trials"].get("frames_by_condition", {})
            trials = int(frames_by_condition.get(condition, cfg["trials"]["frames_per_condition"]))
            for t in range(trials):
                rng = seeded_rng(master, 300, ci, condition_index, t)
                word, sample, label = _channel_sample(cfg, code, snr_db, rng, condition)
                tx = bits_to_int(word)
                # Keep the ordinary no-slip decoder as an independent baseline.
                # Event-triggered receivers pay the additional detector cost; the
                # baseline does not.  This avoids a trivially favorable normal-frame
                # overhead ratio.
                no_slip_baseline = normal_code_decoder(sample.y, sample.n0, code, decoder_cfg)
                detector_start = time.perf_counter()
                if no_slip_baseline.decoded_bits is None:
                    metric = float("inf")
                else:
                    metric = normalized_no_slip_residual(
                        sample.y, sample.n0, np.asarray(no_slip_baseline.decoded_bits, dtype=np.uint8)
                    )
                alarm = bool(metric > thresholds[code_id])
                detector_wall = time.perf_counter() - detector_start
                no_slip_event = _add_detector_cost(no_slip_baseline, detector_wall)
                frame_uid = f"{code_id}_{condition}_t{t}"
                rows.append(decoder_row(
                    no_slip_baseline,
                    frame_uid=frame_uid,
                    code_id=code_id,
                    code=code,
                    condition=condition,
                    tx_int=tx,
                    snr_db=snr_db,
                    true_label=label,
                    threshold=thresholds[code_id],
                ))

                first = adaptive = l2 = state_osd = chase = polar_specific = None
                if alarm:
                    first = latent_list_decode(
                        sample.y, sample.n0, code, recovery_states,
                        list_size=1,
                        query_cap=int(decoder_cfg["ls_query_cap"]),
                        marginal_rescore=False,
                        score_batch_size=int(decoder_cfg["score_batch_size"]),
                    )
                    adaptive = latent_adaptive_l2(
                        sample.y, sample.n0, code, recovery_states,
                        logsumexp_gain_threshold=float(decoder_cfg["adaptive_logsumexp_gain_threshold"]),
                        query_cap=int(decoder_cfg.get("adaptive_query_cap", decoder_cfg["ls_query_cap"])),
                        score_batch_size=int(decoder_cfg["score_batch_size"]),
                    )
                    if bool(decoder_cfg.get("run_fixed_l2", True)):
                        l2 = latent_list_decode(
                            sample.y, sample.n0, code, recovery_states,
                            list_size=2,
                            query_cap=int(decoder_cfg["ls_query_cap"]),
                            marginal_rescore=True,
                            score_batch_size=int(decoder_cfg["score_batch_size"]),
                        )
                    state_osd = latent_osd_batch(
                        sample.y, sample.n0, code, recovery_states,
                        order=int(decoder_cfg["osd_order"]),
                        pool_size=int(decoder_cfg["osd_pool_size"]),
                        state_limit=int(decoder_cfg["osd_state_limit_after_alarm"]),
                        candidate_cap=int(decoder_cfg["osd_candidate_cap"]),
                        score_batch_size=int(decoder_cfg["score_batch_size"]),
                    )
                    if code.family == "extended_bch" and bool(decoder_cfg.get("run_bch_chase", True)):
                        chase = bch_chase_state_sweep(
                            sample.y, sample.n0, code, recovery_states,
                            chase_order=int(decoder_cfg["bch_chase_order"]),
                            chase_pool=int(decoder_cfg["bch_chase_pool"]),
                            chase_state_limit=int(decoder_cfg["bch_chase_state_limit"]),
                            candidate_cap=int(decoder_cfg["osd_candidate_cap"]),
                            score_batch_size=int(decoder_cfg["score_batch_size"]),
                        )
                    if code.family == "polar" and bool(decoder_cfg.get("run_polar_scflip", True)):
                        polar_specific = polar_scflip_state_sweep(
                            sample.y, sample.n0, code, recovery_states,
                            flip_trials=int(decoder_cfg["polar_flip_trials"]),
                            score_batch_size=int(decoder_cfg["score_batch_size"]),
                        )

                event_results = [
                    combine_event_result(no_slip_event, first, decoder_name="event_ls_first", alarm=alarm, trigger_metric=metric),
                    combine_event_result(no_slip_event, adaptive, decoder_name="event_ls_adaptive_l2", alarm=alarm, trigger_metric=metric),
                    combine_event_result(no_slip_event, state_osd, decoder_name="event_state_sweep_osd", alarm=alarm, trigger_metric=metric),
                ]
                if bool(decoder_cfg.get("run_fixed_l2", False)):
                    event_results.append(
                        combine_event_result(no_slip_event, l2, decoder_name="event_ls_fixed_l2", alarm=alarm, trigger_metric=metric)
                    )
                if code.family == "extended_bch":
                    event_results.append(
                        combine_event_result(no_slip_event, chase, decoder_name="event_state_sweep_chase_bch", alarm=alarm, trigger_metric=metric)
                    )
                if code.family == "polar":
                    event_results.append(
                        combine_event_result(no_slip_event, polar_specific, decoder_name="event_state_sweep_polar_scflip", alarm=alarm, trigger_metric=metric)
                    )
                for result in event_results:
                    rows.append(decoder_row(
                        result,
                        frame_uid=frame_uid,
                        code_id=code_id,
                        code=code,
                        condition=condition,
                        tx_int=tx,
                        snr_db=snr_db,
                        true_label=label,
                        threshold=thresholds[code_id],
                    ))
                frame_counter += 1
                if frame_counter % int(cfg["trials"].get("checkpoint_every", 10)) == 0:
                    atomic_csv(pd.DataFrame(rows), run_dir / "performance_trials.csv")
    df = pd.DataFrame(rows)
    atomic_csv(df, run_dir / "performance_trials.csv")
    return df


def aggregate_performance(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    keys = ["code_id", "family", "n", "k", "rate", "condition", "snr_db", "decoder"]
    metric_cols = [
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
            "bler_wilson_low": lo,
            "bler_wilson_high": hi,
            "alarm_rate": float(g["alarm"].astype(float).mean()),
            "cap_rate": float(g["cap_hit"].astype(float).mean()),
            "success_rate": float(g["success"].astype(float).mean()),
        })
        for col in metric_cols:
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
    rows = []
    for code_id, g in base.groupby("code_id", sort=True):
        row: dict[str, Any] = {"code_id": code_id}
        for condition in sorted(g["condition"].unique()):
            h = g[g["condition"] == condition]
            row[f"alarm_rate_{condition}"] = float(h["alarm"].astype(float).mean())
            row[f"trials_{condition}"] = len(h)
        rows.append(row)
    return pd.DataFrame(rows)


def _pair(df: pd.DataFrame, code_id: str, conditions: list[str], a: str, b: str) -> pd.DataFrame:
    g = df[(df["code_id"] == code_id) & df["condition"].isin(conditions)]
    aa = g[g["decoder"] == a].set_index("frame_uid")
    bb = g[g["decoder"] == b].set_index("frame_uid")
    shared = aa.index.intersection(bb.index)
    if len(shared) == 0:
        return pd.DataFrame()
    cols = ["decoded_correct", "cap_hit", "wall_seconds", "components_generated", "bit_metric_accumulations"]
    out = pd.DataFrame(index=shared)
    for col in cols:
        out[f"a_{col}"] = aa.loc[shared, col]
        out[f"b_{col}"] = bb.loc[shared, col]
    return out.reset_index()


def performance_comparisons(cfg: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    rng = seeded_rng(int(cfg["master_seed"]), 800)
    rows = []
    candidates = ["event_ls_first", "event_ls_adaptive_l2", "event_ls_fixed_l2"]
    slip_conditions = [c for c in cfg["trials"]["conditions"] if c != "no_slip"]
    code_ids = sorted(df["code_id"].unique())
    for code_id in code_ids:
        family = str(df[df["code_id"] == code_id]["family"].iloc[0])
        baselines = ["event_state_sweep_osd"]
        if family == "extended_bch":
            baselines.append("event_state_sweep_chase_bch")
        if family == "polar":
            baselines.append("event_state_sweep_polar_scflip")
        for baseline in baselines:
            for candidate in candidates:
                pair = _pair(df, code_id, slip_conditions, candidate, baseline)
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
                cand_rows = df[(df["code_id"] == code_id) & df["condition"].isin(slip_conditions) & (df["decoder"] == candidate)]
                base_rows = df[(df["code_id"] == code_id) & df["condition"].isin(slip_conditions) & (df["decoder"] == baseline)]
                a_errors = int((~cand_rows["decoded_correct"].astype(bool)).sum())
                a_lo, a_hi = wilson_interval(a_errors, len(cand_rows))
                a_p99 = safe_quantile(pd.to_numeric(cand_rows["wall_seconds"], errors="coerce"), 0.99)
                b_p99 = safe_quantile(pd.to_numeric(base_rows["wall_seconds"], errors="coerce"), 0.99)
                rows.append({
                    "code_id": code_id,
                    "family": family,
                    "candidate": candidate,
                    "baseline": baseline,
                    "paired_trials": dn,
                    "eligible_work_pairs": min(wn, cn),
                    "candidate_bler": a_errors / len(cand_rows),
                    "candidate_bler_low": a_lo,
                    "candidate_bler_high": a_hi,
                    "candidate_cap_rate": float(cand_rows["cap_hit"].astype(float).mean()),
                    "candidate_minus_baseline_error_rate": diff,
                    "error_diff_low": dlo,
                    "error_diff_high": dhi,
                    "baseline_to_candidate_wall_median_ratio": wall,
                    "wall_ratio_low": wlo,
                    "wall_ratio_high": whi,
                    "baseline_to_candidate_component_median_ratio": comp,
                    "component_ratio_low": clo,
                    "component_ratio_high": chi,
                    "baseline_to_candidate_p99_wall_ratio": b_p99 / max(a_p99, np.finfo(float).tiny),
                    "mcnemar": json.dumps(mcnemar_one_sided(
                        ~cand_rows["decoded_correct"].astype(bool),
                        ~base_rows["decoded_correct"].astype(bool),
                    ), sort_keys=True),
                })
    return pd.DataFrame(rows)


def no_slip_overhead(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    no = df[df["condition"] == "no_slip"]
    for code_id, g in no.groupby("code_id"):
        base = g[g["decoder"] == "normal_code_decoder"]
        if base.empty:
            continue
        base_med = safe_quantile(base["wall_seconds"], 0.5)
        base_p99 = safe_quantile(base["wall_seconds"], 0.99)
        for decoder in ["event_ls_first", "event_ls_adaptive_l2", "event_ls_fixed_l2", "event_state_sweep_osd", "event_state_sweep_chase_bch", "event_state_sweep_polar_scflip"]:
            d = g[g["decoder"] == decoder]
            if d.empty:
                continue
            rows.append({
                "code_id": code_id,
                "decoder": decoder,
                "trials": len(d),
                "false_alarm_rate": float(d["alarm"].astype(float).mean()),
                "median_wall_overhead_ratio": safe_quantile(d["wall_seconds"], 0.5) / max(base_med, np.finfo(float).tiny),
                "p99_wall_overhead_ratio": safe_quantile(d["wall_seconds"], 0.99) / max(base_p99, np.finfo(float).tiny),
                "bler": float((~d["decoded_correct"].astype(bool)).mean()),
            })
    return pd.DataFrame(rows)


def physical_mixtures(cfg: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    slip_conditions = [c for c in cfg["trials"]["conditions"] if c != "no_slip"]
    slip_weights = cfg["trials"].get("slip_condition_weights", {})
    if not slip_weights:
        slip_weights = {c: 1.0 / len(slip_conditions) for c in slip_conditions}
    for code_id in sorted(df["code_id"].unique()):
        for decoder in sorted(df[df["code_id"] == code_id]["decoder"].unique()):
            base = df[(df["code_id"] == code_id) & (df["decoder"] == decoder)]
            no = base[base["condition"] == "no_slip"]
            slip_parts = {c: base[base["condition"] == c] for c in slip_conditions}
            if no.empty or any(part.empty for part in slip_parts.values()):
                continue
            for p in cfg["physical_priors"]:
                p = float(p)
                values_wall = []
                values_comp = []
                values_err = []
                weights = []
                for _, row in no.iterrows():
                    values_wall.append(float(row["wall_seconds"]))
                    values_comp.append(float(row["components_generated"]))
                    values_err.append(float(not bool(row["decoded_correct"])))
                    weights.append((1.0 - p) / len(no))
                for condition, part in slip_parts.items():
                    condition_weight = float(slip_weights.get(condition, 0.0))
                    for _, row in part.iterrows():
                        values_wall.append(float(row["wall_seconds"]))
                        values_comp.append(float(row["components_generated"]))
                        values_err.append(float(not bool(row["decoded_correct"])))
                        weights.append(p * condition_weight / len(part))
                w = np.asarray(weights, dtype=float)
                rows.append({
                    "code_id": code_id,
                    "decoder": decoder,
                    "physical_slip_probability": p,
                    "unconditional_bler": float(np.dot(np.asarray(values_err), w) / w.sum()),
                    "mean_wall_seconds": float(np.dot(np.asarray(values_wall), w) / w.sum()),
                    "p99_wall_seconds": weighted_quantile(np.asarray(values_wall), w, 0.99),
                    "p999_wall_seconds": weighted_quantile(np.asarray(values_wall), w, 0.999),
                    "mean_components_generated": float(np.dot(np.asarray(values_comp), w) / w.sum()),
                    "p99_components_generated": weighted_quantile(np.asarray(values_comp), w, 0.99),
                    "p999_components_generated": weighted_quantile(np.asarray(values_comp), w, 0.999),
                })
    return pd.DataFrame(rows)


def adjudicate(
    cfg: dict[str, Any],
    g0_chain: dict[str, Any],
    novelty_df: pd.DataFrame,
    orbit_df: pd.DataFrame,
    detector_df: pd.DataFrame,
    comparisons: pd.DataFrame,
    overhead: pd.DataFrame,
    mixtures: pd.DataFrame,
    unit_tests_passed: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    th = cfg["thresholds"]
    gates: list[dict[str, Any]] = []

    expected_commit = "0d2866b091576fe07521172378ded33da79ed545"
    required_blocked = {"C01", "C02", "C03", "C04", "C10", "C11", "C12"}
    required_provisional = {"C05", "C06", "C07", "C08"}
    disposition = {str(r["claim_id"]): str(r["disposition"]) for _, r in novelty_df.iterrows()}
    blocked_ok = all(disposition.get(cid, "").startswith("BLOCK") for cid in required_blocked)
    provisional_ok = all(disposition.get(cid, "").startswith("ALLOW") for cid in required_provisional)
    chain_ok = bool(
        g0_chain.get("source_commit") == expected_commit
        and g0_chain.get("source_verdict") == "PASS_NARROWED_APPROXIMATE_CLAIM_FREEZE"
        and g0_chain.get("foundational_exact_route") == "STOP_AS_PRIMARY_NOVELTY_ROUTE"
        and g0_chain.get("current_exact_certificate_route") == "STOP_AS_PRIMARY_PRACTICAL_ROUTE"
        and g0_chain.get("field_defining_program") == "HOLD_NOT_AUTHORIZED"
    )
    n0 = bool(chain_ok and blocked_ok and provisional_ok and novelty_df["closest_primary_source"].notna().all())
    gates.append({
        "gate": "G0_CHAIN_LOCK_AND_CLAIM_DISCIPLINE",
        "pass": n0,
        "source_commit": g0_chain.get("source_commit"),
        "source_verdict": g0_chain.get("source_verdict"),
        "broad_exact_route_blocked": blocked_ok,
        "narrow_route_only": provisional_ok,
        "blocked_claims": sorted(required_blocked),
        "provisional_claims": sorted(required_provisional),
        "field_defining_authorized": False,
        "patent_freedom_to_operate": g0_chain.get("patent_freedom_to_operate", "NOT_DETERMINED"),
    })

    structured_ids = [str(x["id"]) for x in cfg["codes"] if bool(x.get("structured", True))]
    orbit_structured = orbit_df[orbit_df["code_id"].isin(structured_ids)]
    c1 = bool(len(orbit_structured) == len(structured_ids) and orbit_structured["orbit_safe"].astype(bool).all())
    gates.append({
        "gate": "C1_STRUCTURED_CODE_ORBIT_SAFETY",
        "pass": c1,
        "structured_codes": structured_ids,
        "unsafe_codes": orbit_structured[~orbit_structured["orbit_safe"].astype(bool)]["code_id"].astype(str).tolist(),
    })

    det = detector_df.set_index("code_id") if not detector_df.empty else pd.DataFrame()
    detector_details = []
    detector_pass_codes = []
    for code_id in structured_ids:
        if code_id not in det.index:
            continue
        row = det.loc[code_id]
        fa = float(row.get("alarm_rate_no_slip", np.nan))
        early = float(row.get("alarm_rate_slip_early", np.nan))
        uniform = float(row.get("alarm_rate_slip_uniform", np.nan))
        qualifies = bool(
            fa <= float(th["max_false_alarm_rate"])
            and early >= float(th["min_early_detection_rate"])
            and uniform >= float(th["min_uniform_detection_rate"])
        )
        detector_pass_codes.append(code_id) if qualifies else None
        detector_details.append({"code_id": code_id, "false_alarm_rate": fa, "early_detection_rate": early, "uniform_detection_rate": uniform, "qualifies": qualifies})
    c2 = len(detector_pass_codes) == len(structured_ids)
    gates.append({"gate": "C2_EVENT_TRIGGER", "pass": c2, "details": detector_details})

    qualifying = []
    if not comparisons.empty:
        for _, row in comparisons[comparisons["baseline"] == "event_state_sweep_osd"].iterrows():
            qualifies = bool(
                int(row["paired_trials"]) >= int(th["min_paired_slip_trials"])
                and int(row["eligible_work_pairs"]) >= int(th["min_eligible_work_pairs"])
                and float(row["candidate_bler_high"]) <= float(th["max_candidate_bler_upper"])
                and float(row["error_diff_high"]) <= float(th["max_candidate_bler_excess"])
                and float(row["candidate_cap_rate"]) <= float(th["max_candidate_cap_rate"])
                and float(row["baseline_to_candidate_wall_median_ratio"]) >= float(th["min_wall_ratio"])
                and float(row["wall_ratio_low"]) > 1.0
                and float(row["baseline_to_candidate_p99_wall_ratio"]) >= float(th["min_p99_wall_ratio"])
            )
            if qualifies:
                qualifying.append(row.to_dict())
    qualifying_codes = sorted(set(str(x["code_id"]) for x in qualifying if str(x["code_id"]) in structured_ids))
    c3_modes_by_code: dict[str, set[str]] = {code_id: set() for code_id in structured_ids}
    for row in qualifying:
        code_id = str(row["code_id"])
        if code_id in c3_modes_by_code:
            c3_modes_by_code[code_id].add(str(row["candidate"]))
    c3 = len(qualifying_codes) == len(structured_ids)
    gates.append({
        "gate": "C3_APPROXIMATE_STRUCTURED_CODE_FRONTIER",
        "pass": c3,
        "qualifying_structured_codes": qualifying_codes,
        "qualifying_modes": sorted(set(str(x["candidate"]) for x in qualifying)),
        "qualifying_modes_by_code": {k: sorted(v) for k, v in c3_modes_by_code.items()},
        "qualifying_rows": qualifying,
    })

    code_specific_details = []
    code_specific_pass_codes: list[str] = []
    c4_modes_by_code: dict[str, set[str]] = {code_id: set() for code_id in structured_ids}
    baseline_by_family = {
        "extended_bch": "event_state_sweep_chase_bch",
        "polar": "event_state_sweep_polar_scflip",
    }
    for code_id in structured_ids:
        family = str(orbit_df[orbit_df["code_id"] == code_id]["family"].iloc[0])
        baseline_name = baseline_by_family.get(family)
        if baseline_name is None:
            continue
        subset = comparisons[(comparisons["code_id"] == code_id) & (comparisons["baseline"] == baseline_name)]
        good = []
        for _, row in subset.iterrows():
            qualifies = bool(
                int(row["eligible_work_pairs"]) >= int(th["min_eligible_work_pairs"])
                and float(row["error_diff_high"]) <= float(th["max_candidate_bler_excess"])
                and float(row["baseline_to_candidate_wall_median_ratio"]) >= float(th["min_code_specific_wall_ratio"])
                and float(row["wall_ratio_low"]) > 1.0
            )
            rec = {**row.to_dict(), "qualifies": qualifies}
            code_specific_details.append(rec)
            if qualifies:
                good.append(rec)
                c4_modes_by_code[code_id].add(str(row["candidate"]))
        if good:
            code_specific_pass_codes.append(code_id)
    c4 = len(code_specific_pass_codes) == len(structured_ids)
    gates.append({
        "gate": "C4_CODE_SPECIFIC_BASELINES",
        "pass": c4,
        "qualifying_codes": code_specific_pass_codes,
        "qualifying_modes_by_code": {k: sorted(v) for k, v in c4_modes_by_code.items()},
        "details": code_specific_details,
    })

    overhead_ok = []
    c5_modes_by_code: dict[str, set[str]] = {code_id: set() for code_id in structured_ids}
    for code_id in structured_ids:
        rows = overhead[(overhead["code_id"] == code_id) & overhead["decoder"].isin(["event_ls_first", "event_ls_adaptive_l2"])]
        good = rows[
            (rows["false_alarm_rate"] <= float(th["max_false_alarm_rate"]))
            & (rows["median_wall_overhead_ratio"] <= float(th["max_no_slip_median_overhead_ratio"]))
            & (rows["p99_wall_overhead_ratio"] <= float(th["max_no_slip_p99_overhead_ratio"]))
        ]
        if not good.empty:
            overhead_ok.append(code_id)
            c5_modes_by_code[code_id].update(str(x) for x in good["decoder"].tolist())
    c5 = len(overhead_ok) == len(structured_ids)
    gates.append({
        "gate": "C5_NO_SLIP_NEUTRALITY",
        "pass": c5,
        "qualifying_codes": overhead_ok,
        "qualifying_modes_by_code": {k: sorted(v) for k, v in c5_modes_by_code.items()},
    })

    physical_details = []
    p_target = float(th["physical_gate_slip_probability"])
    tail_q = float(th.get("physical_tail_quantile", 0.999))
    tail_col = "p999_wall_seconds" if tail_q >= 0.999 - 1e-12 else "p99_wall_seconds"
    physical_baseline_by_family = {
        "extended_bch": "event_state_sweep_chase_bch",
        "polar": "event_state_sweep_polar_scflip",
    }
    for code_id in structured_ids:
        family = str(orbit_df[orbit_df["code_id"] == code_id]["family"].iloc[0])
        baseline_name = physical_baseline_by_family[family]
        base = mixtures[
            (mixtures["code_id"] == code_id)
            & (mixtures["physical_slip_probability"] == p_target)
            & (mixtures["decoder"] == baseline_name)
        ]
        eligible_modes = sorted(
            c3_modes_by_code.get(code_id, set())
            & c4_modes_by_code.get(code_id, set())
            & c5_modes_by_code.get(code_id, set())
        )
        candidates = mixtures[
            (mixtures["code_id"] == code_id)
            & (mixtures["physical_slip_probability"] == p_target)
            & mixtures["decoder"].isin(eligible_modes)
        ]
        if base.empty or candidates.empty:
            continue
        b = base.iloc[0]
        best = None
        for _, cand in candidates.iterrows():
            ratio = float(b[tail_col]) / max(float(cand[tail_col]), np.finfo(float).tiny)
            bler_excess = float(cand["unconditional_bler"] - b["unconditional_bler"])
            qualifies = bool(
                ratio >= float(th["min_physical_tail_wall_ratio"])
                and bler_excess <= float(th["max_physical_unconditional_bler_excess"])
            )
            rec = {
                "code_id": code_id,
                "candidate": str(cand["decoder"]),
                "baseline": baseline_name,
                "tail_quantile": tail_q,
                "tail_wall_ratio": ratio,
                "unconditional_bler_excess": bler_excess,
                "candidate_mean_wall_seconds": float(cand["mean_wall_seconds"]),
                "eligible_modes_after_C3_C4_C5": eligible_modes,
                "qualifies": qualifies,
            }
            # Prefer a coherent, simpler operating point when statistical outcomes tie:
            # lower BLER excess, larger tail advantage, lower mean work, then first-valid.
            key = (
                bool(rec["qualifies"]),
                -float(rec["unconditional_bler_excess"]),
                float(rec["tail_wall_ratio"]),
                -float(rec["candidate_mean_wall_seconds"]),
                str(rec["candidate"]) == "event_ls_first",
            )
            if best is None:
                best = rec
                best_key = key
            elif key > best_key:
                best = rec
                best_key = key
        if best is not None:
            physical_details.append(best)
    c6 = len([x for x in physical_details if x["qualifies"]]) == len(structured_ids)
    gates.append({"gate": "C6_SYNTHETIC_PHYSICAL_MIXTURE_SCREEN", "pass": c6, "details": physical_details, "measured_trace": False})

    c7 = bool(unit_tests_passed)
    gates.append({"gate": "C7_REPRODUCIBILITY", "pass": c7, "validator_required": True})

    if cfg["profile"] == "smoke":
        verdict = "INCONCLUSIVE_SMOKE_ONLY"
        rationale = "Smoke mode validates installation and logic only."
    elif not c7:
        verdict = "STOP_REPRODUCIBILITY"
        rationale = "Unit, schema, or reproducibility controls failed."
    elif not n0:
        verdict = "STOP_CLAIM_CHAIN_VIOLATION"
        rationale = "The completed G0 blocked/allowed claim boundary was altered or lost."
    elif not c1:
        verdict = "STOP_CODE_IDENTIFIABILITY"
        rationale = "The structured codes could not be made orbit-safe within the frozen screening budget."
    elif c2 and c3 and c4 and c5 and c6:
        verdict = "CONTINUE_TO_FINAL_PHYSICAL_MATCHED_GATE"
        rationale = "The approximate event-triggered route survived orbit-screened structured codes, code-specific baselines, normal-frame neutrality, and the synthetic rare-event screen; one final measured/standard-derived matched physical gate is justified."
    elif c3 and c4 and (not c2 or not c5 or not c6):
        verdict = "CONTINUE_CONDITIONAL_ON_EXTERNAL_ALARM_ONLY"
        rationale = "Conditional latent recovery remains favorable, but the internal alarm/no-slip/rare-event architecture is inadequate; continue only with a defensible external alarm."
    elif c3 and not c4:
        verdict = "NARROW_TO_APPLICATION_ONLY"
        rationale = "The generic state-sweep baseline was beaten, but at least one structured-code-specific baseline was not decisively beaten."
    else:
        verdict = "STOP_APPROXIMATE_ROUTE"
        rationale = "The narrowed approximate route did not survive the structured-code matched-baseline gate."

    result = {
        "verdict": verdict,
        "rationale": rationale,
        "profile": cfg["profile"],
        "field_defining_verdict_authorized": False,
        "exact_certificate_route_status": "FROZEN_AS_THEORY_ORACLE_NOT_PRACTICAL_RECEIVER",
        "next_if_continue": [
            "targeted citation and patent follow-up for the provisional C06/C07 combination",
            "measured or standard-derived optical/wireless residual-slip trace",
            "equal-net-rate pilot/differential comparison",
            "optimized iterative or BCJR/code-specific receiver",
        ],
        "gates": gates,
    }
    return result, pd.DataFrame(gates)


def write_figures(
    run_dir: Path,
    orbit_df: pd.DataFrame,
    detector_df: pd.DataFrame,
    comparisons: pd.DataFrame,
    overhead: pd.DataFrame,
    mixtures: pd.DataFrame,
) -> None:
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    if not orbit_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(orbit_df))
        ax.bar(x, orbit_df["max_collision_fraction"].astype(float))
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(orbit_df["code_id"].astype(str), rotation=30, ha="right")
        ax.set_ylabel("Maximum one-slip collision fraction")
        ax.set_title("Orbit-screened code mappings")
        fig.tight_layout()
        fig.savefig(figures / "orbit_screen.png", dpi=180)
        plt.close(fig)
    if not detector_df.empty:
        cols = [c for c in detector_df.columns if c.startswith("alarm_rate_")]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(detector_df))
        width = 0.8 / max(1, len(cols))
        for j, col in enumerate(cols):
            ax.bar(x + (j - (len(cols) - 1) / 2) * width, detector_df[col].astype(float), width=width, label=col.removeprefix("alarm_rate_"))
        ax.set_xticks(x)
        ax.set_xticklabels(detector_df["code_id"].astype(str))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Alarm probability")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "event_trigger.png", dpi=180)
        plt.close(fig)
    if not comparisons.empty:
        d = comparisons[comparisons["baseline"] == "event_state_sweep_osd"].copy()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(d["candidate_minus_baseline_error_rate"], d["baseline_to_candidate_wall_median_ratio"])
        ax.axvline(0.0, linewidth=0.8)
        ax.axhline(1.0, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("Candidate BLER minus state-sweep OSD BLER")
        ax.set_ylabel("Median state-sweep / candidate wall-time ratio")
        for _, row in d.iterrows():
            ax.annotate(f"{row['code_id']}:{row['candidate']}", (row["candidate_minus_baseline_error_rate"], row["baseline_to_candidate_wall_median_ratio"]), fontsize=6)
        fig.tight_layout()
        fig.savefig(figures / "structured_frontier.png", dpi=180)
        plt.close(fig)
    if not overhead.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        d = overhead[overhead["decoder"].isin(["event_ls_first", "event_ls_adaptive_l2"])].copy()
        labels = d["code_id"].astype(str) + ":" + d["decoder"].astype(str)
        x = np.arange(len(d))
        ax.bar(x, d["p99_wall_overhead_ratio"].astype(float))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("No-slip p99 wall-time overhead")
        fig.tight_layout()
        fig.savefig(figures / "no_slip_overhead.png", dpi=180)
        plt.close(fig)
    if not mixtures.empty:
        target = max(float(x) for x in mixtures["physical_slip_probability"].unique())
        d = mixtures[(mixtures["physical_slip_probability"] == target) & mixtures["decoder"].isin(["event_ls_first", "event_ls_adaptive_l2", "event_state_sweep_osd"])]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        labels = d["code_id"].astype(str) + ":" + d["decoder"].astype(str)
        x = np.arange(len(d))
        ax.bar(x, d["p99_wall_seconds"].astype(float))
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
        ax.set_ylabel(f"Unconditional p99 wall time at p_slip={target:g}")
        fig.tight_layout()
        fig.savefig(figures / "physical_mixture_p99.png", dpi=180)
        plt.close(fig)


def write_report(run_dir: Path, verdict: dict[str, Any], novelty_df: pd.DataFrame) -> None:
    gate_lines = [f"- **{g['gate']}**: {'PASS' if g['pass'] else 'FAIL'}" for g in verdict["gates"]]
    text = f"""# LS-GRAND G3P structured matched precursor report

## Classification

**{verdict['verdict']}**

{verdict['rationale']}

This result cannot authorize a field-defining claim.  The exact certificate route
is frozen as a mathematical oracle because v1.1 rejected its practical efficiency.

## Gate status

{os.linesep.join(gate_lines)}

## Frozen practical receiver

- normal operation: frozen code-specific decoder (Chase-BCH, polar SC-Flip, or OSD control);
- event trigger: code-aided normalized residual calibrated on independent no-slip frames;
- recovery candidates: globally ordered LS first-valid and path-multiplicity-triggered adaptive L=2; fixed L=2 is diagnostic-only;
- matched baselines: state-pruned OSD for every code, state-sweep Chase-BCH for eBCH, and state-sweep polar SC-Flip for the polar code;
- channel screen: QPSK with rare persistent +/-pi/2 state jumps, AWGN, residual Wiener phase, and small frequency-offset mismatch;
- code screen: extended BCH(64,45), polar(64,48), and a random-linear control, each with deterministic orbit screening/interleaving.

## Claim discipline

The G0 claim matrix is an immutable scientific chain lock; narrow C06/C07 novelty and patent freedom to operate remain provisional.
The first-valid/tiny-list route is a narrow practical hypothesis, not a new generic
hidden-state or threshold-aggregation principle. It is publishable only if the
structured/physical evidence is strong and the precise C06/C07 combination survives
targeted independent review.

## Required review files

- `FINAL_G3P_VERDICT.json`
- `gate_status.csv`
- `code_orbit_summary.csv`
- `detector_summary.csv`
- `performance_aggregate.csv`
- `performance_comparisons.csv`
- `no_slip_overhead.csv`
- `physical_mixture.csv`
- `G0_CHAIN_LOCK.json`
- `G0_CLAIM_MATRIX_FROZEN.csv`
- `VALIDATION_REPORT.json`
"""
    (run_dir / "FINAL_G3P_REPORT.md").write_text(text)


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


def run_campaign(
    package_root: Path,
    config_path: Path,
    output_root: Path,
    run_name: str,
    *,
    unit_tests_passed: bool,
) -> Path:
    cfg = load_config(config_path)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "FROZEN_CONFIG.json", cfg)
    write_json(run_dir / "RUN_STATE.json", {"status": "RUNNING", "started_utc": utc_stamp()})
    g0_chain = load_g0_chain_lock(package_root)
    novelty = load_novelty_matrix(package_root)
    atomic_csv(novelty, run_dir / "G0_CLAIM_MATRIX_FROZEN.csv")
    write_json(run_dir / "G0_CHAIN_LOCK.json", g0_chain)
    codes, orbit, orbit_transforms = build_screened_codes(cfg, run_dir)
    thresholds, calibration = calibrate_detector(cfg, codes, orbit, run_dir)
    trials = run_performance(cfg, codes, thresholds, run_dir)
    aggregate = aggregate_performance(trials)
    atomic_csv(aggregate, run_dir / "performance_aggregate.csv")
    detector = detector_summary(trials)
    atomic_csv(detector, run_dir / "detector_summary.csv")
    comparisons = performance_comparisons(cfg, trials)
    atomic_csv(comparisons, run_dir / "performance_comparisons.csv")
    overhead = no_slip_overhead(trials)
    atomic_csv(overhead, run_dir / "no_slip_overhead.csv")
    mixtures = physical_mixtures(cfg, trials)
    atomic_csv(mixtures, run_dir / "physical_mixture.csv")
    verdict, gates = adjudicate(cfg, g0_chain, novelty, orbit, detector, comparisons, overhead, mixtures, unit_tests_passed)
    write_json(run_dir / "FINAL_G3P_VERDICT.json", verdict)
    atomic_csv(gates, run_dir / "gate_status.csv")
    write_figures(run_dir, orbit, detector, comparisons, overhead, mixtures)
    write_report(run_dir, verdict, novelty)
    write_json(run_dir / "REPRODUCIBILITY_MANIFEST.json", environment_manifest(package_root, run_dir, cfg))
    write_json(run_dir / "RUN_STATE.json", {"status": "COMPLETE", "completed_utc": utc_stamp(), "verdict": verdict["verdict"]})
    manifest = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "RESULT_SHA256_MANIFEST.json":
            manifest[str(path.relative_to(run_dir))] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    write_json(run_dir / "RESULT_SHA256_MANIFEST.json", manifest)
    return run_dir
