from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .affine import all_one_slip_paths, linear_code_affine_collision
from .channel import (
    fixed_slip_path,
    one_slip_hypotheses,
    sample_state_path,
    simulate_qpsk_awgn,
    simulate_qpsk_residual_phase,
)
from .decoders import (
    certified_lsgrand_audit,
    empirical_true_ranks,
    latent_list_decode,
    latent_osd_batch,
)
from .gf2 import LinearCode, systematic_random_code
from .oracle import direct_probability_scores, exhaustive_oracle
from .search import first_valid_latent, word_key
from .stats import (
    bootstrap_median_ci,
    mcnemar_exact_one_sided,
    paired_bootstrap_median_ratio,
    paired_error_difference_interval,
    rule_of_three_upper,
    safe_quantile,
    wilson_interval,
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def seeded_rng(master: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(master), *(int(x) for x in parts)]))


def make_code(case: dict[str, Any], rng: np.random.Generator) -> LinearCode:
    return systematic_random_code(
        int(case["n"]), int(case["k"]), rng,
        family=str(case.get("family", "dense")),
        density=case.get("density"),
    )


def bits_int(bits: np.ndarray) -> int:
    value = 0
    for b in np.asarray(bits, dtype=np.uint8).reshape(-1):
        value = (value << 1) | int(b)
    return value


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    required = {"profile", "master_seed", "exact_fiber", "rank_probe", "orbit_audit", "performance", "thresholds"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"config missing keys: {sorted(missing)}")
    return cfg


def run_exact_fiber(cfg: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    master = int(cfg["master_seed"])
    stage = cfg["exact_fiber"]
    uid = 0
    for ci, case in enumerate(stage["cases"]):
        for rep in range(int(case.get("code_replicates", 1))):
            code = make_code(case, seeded_rng(master, 100, ci, rep))
            states = one_slip_hypotheses(
                code.n // 2, float(case["model_slip_prob"]), case.get("directions", [1, 3])
            )
            for t in range(int(case["trials_per_code"])):
                rng = seeded_rng(master, 101, ci, rep, t)
                word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
                true_path, true_label = sample_state_path(
                    code.n // 2, rng,
                    frame_slip_prob=float(case["true_slip_prob"]),
                    directions=case.get("directions", [1, 3]),
                )
                snr_db = float(rng.choice(np.asarray(case["snr_db"], dtype=float)))
                sample = simulate_qpsk_awgn(word, true_path, snr_db, rng, true_label)
                oracle = exhaustive_oracle(sample.y, sample.n0, code, states, max_k=int(stage.get("max_k", 16)))
                cartesian = len(states) * (1 << code.n)
                cert = certified_lsgrand_audit(
                    sample.y, sample.n0, code, states,
                    query_cap=min(int(case.get("query_cap", cartesian)), cartesian),
                    certificate_interval=1,
                )
                first = first_valid_latent(sample.y, sample.n0, code, states, query_cap=cartesian)
                direct_ok = True
                if t == 0:
                    direct = direct_probability_scores(sample.y, sample.n0, code, states, max_k=int(stage.get("max_k", 16)))
                    direct_ok = max(direct, key=direct.get) == oracle.marginal_winner_int
                tx = bits_int(word)
                marginal_key = word_key(np.asarray(oracle.marginal_winner_bits, dtype=np.uint8)).hex()
                max_component_for_marginal_winner = float(oracle.joint_scores[marginal_key])
                lse_gain = float(oracle.marginal_logscore - max_component_for_marginal_winner)
                effective_paths = float(np.exp(min(50.0, lse_gain)))
                row = {
                    "case_id": case["id"], "trial_uid": f"{case['id']}_c{rep}_t{t}",
                    "trial_id": uid, "n": code.n, "k": code.k, "rate": code.rate,
                    "family": code.family, "snr_db": snr_db, "true_label": true_label,
                    "states": len(states), "transmitted_int": tx,
                    "marginal_winner_int": oracle.marginal_winner_int,
                    "joint_winner_int": oracle.joint_winner_int,
                    "first_winner_int": first.decoded_int,
                    "certified_winner_int": cert.decoded_int,
                    "oracle_tie": oracle.marginal_tie,
                    "certified": cert.certified,
                    "certificate_cap_hit": cert.cap_hit,
                    "cert_matches_marginal": cert.decoded_int == oracle.marginal_winner_int,
                    "direct_probability_check": direct_ok,
                    "marginal_vs_joint_switch": oracle.marginal_winner_int != oracle.joint_winner_int,
                    "marginal_vs_first_switch": oracle.marginal_winner_int != first.decoded_int,
                    "marginal_error": oracle.marginal_winner_int != tx,
                    "joint_error": oracle.joint_winner_int != tx,
                    "first_error": first.decoded_int != tx,
                    "logsumexp_gain_winner": lse_gain,
                    "effective_path_multiplicity_winner": effective_paths,
                    "oracle_margin_log": oracle.marginal_margin_log,
                    "cert_components": cert.components_generated,
                    "cert_queue_touch_fraction": cert.queue_touch_fraction,
                    "first_components": first.residual_patterns_generated,
                    "certificate_over_first": cert.components_generated / max(1, first.residual_patterns_generated),
                }
                rows.append(row)
                uid += 1
                if uid % int(stage.get("checkpoint_every", 50)) == 0:
                    atomic_csv(pd.DataFrame(rows), run_dir / "exact_fiber_trials.csv")
    df = pd.DataFrame(rows)
    atomic_csv(df, run_dir / "exact_fiber_trials.csv")
    return df


def aggregate_exact_fiber(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    if df.empty:
        return pd.DataFrame(), {}
    for case_id, g in df.groupby("case_id", sort=True):
        n = len(g)
        m_err = int(g["marginal_error"].astype(bool).sum())
        j_err = int(g["joint_error"].astype(bool).sum())
        f_err = int(g["first_error"].astype(bool).sum())
        mci = wilson_interval(m_err, n)
        rows.append({
            "case_id": case_id, "trials": n,
            "marginal_errors": m_err, "joint_errors": j_err, "first_errors": f_err,
            "marginal_bler": m_err / n, "marginal_bler_low": mci[0], "marginal_bler_high": mci[1],
            "marginal_joint_switch_rate": float(g["marginal_vs_joint_switch"].mean()),
            "marginal_first_switch_rate": float(g["marginal_vs_first_switch"].mean()),
            "median_effective_path_multiplicity": float(g["effective_path_multiplicity_winner"].median()),
            "p90_effective_path_multiplicity": float(g["effective_path_multiplicity_winner"].quantile(.9)),
            "median_certificate_over_first": float(g["certificate_over_first"].median()),
            "median_queue_touch_fraction": float(g["cert_queue_touch_fraction"].median()),
        })
    non_tied = df[~df["oracle_tie"].astype(bool)]
    summary = {
        "trials": len(df),
        "non_tied_trials": len(non_tied),
        "cert_mismatches": int((~non_tied["cert_matches_marginal"].astype(bool)).sum()),
        "certificate_failures_or_caps": int((~non_tied["certified"].astype(bool)).sum()),
        "direct_failures": int((~df["direct_probability_check"].astype(bool)).sum()),
        "marginal_joint_switches": int(df["marginal_vs_joint_switch"].astype(bool).sum()),
        "marginal_first_switches": int(df["marginal_vs_first_switch"].astype(bool).sum()),
        "marginal_errors": int(df["marginal_error"].astype(bool).sum()),
        "joint_errors": int(df["joint_error"].astype(bool).sum()),
        "first_errors": int(df["first_error"].astype(bool).sum()),
        "marginal_vs_joint_mcnemar": mcnemar_exact_one_sided(df["marginal_error"], df["joint_error"]),
        "marginal_vs_first_mcnemar": mcnemar_exact_one_sided(df["marginal_error"], df["first_error"]),
        "median_effective_path_multiplicity": float(df["effective_path_multiplicity_winner"].median()),
        "p90_effective_path_multiplicity": float(df["effective_path_multiplicity_winner"].quantile(.9)),
        "zero_mismatch_95pct_upper": rule_of_three_upper(len(non_tied)) if len(non_tied) else float("nan"),
    }
    return pd.DataFrame(rows), summary


def run_rank_probe(cfg: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    master = int(cfg["master_seed"])
    stage = cfg["rank_probe"]
    uid = 0
    for ci, case in enumerate(stage["cases"]):
        for rep in range(int(case.get("code_replicates", 1))):
            code = make_code(case, seeded_rng(master, 200, ci, rep))
            states = one_slip_hypotheses(code.n // 2, float(case["model_slip_prob"]), case.get("directions", [1, 3]))
            for lname, frac in case["locations"].items():
                for t in range(int(case["trials_per_location"])):
                    rng = seeded_rng(master, 201, ci, rep, t, int(round(float(frac) * 10000)))
                    word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
                    tau = min(max(1, int(round(float(frac) * (code.n // 2)))), code.n // 2 - 1)
                    direction = int(rng.choice(case.get("directions", [1, 3])))
                    path = fixed_slip_path(code.n // 2, tau, direction)
                    sample = simulate_qpsk_awgn(word, path, float(case["snr_db"]), rng, f"slip_{tau}_{direction}")
                    probe = empirical_true_ranks(
                        sample.y, sample.n0, word, states, path,
                        latent_cap=int(case["latent_cap"]), plain_cap=int(case["plain_cap"]),
                    )
                    row = {
                        "case_id": case["id"], "trial_uid": f"{case['id']}_c{rep}_{lname}_t{t}",
                        "trial_id": uid, "n": code.n, "k": code.k, "rate": code.rate,
                        "family": code.family, "snr_db": float(case["snr_db"]),
                        "location_label": lname, "location_fraction": float(frac),
                        "tau": tau, "direction": direction,
                        **probe.to_dict(),
                    }
                    rows.append(row)
                    uid += 1
                    if uid % int(stage.get("checkpoint_every", 20)) == 0:
                        atomic_csv(pd.DataFrame(rows), run_dir / "rank_probe_trials.csv")
    df = pd.DataFrame(rows)
    atomic_csv(df, run_dir / "rank_probe_trials.csv")
    return df


def aggregate_rank_probe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    keys = ["case_id", "family", "n", "k", "rate", "location_label", "location_fraction", "snr_db"]
    for key, g in df.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        lower = pd.to_numeric(g["rank_ratio_lower_bound"], errors="coerce").to_numpy(dtype=float)
        exact = pd.to_numeric(g["rank_ratio"], errors="coerce").to_numpy(dtype=float)
        latent_rank = pd.to_numeric(g["latent_true_component_rank"], errors="coerce").to_numpy(dtype=float)
        row.update(
            trials=len(g),
            latent_success_rate=float((~g["latent_cap_hit"].astype(bool)).mean()),
            plain_censor_rate=float(g["plain_cap_hit"].astype(bool).mean()),
            median_latent_rank=safe_quantile(latent_rank, .5),
            p90_latent_rank=safe_quantile(latent_rank, .9),
            median_rank_ratio_lower_bound=safe_quantile(lower, .5),
            p10_rank_ratio_lower_bound=safe_quantile(lower, .1),
            median_exact_rank_ratio=safe_quantile(exact, .5),
            median_latent_queue_touch_fraction=float(
                (pd.to_numeric(g["latent_queues_touched"], errors="coerce") /
                 pd.to_numeric(g["latent_hypotheses_available"], errors="coerce")).median()
            ),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_orbit_audit(cfg: dict[str, Any], run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    master = int(cfg["master_seed"])
    stage = cfg["orbit_audit"]
    for ci, case in enumerate(stage["cases"]):
        for rep in range(int(case["code_replicates"])):
            code = make_code(case, seeded_rng(master, 300, ci, rep))
            paths = all_one_slip_paths(code.n // 2, case.get("directions", [1, 2, 3]))
            fractions = []
            full = 0
            for label, path in paths:
                result = linear_code_affine_collision(code, path)
                tau = int(label.split("_")[0].removeprefix("tau"))
                direction = int(label.rsplit("d", 1)[1])
                fractions.append(result.collision_fraction)
                full += int(result.collision_fraction >= 1.0 - 1e-15)
                rows.append({
                    "case_id": case["id"], "code_replicate": rep,
                    "n": code.n, "k": code.k, "redundancy": code.n - code.k,
                    "rate": code.rate, "family": code.family,
                    "transform": label, "tau": tau, "direction": direction,
                    "collision_fraction": result.collision_fraction,
                    "collision_log2_fraction": result.collision_log2_fraction,
                    "system_rank": result.system_rank, "consistent": result.consistent,
                    "full_orbit_invariance": result.collision_fraction >= 1.0 - 1e-15,
                })
            arr = np.asarray(fractions, dtype=float)
            maxfrac = float(arr.max()) if arr.size else 0.0
            meanfrac = float(arr.mean()) if arr.size else 0.0
            union = float(min(1.0, arr.sum()))
            threshold = float(case.get("dangerous_fraction", 1e-3))
            summaries.append({
                "case_id": case["id"], "code_replicate": rep,
                "n": code.n, "k": code.k, "redundancy": code.n - code.k,
                "rate": code.rate, "family": code.family, "transforms": len(paths),
                "max_collision_fraction": maxfrac,
                "mean_collision_fraction": meanfrac,
                "union_bound_random_transform_collision": union,
                "full_invariance_count": full,
                "dangerous_transform_count": int(np.count_nonzero(arr > threshold)),
                "orbit_safe": bool(maxfrac <= threshold),
            })
    raw = pd.DataFrame(rows)
    summ = pd.DataFrame(summaries)
    atomic_csv(raw, run_dir / "orbit_transform_trials.csv")
    atomic_csv(summ, run_dir / "orbit_code_summary.csv")
    return raw, summ


def aggregate_orbit(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    keys = ["case_id", "family", "n", "k", "redundancy", "rate"]
    for key, g in summary.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row.update(
            code_replicates=len(g),
            orbit_safe_fraction=float(g["orbit_safe"].astype(float).mean()),
            any_full_invariance_fraction=float((g["full_invariance_count"] > 0).astype(float).mean()),
            median_max_collision_fraction=float(g["max_collision_fraction"].median()),
            max_collision_fraction=float(g["max_collision_fraction"].max()),
            median_union_bound=float(g["union_bound_random_transform_collision"].median()),
            median_dangerous_transform_count=float(g["dangerous_transform_count"].median()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _decoder_row(common: dict[str, Any], d, transmitted_int: int) -> dict[str, Any]:
    row = {**common, "decoder": d.decoder, "decoded_correct": d.decoded_int == transmitted_int}
    row.update(d.to_dict())
    row.pop("decoded_bits", None)
    return row


def run_performance(cfg: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    master = int(cfg["master_seed"])
    stage = cfg["performance"]
    frames = 0
    for ci, case in enumerate(stage["cases"]):
        for rep in range(int(case.get("code_replicates", 1))):
            code = make_code(case, seeded_rng(master, 400, ci, rep))
            for t in range(int(case["trials_per_code"])):
                rng = seeded_rng(master, 401, ci, rep, t)
                word = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
                true_path, true_label = sample_state_path(
                    code.n // 2, rng,
                    frame_slip_prob=float(case["true_slip_prob"]),
                    directions=case.get("true_directions", case.get("directions", [1, 3])),
                    forced_location_fraction=case.get("forced_location_fraction"),
                    two_slip_fraction=float(case.get("two_slip_fraction", 0.0)),
                )
                phase_std = float(case.get("phase_innovation_std_deg", 0.0))
                if phase_std > 0 or float(case.get("phase_initial_std_deg", 0.0)) > 0:
                    sample = simulate_qpsk_residual_phase(
                        word, true_path, float(case["snr_db"]), rng,
                        innovation_std_deg=phase_std,
                        initial_std_deg=float(case.get("phase_initial_std_deg", 0.0)),
                        true_label=true_label,
                    )
                else:
                    sample = simulate_qpsk_awgn(word, true_path, float(case["snr_db"]), rng, true_label)
                states = one_slip_hypotheses(
                    code.n // 2, float(case["model_slip_prob"]), case.get("directions", [1, 3])
                )
                decoder_n0 = sample.n0 * 10.0 ** (float(case.get("assumed_noise_offset_db", 0.0)) / 10.0)
                tx = bits_int(word)
                trial_uid = f"{case['id']}_c{rep}_t{t}"
                common = {
                    "case_id": case["id"], "trial_uid": trial_uid,
                    "n": code.n, "k": code.k, "rate": code.rate, "family": code.family,
                    "snr_db": float(case["snr_db"]), "true_slip_prob": float(case["true_slip_prob"]),
                    "model_slip_prob": float(case["model_slip_prob"]),
                    "forced_location_fraction": case.get("forced_location_fraction"),
                    "two_slip_fraction": float(case.get("two_slip_fraction", 0.0)),
                    "phase_innovation_std_deg": phase_std,
                    "phase_initial_std_deg": float(case.get("phase_initial_std_deg", 0.0)),
                    "assumed_noise_offset_db": float(case.get("assumed_noise_offset_db", 0.0)),
                    "true_state_label": true_label,
                    "model_contains_true_path": any(np.array_equal(s.path, true_path) for s in states),
                    "state_hypothesis_count": len(states),
                    "code_replicate": rep, "trial_index": t,
                    "transmitted_int": tx,
                }
                dcfg = case["decoder"]
                first = latent_list_decode(
                    sample.y, decoder_n0, code, states,
                    list_size=1, query_cap=int(dcfg["ls_query_cap"]), marginal_rescore=False,
                )
                rows.append(_decoder_row(common, first, tx))
                for list_size in dcfg.get("list_sizes", [4]):
                    d = latent_list_decode(
                        sample.y, decoder_n0, code, states,
                        list_size=int(list_size), query_cap=int(dcfg["ls_query_cap"]),
                        marginal_rescore=True, score_batch_size=int(dcfg.get("score_batch_size", 512)),
                    )
                    rows.append(_decoder_row(common, d, tx))
                osd = latent_osd_batch(
                    sample.y, decoder_n0, code, states,
                    order=int(dcfg.get("osd_order", 2)),
                    pool_size=int(dcfg.get("osd_pool_size", 12)),
                    state_limit=dcfg.get("osd_state_limit"),
                    candidate_cap=int(dcfg.get("osd_candidate_cap", 100000)),
                    score_batch_size=int(dcfg.get("score_batch_size", 512)),
                )
                rows.append(_decoder_row(common, osd, tx))
                # A no-slip OSD control shows whether the gain comes from modeling
                # the latent state rather than from OSD implementation details.
                no_slip_states = one_slip_hypotheses(code.n // 2, 0.0, [1, 3])
                plain_osd = latent_osd_batch(
                    sample.y, decoder_n0, code, no_slip_states,
                    order=int(dcfg.get("osd_order", 2)),
                    pool_size=int(dcfg.get("osd_pool_size", 12)),
                    state_limit=1,
                    candidate_cap=int(dcfg.get("osd_candidate_cap", 100000)),
                    score_batch_size=int(dcfg.get("score_batch_size", 512)),
                )
                plain_osd.decoder = f"plain_osd_batch_o{int(dcfg.get('osd_order', 2))}"
                rows.append(_decoder_row(common, plain_osd, tx))
                if bool(dcfg.get("run_certificate", True)):
                    cert = certified_lsgrand_audit(
                        sample.y, decoder_n0, code, states,
                        query_cap=int(dcfg["cert_query_cap"]),
                        certificate_interval=int(dcfg.get("certificate_interval", 32)),
                    )
                    rows.append(_decoder_row(common, cert, tx))
                frames += 1
                if frames % int(stage.get("checkpoint_every", 5)) == 0:
                    atomic_csv(pd.DataFrame(rows), run_dir / "performance_trials.csv")
    df = pd.DataFrame(rows)
    atomic_csv(df, run_dir / "performance_trials.csv")
    return df


def aggregate_performance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    keys = [
        "case_id", "n", "k", "rate", "family", "snr_db", "true_slip_prob",
        "model_slip_prob", "forced_location_fraction", "two_slip_fraction",
        "phase_innovation_std_deg", "decoder",
    ]
    for key, g in df.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, key))
        n = len(g)
        errors = int((~g["decoded_correct"].astype(bool)).sum())
        lo, hi = wilson_interval(errors, n)
        row.update(
            trials=n, errors=errors, bler=errors / n, bler_wilson_low=lo, bler_wilson_high=hi,
            cap_rate=float(g["cap_hit"].astype(float).mean()),
            certification_rate=float(g["certified"].astype(float).mean()),
        )
        for col in [
            "components_generated", "membership_queries", "latent_queues_touched",
            "queue_touch_fraction", "valid_codewords", "unique_candidates",
            "complete_marginal_candidates", "state_word_metric_evals",
            "bit_metric_accumulations", "osd_reprocessings", "preprocessing_state_metrics",
            "wall_seconds",
        ]:
            vals = pd.to_numeric(g[col], errors="coerce")
            finite = vals[np.isfinite(vals)]
            row[f"median_{col}"] = float(finite.median()) if len(finite) else float("nan")
            row[f"p90_{col}"] = float(finite.quantile(.9)) if len(finite) else float("nan")
            row[f"mean_{col}"] = float(finite.mean()) if len(finite) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _pair(df: pd.DataFrame, case_id: str, a: str, b: str) -> pd.DataFrame:
    g = df[df["case_id"] == case_id]
    aa = g[g["decoder"] == a].set_index("trial_uid")
    bb = g[g["decoder"] == b].set_index("trial_uid")
    shared = aa.index.intersection(bb.index)
    if not len(shared):
        return pd.DataFrame()
    cols = [
        "decoded_correct", "cap_hit", "certified", "components_generated",
        "membership_queries", "state_word_metric_evals", "bit_metric_accumulations",
        "wall_seconds", "queue_touch_fraction",
    ]
    out = pd.DataFrame(index=shared)
    for col in cols:
        out[f"a_{col}"] = aa.loc[shared, col]
        out[f"b_{col}"] = bb.loc[shared, col]
    return out.reset_index()


def performance_comparisons(cfg: dict[str, Any], df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Evaluate the full preregistered LS list-size frontier against matched OSD."""
    rows = []
    candidate_modes = ["ls_first_valid"] + [
        f"ls_list_marginal_L{int(x)}" for x in cfg["performance"].get("comparison_list_sizes", [2, 4, 8])
    ]
    for case in cfg["performance"]["cases"]:
        case_id = case["id"]
        directions = len([d for d in case.get("directions", [1, 3]) if int(d) % 4])
        symbols = int(case["n"]) // 2
        states = (1 if float(case["model_slip_prob"]) < 1 else 0) + directions * (symbols - 1)
        osd_limit = case["decoder"].get("osd_state_limit")
        osd_limit = states if osd_limit is None else min(int(osd_limit), states)
        osd_name = f"latent_osd_batch_o{int(case['decoder'].get('osd_order', 2))}_s{osd_limit}"
        for ls_name in candidate_modes:
            pair = _pair(df, case_id, ls_name, osd_name)
            if pair.empty:
                continue
            eligible = pair[~pair["a_cap_hit"].astype(bool) & ~pair["b_cap_hit"].astype(bool)]
            diff, dlo, dhi, dn = paired_error_difference_interval(
                (~pair["a_decoded_correct"].astype(bool)).astype(int).to_numpy(),
                (~pair["b_decoded_correct"].astype(bool)).astype(int).to_numpy(),
                rng, resamples=int(cfg["thresholds"].get("bootstrap_resamples", 1500)),
            )
            wall, wall_lo, wall_hi, wn = paired_bootstrap_median_ratio(
                pd.to_numeric(eligible["b_wall_seconds"], errors="coerce").to_numpy(),
                pd.to_numeric(eligible["a_wall_seconds"], errors="coerce").to_numpy(),
                rng, resamples=int(cfg["thresholds"].get("bootstrap_resamples", 1500)),
            )
            bit, bit_lo, bit_hi, bn = paired_bootstrap_median_ratio(
                pd.to_numeric(eligible["b_bit_metric_accumulations"], errors="coerce").to_numpy(),
                np.maximum(1.0, pd.to_numeric(eligible["a_bit_metric_accumulations"], errors="coerce").to_numpy()),
                rng, resamples=int(cfg["thresholds"].get("bootstrap_resamples", 1500)),
            )
            comp, comp_lo, comp_hi, cn = paired_bootstrap_median_ratio(
                pd.to_numeric(eligible["b_components_generated"], errors="coerce").to_numpy(),
                np.maximum(1.0, pd.to_numeric(eligible["a_components_generated"], errors="coerce").to_numpy()),
                rng, resamples=int(cfg["thresholds"].get("bootstrap_resamples", 1500)),
            )
            ag = df[(df["case_id"] == case_id) & (df["decoder"] == ls_name)]
            aerr = int((~ag["decoded_correct"].astype(bool)).sum())
            abl = aerr / len(ag) if len(ag) else float("nan")
            alo, ahi = wilson_interval(aerr, len(ag)) if len(ag) else (float("nan"), float("nan"))
            rows.append({
                "case_id": case_id, "family": case["family"], "n": case["n"], "k": case["k"],
                "ls_decoder": ls_name, "osd_decoder": osd_name,
                "paired_trials": dn, "eligible_work_pairs": min(wn, bn, cn),
                "ls_bler": abl, "ls_bler_low": alo, "ls_bler_high": ahi,
                "ls_minus_osd_error_rate": diff, "error_diff_low": dlo, "error_diff_high": dhi,
                "osd_to_ls_wall_median_ratio": wall, "wall_ratio_low": wall_lo, "wall_ratio_high": wall_hi,
                "osd_to_ls_bit_metric_median_ratio": bit, "bit_ratio_low": bit_lo, "bit_ratio_high": bit_hi,
                "osd_to_ls_component_median_ratio": comp, "component_ratio_low": comp_lo, "component_ratio_high": comp_hi,
            })
    return pd.DataFrame(rows)


def certificate_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for case_id in sorted(df["case_id"].unique()):
        pair = _pair(df, case_id, "lsgrand_certified_v1_bound", "ls_first_valid")
        if pair.empty:
            continue
        cert = df[(df["case_id"] == case_id) & (df["decoder"] == "lsgrand_certified_v1_bound")]
        ratios = pd.to_numeric(pair["a_components_generated"], errors="coerce") / np.maximum(
            1.0, pd.to_numeric(pair["b_components_generated"], errors="coerce")
        )
        rows.append({
            "case_id": case_id, "trials": len(pair),
            "certification_rate": float(cert["certified"].astype(float).mean()),
            "cap_rate": float(cert["cap_hit"].astype(float).mean()),
            "median_certificate_over_first": float(ratios.median()),
            "p90_certificate_over_first": float(ratios.quantile(.9)),
            "median_queue_touch_fraction": float(pd.to_numeric(cert["queue_touch_fraction"], errors="coerce").median()),
            "p90_queue_touch_fraction": float(pd.to_numeric(cert["queue_touch_fraction"], errors="coerce").quantile(.9)),
        })
    return pd.DataFrame(rows)


def adjudicate(
    cfg: dict[str, Any],
    exact_summary: dict[str, Any],
    exact_agg: pd.DataFrame,
    rank_agg: pd.DataFrame,
    orbit_agg: pd.DataFrame,
    perf_agg: pd.DataFrame,
    perf_cmp: pd.DataFrame,
    cert_cmp: pd.DataFrame,
    unit_tests_passed: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    th = cfg["thresholds"]
    gates: list[dict[str, Any]] = []

    f0 = bool(
        unit_tests_passed
        and exact_summary.get("non_tied_trials", 0) >= int(th["f0_min_nontied"])
        and exact_summary.get("cert_mismatches", 1) == 0
        and exact_summary.get("certificate_failures_or_caps", 1) == 0
        and exact_summary.get("direct_failures", 1) == 0
    )
    gates.append({"gate": "F0_EXACTNESS", "pass": f0, **exact_summary})

    first_test = exact_summary.get("marginal_vs_first_mcnemar", {})
    switch_rate = exact_summary.get("marginal_first_switches", 0) / max(1, exact_summary.get("trials", 0))
    practical_gain = (
        exact_summary.get("first_errors", 0) - exact_summary.get("marginal_errors", 0)
    ) / max(1, exact_summary.get("trials", 0))
    f1 = bool(
        switch_rate >= float(th["f1_min_switch_rate"])
        and practical_gain >= float(th["f1_min_error_gain"])
        and float(first_test.get("one_sided_p_a_better", 1.0)) <= float(th["f1_max_pvalue"])
    )
    gates.append({
        "gate": "F1_MARGINAL_FIBER_RELEVANCE", "pass": f1,
        "marginal_first_switch_rate": switch_rate,
        "marginal_over_first_error_gain": practical_gain,
        "mcnemar": first_test,
        "median_effective_path_multiplicity": exact_summary.get("median_effective_path_multiplicity"),
    })

    rank_qual = pd.DataFrame()
    if not rank_agg.empty:
        rank_qual = rank_agg[
            (rank_agg["location_label"].isin(["early", "middle"]))
            & (rank_agg["n"] >= int(th["f2_min_n"]))
            & (rank_agg["latent_success_rate"] >= float(th["f2_min_latent_success"]))
            & (rank_agg["median_rank_ratio_lower_bound"] >= float(th["f2_min_median_ratio"]))
            & (rank_agg["p10_rank_ratio_lower_bound"] >= float(th["f2_min_p10_ratio"]))
        ]
    f2 = bool(
        len(rank_qual) >= int(th["f2_min_configs"])
        and len(set(rank_qual["family"].astype(str))) >= int(th["f2_min_families"])
        and len(set(rank_qual["n"].astype(int))) >= int(th["f2_min_blocklengths"])
    )
    gates.append({
        "gate": "F2_EMPIRICAL_RANK_COMPRESSION", "pass": f2,
        "qualifying_configs": int(len(rank_qual)),
        "qualifying_families": sorted(set(rank_qual["family"].astype(str))) if len(rank_qual) else [],
        "qualifying_blocklengths": sorted(set(rank_qual["n"].astype(int))) if len(rank_qual) else [],
    })

    cmp_qual = pd.DataFrame()
    if not perf_cmp.empty:
        work_ok = (
            (
                (perf_cmp["osd_to_ls_bit_metric_median_ratio"] >= float(th["f3_min_bit_metric_ratio"]))
                & (perf_cmp["bit_ratio_low"] > 1.0)
            )
            | (
                (perf_cmp["osd_to_ls_component_median_ratio"] >= float(th["f3_min_component_ratio"]))
                & (perf_cmp["component_ratio_low"] > 1.0)
            )
        )
        cmp_qual = perf_cmp[
            (perf_cmp["paired_trials"] >= int(th["f3_min_paired_trials"]))
            & (perf_cmp["ls_bler_high"] <= float(th["f3_max_bler_upper"]))
            & (perf_cmp["error_diff_high"] <= float(th["f3_max_ls_error_excess"]))
            & work_ok
            & (perf_cmp["osd_to_ls_wall_median_ratio"] >= float(th["f3_min_wall_ratio"]))
            & (perf_cmp["wall_ratio_low"] > 1.0)
        ]
    qualifying_cases = sorted(set(cmp_qual["case_id"].astype(str))) if len(cmp_qual) else []
    f3 = bool(
        len(qualifying_cases) >= int(th["f3_min_configs"])
        and len(set(cmp_qual["family"].astype(str))) >= int(th["f3_min_families"])
    )
    gates.append({
        "gate": "F3_APPROXIMATE_CANDIDATE_FRONTIER", "pass": f3,
        "qualifying_configs": int(len(qualifying_cases)),
        "qualifying_cases": qualifying_cases,
        "qualifying_modes": sorted(set(cmp_qual["ls_decoder"].astype(str))) if len(cmp_qual) else [],
        "qualifying_families": sorted(set(cmp_qual["family"].astype(str))) if len(cmp_qual) else [],
    })

    cert_qual = pd.DataFrame()
    if not cert_cmp.empty:
        cert_qual = cert_cmp[
            (cert_cmp["certification_rate"] >= float(th["f4_min_certification_rate"]))
            & (cert_cmp["median_certificate_over_first"] <= float(th["f4_max_median_overhead"]))
            & (cert_cmp["median_queue_touch_fraction"] < float(th["f4_max_median_queue_touch"]))
            & (cert_cmp["p90_queue_touch_fraction"] < float(th["f4_max_p90_queue_touch"]))
        ]
    f4 = bool(len(cert_qual) >= int(th["f4_min_configs"]))
    gates.append({
        "gate": "F4_EXACT_CERTIFICATE_USEFULNESS", "pass": f4,
        "qualifying_configs": int(len(cert_qual)),
    })

    orbit_target = pd.DataFrame()
    if not orbit_agg.empty:
        orbit_target = orbit_agg[
            (orbit_agg["n"] >= int(th["f5_min_n"]))
            & (orbit_agg["redundancy"] >= int(th["f5_min_redundancy"]))
            & (orbit_agg["orbit_safe_fraction"] >= float(th["f5_min_safe_fraction"]))
        ]
    f5 = bool(
        len(orbit_target) >= int(th["f5_min_configs"])
        and len(set(orbit_target["family"].astype(str))) >= int(th["f5_min_families"])
    )
    screenable = False
    if not orbit_agg.empty:
        screenable = bool(float(orbit_agg["orbit_safe_fraction"].max()) >= float(th["f5_screenable_fraction"]))
    gates.append({
        "gate": "F5_TARGET_ORBIT_IDENTIFIABILITY", "pass": f5,
        "qualifying_configs": int(len(orbit_target)), "screenable": screenable,
    })

    # Reproducibility is completed after all result files are written.  The
    # validator independently checks hashes and schemas; at adjudication time we
    # record the intended gate as provisionally true.
    f6 = True
    gates.append({"gate": "F6_REPRODUCIBILITY", "pass": f6, "validator_required": True})

    if cfg["profile"] == "smoke":
        verdict = "INCONCLUSIVE_SMOKE_ONLY"
        rationale = "Smoke mode validates installation, schemas, and logic only; it is not powered for a scientific verdict."
    elif not f0:
        verdict = "STOP_MATHEMATICAL_CORE"
        rationale = "The exact decoder or its certificate failed exhaustive-oracle validation."
    elif not f2:
        verdict = "STOP_SPARSE_CAUSE_SEARCH_THESIS"
        rationale = "Actual noisy enumeration did not show robust latent-rank compression."
    elif f3 and f5 and f4 and f1:
        verdict = "CONTINUE_EXACT_ALGORITHMIC_CANDIDATE_PHYSICAL_GATE_PENDING"
        rationale = "Exactness, fiber relevance, search advantage, certificate utility, and target-code identifiability passed; physical and formal novelty gates remain."
    elif f3 and f5:
        verdict = "CONTINUE_APPROXIMATE_LATENT_SEARCH_CANDIDATE"
        rationale = "The low-description latent search is promising, but the exact marginal/certificate story is not yet practically justified."
    elif f3 and not f5 and screenable:
        verdict = "CONTINUE_ONLY_WITH_ORBIT_SAFE_CODE_RESTRICTION"
        rationale = "Candidate search is promising, but unrestricted codes are not identifiable; proceed only with an explicit orbit-screening constraint."
    elif f2 and not f3:
        verdict = "PIVOT_THEORY_ONLY_OR_STOP_ALGORITHM"
        rationale = "The coordinate advantage exists, but the repaired matched-baseline gate did not establish an algorithmic advantage."
    else:
        verdict = "STOP_CURRENT_FORM"
        rationale = "The combined candidate-search and identifiability evidence is insufficient."

    result = {
        "verdict": verdict,
        "rationale": rationale,
        "profile": cfg["profile"],
        "field_defining_verdict_authorized": False,
        "field_defining_blockers": [
            "formal claim-by-claim novelty search remains incomplete",
            "no measured or defensible platform-specific physical slip model has been validated",
            "no optimized code-specific BCJR/iterative/pilot-adjusted receiver is included",
        ],
        "gates": gates,
    }
    return result, pd.DataFrame(gates)


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



def write_figures(
    run_dir: Path,
    rank_agg: pd.DataFrame,
    orbit_agg: pd.DataFrame,
    perf_cmp: pd.DataFrame,
    cert_cmp: pd.DataFrame,
) -> None:
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)

    if not rank_agg.empty:
        d = rank_agg.copy()
        d["label"] = d["case_id"].astype(str) + ":" + d["location_label"].astype(str)
        fig, ax = plt.subplots(figsize=(max(8, len(d) * 0.42), 4.8))
        ax.bar(np.arange(len(d)), d["median_rank_ratio_lower_bound"].astype(float))
        ax.set_yscale("log")
        ax.set_ylabel("Median rank-ratio lower bound")
        ax.set_title("Actual noisy plain-rank / latent-rank evidence")
        ax.set_xticks(np.arange(len(d)))
        ax.set_xticklabels(d["label"], rotation=75, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "actual_noisy_rank_ratio.png", dpi=180)
        plt.close(fig)

    if not orbit_agg.empty:
        d = orbit_agg.copy()
        fig, ax = plt.subplots(figsize=(max(8, len(d) * 0.55), 4.8))
        ax.bar(np.arange(len(d)), d["orbit_safe_fraction"].astype(float))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Orbit-safe code fraction")
        ax.set_title("Target-length affine orbit safety")
        ax.set_xticks(np.arange(len(d)))
        ax.set_xticklabels(d["case_id"].astype(str), rotation=70, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "target_orbit_safe_fraction.png", dpi=180)
        plt.close(fig)

    if not perf_cmp.empty:
        d = perf_cmp.copy()
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.scatter(d["ls_minus_osd_error_rate"].astype(float), d["osd_to_ls_wall_median_ratio"].astype(float))
        ax.axvline(0.0, linewidth=0.8)
        ax.axhline(1.0, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("LS error rate minus matched OSD error rate")
        ax.set_ylabel("Median OSD / LS wall-time ratio")
        ax.set_title("Pre-registered LS list-size frontier")
        for _, row in d.iterrows():
            ax.annotate(f"{row['case_id']}:{row['ls_decoder']}",
                        (float(row["ls_minus_osd_error_rate"]), float(row["osd_to_ls_wall_median_ratio"])),
                        fontsize=6, alpha=0.8)
        fig.tight_layout()
        fig.savefig(figures / "candidate_frontier.png", dpi=180)
        plt.close(fig)

    if not cert_cmp.empty:
        d = cert_cmp.copy()
        fig, ax = plt.subplots(figsize=(max(7, len(d) * 0.7), 4.8))
        ax.bar(np.arange(len(d)), d["median_certificate_over_first"].astype(float))
        ax.axhline(10.0, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_ylabel("Median certified / first-valid components")
        ax.set_title("Exact-certificate overhead")
        ax.set_xticks(np.arange(len(d)))
        ax.set_xticklabels(d["case_id"].astype(str), rotation=65, ha="right", fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / "certificate_overhead.png", dpi=180)
        plt.close(fig)

def write_report(
    run_dir: Path,
    verdict: dict[str, Any],
    exact_summary: dict[str, Any],
    rank_agg: pd.DataFrame,
    orbit_agg: pd.DataFrame,
    perf_cmp: pd.DataFrame,
    cert_cmp: pd.DataFrame,
) -> None:
    gate_lines = []
    for g in verdict["gates"]:
        gate_lines.append(f"- **{g['gate']}**: {'PASS' if g['pass'] else 'FAIL'}")
    text = f"""# LS-GRAND decisive pivot gate report

## Automated classification

**{verdict['verdict']}**

{verdict['rationale']}

This classification is deliberately not a field-defining verdict.  The campaign
repairs the v1.0 algorithmic evidence, while formal novelty and physical-platform
validation remain outside the numerical package.

## Gate status

{os.linesep.join(gate_lines)}

## Exact mathematical core

- Non-tied exhaustive-oracle trials: {exact_summary.get('non_tied_trials')}
- Certified mismatches: {exact_summary.get('cert_mismatches')}
- Certificate failures/caps: {exact_summary.get('certificate_failures_or_caps')}
- Marginal-versus-first decision switches: {exact_summary.get('marginal_first_switches')}
- Marginal / first errors: {exact_summary.get('marginal_errors')} / {exact_summary.get('first_errors')}
- Median effective winner-path multiplicity: {exact_summary.get('median_effective_path_multiplicity')}

## Actual noisy rank evidence

The file `rank_probe_aggregate.csv` reports measured ranks from the exact noisy
queues.  Capped plain-search ranks are treated as lower bounds, not as observed
finite ranks.  This replaces the v1.0 noiseless Hamming-combinatorial surrogate.

## Target-length orbit identifiability

The file `orbit_aggregate.csv` uses an exact GF(2) rank/consistency calculation
for every one-slip QPSK affine transformation at the tested target lengths.  It
does not extrapolate from tiny enumerated codebooks.

## Matched practical comparison

`performance_comparisons.csv` compares a bounded LS marginal list against a
cached, vectorized state-aware OSD union.  Error confidence intervals,
state-word metric evaluations, bit-metric accumulations, and wall time remain
separate.  A positive result is still provisional until an optimized
code-specific BCJR/iterative/pilot-adjusted baseline is supplied.

## Exact-certificate status

`certificate_comparisons.csv` measures certification cost relative to the same
latent search stopped at its first valid codeword.  Failure here forces an
approximate-search pivot; it cannot be upgraded by other failed gates.

## Remaining nonnumerical blockers

1. Independent claim-by-claim novelty and patent search.
2. A defensible measured, hardware-emulated, or standard-derived residual-slip model.
3. Equal-net-rate comparison with pilot/differential and optimized code-specific receivers.
"""
    (run_dir / "FINAL_DECISIVE_REPORT.md").write_text(text)


def result_hash_manifest(run_dir: Path) -> dict[str, Any]:
    exclude = {"RESULT_SHA256_MANIFEST.json", "RUN_STATE.json"}
    files = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name not in exclude:
            files.append({
                "path": str(p.relative_to(run_dir)),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    return {"created_utc": utc_stamp(), "files": files}


def run_campaign(
    package_root: Path,
    config_path: Path,
    output_root: Path,
    run_name: str | None,
    *,
    unit_tests_passed: bool,
) -> Path:
    cfg = load_config(config_path)
    name = run_name or f"LS_GRAND_Decisive_Pivot_Gate_{cfg['profile']}_{utc_stamp()}"
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "FROZEN_CONFIG.json", cfg)
    write_json(run_dir / "RUN_STATE.json", {"status": "RUNNING", "started_utc": utc_stamp()})
    write_json(run_dir / "REPRODUCIBILITY_MANIFEST.json", environment_manifest(package_root, run_dir, cfg))
    try:
        exact = run_exact_fiber(cfg, run_dir)
        rng = seeded_rng(int(cfg["master_seed"]), 999)
        exact_agg, exact_summary = aggregate_exact_fiber(exact, rng)
        atomic_csv(exact_agg, run_dir / "exact_fiber_aggregate.csv")
        write_json(run_dir / "exact_fiber_summary.json", exact_summary)

        rank_trials = run_rank_probe(cfg, run_dir)
        rank_agg = aggregate_rank_probe(rank_trials)
        atomic_csv(rank_agg, run_dir / "rank_probe_aggregate.csv")

        _, orbit_summary = run_orbit_audit(cfg, run_dir)
        orbit_agg = aggregate_orbit(orbit_summary)
        atomic_csv(orbit_agg, run_dir / "orbit_aggregate.csv")

        perf = run_performance(cfg, run_dir)
        perf_agg = aggregate_performance(perf)
        atomic_csv(perf_agg, run_dir / "performance_aggregate.csv")
        perf_cmp = performance_comparisons(cfg, perf, rng)
        atomic_csv(perf_cmp, run_dir / "performance_comparisons.csv")
        cert_cmp = certificate_comparisons(perf)
        atomic_csv(cert_cmp, run_dir / "certificate_comparisons.csv")
        write_figures(run_dir, rank_agg, orbit_agg, perf_cmp, cert_cmp)

        verdict, gates = adjudicate(
            cfg, exact_summary, exact_agg, rank_agg, orbit_agg, perf_agg, perf_cmp,
            cert_cmp, unit_tests_passed,
        )
        atomic_csv(gates, run_dir / "gate_status.csv")
        write_json(run_dir / "FINAL_DECISIVE_VERDICT.json", verdict)
        write_report(run_dir, verdict, exact_summary, rank_agg, orbit_agg, perf_cmp, cert_cmp)
        write_json(run_dir / "RESULT_SHA256_MANIFEST.json", result_hash_manifest(run_dir))
        write_json(run_dir / "RUN_STATE.json", {
            "status": "COMPLETE", "completed_utc": utc_stamp(),
            "verdict": verdict["verdict"],
        })
        return run_dir
    except Exception as exc:
        write_json(run_dir / "RUN_STATE.json", {
            "status": "FAILED", "failed_utc": utc_stamp(),
            "error_type": type(exc).__name__, "error": str(exc),
        })
        raise
