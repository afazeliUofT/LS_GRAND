from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .channel import (
    fixed_slip_path,
    one_slip_hypotheses,
    sample_state_path,
    simulate_qpsk_awgn,
)
from .diagnostics import (
    canonical_collision_paths,
    exact_orbit_collision_records,
    rank_separation_record,
)
from .gf2 import LinearCode, systematic_random_code
from .oracle import direct_probability_scores, exhaustive_oracle
from .reporting import (
    aggregate_decoder_trials,
    environment_manifest,
    save_figures,
    write_json,
)
from .search import (
    certified_lsgrand,
    first_valid_latent,
    latent_osd,
    per_state_grand_sweep,
    plain_grand,
)
from .stats import paired_bootstrap_median_ratio, paired_error_difference_interval, rule_of_three_upper


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    required = {"profile", "master_seed", "exactness", "collisions", "rank_separation", "performance", "mismatch", "verdict_thresholds"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"config missing keys: {sorted(missing)}")
    return cfg


def seeded_rng(master: int, *parts: int) -> np.random.Generator:
    ss = np.random.SeedSequence([int(master), *(int(x) for x in parts)])
    return np.random.default_rng(ss)


def make_code(case: dict[str, Any], rng: np.random.Generator) -> LinearCode:
    return systematic_random_code(
        n=int(case["n"]),
        k=int(case["k"]),
        rng=rng,
        family=str(case.get("family", "dense")),
        density=case.get("density"),
    )


def run_exactness(cfg: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    master = int(cfg["master_seed"])
    stage = cfg["exactness"]
    global_trial = 0
    for ci, case in enumerate(stage["cases"]):
        trials = int(case["trials"])
        for t in range(trials):
            rng = seeded_rng(master, 100, ci, t)
            code = make_code(case, rng)
            message = rng.integers(0, 2, size=code.k, dtype=np.uint8)
            word = code.encode(message)
            true_path, true_label = sample_state_path(
                code.n // 2,
                rng,
                frame_slip_prob=float(case["true_slip_prob"]),
                directions=case.get("directions", [1, 3]),
            )
            snr_db = float(rng.choice(np.asarray(case["snr_db"], dtype=float)))
            sample = simulate_qpsk_awgn(word, true_path, snr_db, rng, true_label)
            states = one_slip_hypotheses(
                code.n // 2,
                frame_slip_prob=float(case["model_slip_prob"]),
                directions=case.get("directions", [1, 3]),
            )
            oracle = exhaustive_oracle(sample.y, sample.n0, code, states, max_k=int(stage.get("max_k", 18)))
            full_cartesian = len(states) * (1 << code.n)
            cap = min(int(case.get("query_cap", full_cartesian)), full_cartesian)
            ls = certified_lsgrand(
                sample.y,
                sample.n0,
                code,
                states,
                query_cap=cap,
                certificate_interval=int(case.get("certificate_interval", 1)),
                oracle_word=np.asarray(oracle.marginal_winner_bits, dtype=np.uint8),
            )
            first = first_valid_latent(sample.y, sample.n0, code, states, query_cap=cap)
            direct_ok = True
            if t == 0 and code.k <= 16:
                direct = direct_probability_scores(sample.y, sample.n0, code, states, max_k=16)
                direct_winner = max(direct, key=direct.get)
                direct_ok = direct_winner == oracle.marginal_winner_int
            decoded_int = ls.decoded_int
            row = {
                "experiment": "exactness",
                "case_id": case["id"],
                "trial_id": global_trial,
                "n": code.n,
                "k": code.k,
                "rate": code.rate,
                "family": code.family,
                "snr_db": snr_db,
                "true_label": true_label,
                "states": len(states),
                "full_cartesian_pairs": full_cartesian,
                "oracle_marginal_int": oracle.marginal_winner_int,
                "oracle_joint_int": oracle.joint_winner_int,
                "transmitted_int": int("".join(str(int(x)) for x in word), 2),
                "oracle_tie": oracle.marginal_tie,
                "oracle_margin_log": oracle.marginal_margin_log,
                "certified": ls.certified,
                "ls_decoded_int": decoded_int,
                "ls_matches_marginal": decoded_int == oracle.marginal_winner_int,
                "certificate_verified": ls.certified and decoded_int == oracle.marginal_winner_int,
                "first_decoded_int": first.decoded_int,
                "first_matches_marginal": first.decoded_int == oracle.marginal_winner_int,
                "joint_matches_marginal": oracle.joint_winner_int == oracle.marginal_winner_int,
                "marginal_error": oracle.marginal_winner_int != int("".join(str(int(x)) for x in word), 2),
                "joint_error": oracle.joint_winner_int != int("".join(str(int(x)) for x in word), 2),
                "first_error": first.decoded_int != int("".join(str(int(x)) for x in word), 2),
                "direct_probability_check": direct_ok,
                "membership_queries": ls.membership_queries,
                "query_fraction_of_cartesian": ls.membership_queries / full_cartesian,
                "latent_queues_touched": ls.latent_queues_touched,
                "queue_touch_fraction": ls.queue_touch_fraction,
                "valid_witnesses": ls.valid_witnesses,
                "unique_codewords_seen": ls.unique_codewords_seen,
                "certificate_query_overhead": ls.certificate_query_overhead,
                "certificate_margin_log": ls.certificate_margin_log,
                "ls_cap_hit": ls.cap_hit,
                "ls_wall_seconds": ls.wall_seconds,
                "first_queries": first.membership_queries,
                "first_wall_seconds": first.wall_seconds,
            }
            rows.append(row)
            global_trial += 1
            if global_trial % int(stage.get("checkpoint_every", 10)) == 0:
                atomic_csv(pd.DataFrame(rows), run_dir / "exactness_trials.csv")
    df = pd.DataFrame(rows)
    atomic_csv(df, run_dir / "exactness_trials.csv")
    return df


def run_collisions(cfg: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    master = int(cfg["master_seed"])
    stage = cfg["collisions"]
    for ci, case in enumerate(stage["cases"]):
        for rep in range(int(case["replicates"])):
            rng = seeded_rng(master, 200, ci, rep)
            code = make_code(case, rng)
            for rec in exact_orbit_collision_records(code, canonical_collision_paths(code.n), max_k=int(stage.get("max_k", 18))):
                rec.update(case_id=case["id"], code_replicate=rep)
                rows.append(rec)
    df = pd.DataFrame(rows)
    atomic_csv(df, run_dir / "collision_trials.csv")
    if not df.empty:
        agg = (
            df.groupby(["case_id", "family", "n", "k", "rate", "transform"], as_index=False)
            .agg(
                mean_collision_fraction=("collision_fraction", "mean"),
                max_collision_fraction=("collision_fraction", "max"),
                mean_fixed_fraction=("fixed_fraction", "mean"),
            )
        )
    else:
        agg = pd.DataFrame()
    atomic_csv(agg, run_dir / "collision_aggregate.csv")
    return agg


def run_rank_separation(cfg: dict[str, Any], run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    master = int(cfg["master_seed"])
    stage = cfg["rank_separation"]
    counter = 0
    for ci, case in enumerate(stage["cases"]):
        for rep in range(int(case.get("code_replicates", 1))):
            code_rng = seeded_rng(master, 300, ci, rep)
            code = make_code(case, code_rng)
            family_size = len(one_slip_hypotheses(code.n // 2, float(case.get("model_slip_prob", 0.5)), case.get("directions", [1, 2, 3])))
            for location_label, frac in case["locations"].items():
                for t in range(int(case["trials_per_location"])):
                    rng = seeded_rng(master, 301, ci, rep, t, int(round(float(frac) * 1000)))
                    message = rng.integers(0, 2, size=code.k, dtype=np.uint8)
                    word = code.encode(message)
                    tau = min(max(1, int(round(float(frac) * (code.n // 2)))), code.n // 2 - 1)
                    d = int(rng.choice(case.get("directions", [1, 3])))
                    path = fixed_slip_path(code.n // 2, tau, d)
                    rec = rank_separation_record(word, path, family_size)
                    rec.update(
                        case_id=case["id"], family=code.family, k=code.k, rate=code.rate,
                        code_replicate=rep, trial_id=counter, location_label=location_label,
                        location_fraction=float(frac), tau=tau, direction=d,
                    )
                    rows.append(rec)
                    counter += 1
    df = pd.DataFrame(rows)
    atomic_csv(df, run_dir / "rank_separation_trials.csv")
    if not df.empty:
        agg = (
            df.groupby(["case_id", "family", "n", "k", "rate", "location_label", "location_fraction"], as_index=False)
            .agg(
                trials=("trial_id", "count"),
                median_apparent_error_fraction=("apparent_error_fraction", "median"),
                median_hard_grand_log2_rank_lower_bound=("hard_grand_log2_rank_lower_bound", "median"),
                median_latent_family_log2_size=("latent_family_log2_size", "median"),
                median_log2_coordinate_separation_lower_bound=("log2_coordinate_separation_lower_bound", "median"),
                min_log2_coordinate_separation_lower_bound=("log2_coordinate_separation_lower_bound", "min"),
            )
        )
    else:
        agg = pd.DataFrame()
    atomic_csv(agg, run_dir / "rank_separation_aggregate.csv")
    return agg


def _decoder_rows_for_frame(
    experiment: str,
    config_id: str,
    trial_uid: str,
    code: LinearCode,
    sample,
    states,
    transmitted_int: int,
    common: dict[str, Any],
    decoder_cfg: dict[str, Any],
    decoder_n0: float,
) -> list[dict[str, Any]]:
    decoders = []
    ls = certified_lsgrand(
        sample.y, decoder_n0, code, states,
        query_cap=int(decoder_cfg["ls_query_cap"]),
        certificate_interval=int(decoder_cfg.get("certificate_interval", 16)),
    )
    decoders.append(ls)
    decoders.append(
        first_valid_latent(sample.y, decoder_n0, code, states, query_cap=int(decoder_cfg["first_query_cap"]))
    )
    decoders.append(
        per_state_grand_sweep(
            sample.y, decoder_n0, code, states,
            per_state_cap=int(decoder_cfg["per_state_cap"]),
            global_cap=int(decoder_cfg["sweep_global_cap"]),
        )
    )
    decoders.append(plain_grand(sample.y, decoder_n0, code, query_cap=int(decoder_cfg["plain_query_cap"])))
    decoders.append(
        latent_osd(
            sample.y, decoder_n0, code, states,
            order=int(decoder_cfg["osd_order"]),
            pool_size=int(decoder_cfg["osd_pool_size"]),
            state_limit=int(decoder_cfg["osd_state_limit"]),
            candidate_cap=int(decoder_cfg["osd_candidate_cap"]),
        )
    )
    rows: list[dict[str, Any]] = []
    for d in decoders:
        row = {
            "experiment": experiment,
            "config_id": config_id,
            "trial_uid": trial_uid,
            "n": code.n,
            "k": code.k,
            "rate": code.rate,
            "family": code.family,
            "snr_db": sample.snr_db,
            "transmitted_int": transmitted_int,
            "decoder": d.decoder,
            "decoded_int": d.decoded_int,
            "decoded_correct": d.decoded_int == transmitted_int,
            **common,
            **d.to_dict(),
        }
        # Keep the scalar decoded_int field from above, not the duplicate dataclass value.
        row["decoded_int"] = d.decoded_int
        row.pop("decoded_bits", None)
        rows.append(row)
    return rows


def run_performance_stage(
    cfg: dict[str, Any],
    run_dir: Path,
    stage_name: str,
    output_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    master = int(cfg["master_seed"])
    stage = cfg[stage_name]
    frame_count = 0
    for ci, case in enumerate(stage["cases"]):
        for rep in range(int(case.get("code_replicates", 1))):
            code_rng = seeded_rng(master, 400 if stage_name == "performance" else 500, ci, rep)
            code = make_code(case, code_rng)
            for t in range(int(case["trials"])):
                rng = seeded_rng(master, 401 if stage_name == "performance" else 501, ci, rep, t)
                message = rng.integers(0, 2, size=code.k, dtype=np.uint8)
                word = code.encode(message)
                true_path, true_label = sample_state_path(
                    code.n // 2,
                    rng,
                    frame_slip_prob=float(case["true_slip_prob"]),
                    directions=case.get("true_directions", case.get("directions", [1, 3])),
                    forced_location_fraction=case.get("forced_location_fraction"),
                    two_slip_fraction=float(case.get("two_slip_fraction", 0.0)),
                )
                sample = simulate_qpsk_awgn(word, true_path, float(case["snr_db"]), rng, true_label)
                states = one_slip_hypotheses(
                    code.n // 2,
                    frame_slip_prob=float(case["model_slip_prob"]),
                    directions=case.get("directions", [1, 3]),
                )
                n0_factor = 10.0 ** (float(case.get("assumed_noise_offset_db", 0.0)) / 10.0)
                decoder_n0 = sample.n0 * n0_factor
                model_contains = any(np.array_equal(s.path, true_path) for s in states)
                transmitted_int = int("".join(str(int(x)) for x in word), 2)
                common = {
                    "true_slip_prob": float(case["true_slip_prob"]),
                    "model_slip_prob": float(case["model_slip_prob"]),
                    "forced_location_fraction": case.get("forced_location_fraction"),
                    "two_slip_fraction": float(case.get("two_slip_fraction", 0.0)),
                    "assumed_noise_offset_db": float(case.get("assumed_noise_offset_db", 0.0)),
                    "true_state_label": true_label,
                    "model_contains_true_path": model_contains,
                    "state_hypothesis_count": len(states),
                    "code_replicate": rep,
                    "trial_index": t,
                }
                trial_uid = f"{case['id']}_c{rep}_t{t}"
                rows.extend(
                    _decoder_rows_for_frame(
                        stage_name, case["id"], trial_uid, code, sample, states,
                        transmitted_int, common, case["decoder"], decoder_n0,
                    )
                )
                frame_count += 1
                if frame_count % int(stage.get("checkpoint_every", 5)) == 0:
                    atomic_csv(pd.DataFrame(rows), run_dir / output_name)
    df = pd.DataFrame(rows)
    atomic_csv(df, run_dir / output_name)
    return df


def certificate_aggregate(perf_df: pd.DataFrame) -> pd.DataFrame:
    if perf_df.empty:
        return pd.DataFrame()
    ls = perf_df[perf_df["decoder"] == "lsgrand_certified"].copy()
    rows = []
    for config_id, g in ls.groupby("config_id"):
        rows.append(
            {
                "config_id": config_id,
                "trials": len(g),
                "certification_rate": float(g["certified"].astype(float).mean()),
                "cap_rate": float(g["cap_hit"].astype(float).mean()),
                "median_queue_touch_fraction": float(pd.to_numeric(g["queue_touch_fraction"], errors="coerce").median()),
                "p90_queue_touch_fraction": float(pd.to_numeric(g["queue_touch_fraction"], errors="coerce").quantile(0.9)),
                "median_certificate_query_overhead": float(pd.to_numeric(g["certificate_query_overhead"], errors="coerce").median()),
                "p90_certificate_query_overhead": float(pd.to_numeric(g["certificate_query_overhead"], errors="coerce").quantile(0.9)),
                "median_membership_queries": float(pd.to_numeric(g["membership_queries"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def _pair_decoder_rows(df: pd.DataFrame, config_id: str, a: str, b: str) -> pd.DataFrame:
    g = df[df["config_id"] == config_id]
    aa = g[g["decoder"] == a].set_index("trial_uid")
    bb = g[g["decoder"] == b].set_index("trial_uid")
    shared = aa.index.intersection(bb.index)
    if len(shared) == 0:
        return pd.DataFrame()
    out = pd.DataFrame(index=shared)
    for col in ["decoded_correct", "certified", "cap_hit", "state_codeword_likelihoods", "wall_seconds", "membership_queries"]:
        out[f"a_{col}"] = aa.loc[shared, col]
        out[f"b_{col}"] = bb.loc[shared, col]
    return out.reset_index()


def adjudicate(
    cfg: dict[str, Any],
    run_dir: Path,
    exact_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    mismatch_df: pd.DataFrame,
    cert_agg: pd.DataFrame,
    rank_agg: pd.DataFrame,
    collision_agg: pd.DataFrame,
    unit_tests_passed: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    th = cfg["verdict_thresholds"]
    rng = seeded_rng(int(cfg["master_seed"]), 900)
    gates: list[dict[str, Any]] = []

    non_tied = exact_df[~exact_df["oracle_tie"].astype(bool)] if not exact_df.empty else pd.DataFrame()
    exact_mismatches = int((~non_tied["ls_matches_marginal"].astype(bool)).sum()) if not non_tied.empty else 0
    certificate_failures = int((~non_tied["certificate_verified"].astype(bool)).sum()) if not non_tied.empty else 0
    direct_failures = int((~exact_df["direct_probability_check"].astype(bool)).sum()) if not exact_df.empty else 1
    g0_pass = bool(unit_tests_passed and len(non_tied) >= int(th["g0_min_nontied_trials"]) and exact_mismatches == 0 and certificate_failures == 0 and direct_failures == 0)
    gates.append({
        "gate": "G0_EXACTNESS", "pass": g0_pass,
        "nontied_trials": len(non_tied), "exact_mismatches": exact_mismatches,
        "certificate_failures": certificate_failures, "direct_failures": direct_failures,
        "zero_mismatch_95pct_upper": rule_of_three_upper(len(non_tied)) if exact_mismatches == 0 else None,
        "unit_tests_passed": unit_tests_passed,
    })

    marginal_joint_events = int((~exact_df["joint_matches_marginal"].astype(bool)).sum()) if not exact_df.empty else 0
    first_events = int((~exact_df["first_matches_marginal"].astype(bool)).sum()) if not exact_df.empty else 0
    marginal_errors = int(exact_df["marginal_error"].astype(bool).sum()) if not exact_df.empty else 0
    joint_errors = int(exact_df["joint_error"].astype(bool).sum()) if not exact_df.empty else 0
    first_errors = int(exact_df["first_error"].astype(bool).sum()) if not exact_df.empty else 0
    g1_pass = bool(
        marginal_joint_events + first_events >= int(th["g1_min_disagreement_events"])
        and (marginal_errors < joint_errors or marginal_errors < first_errors)
    )
    gates.append({
        "gate": "G1_FIBER_EFFECT", "pass": g1_pass,
        "marginal_joint_disagreements": marginal_joint_events,
        "marginal_first_disagreements": first_events,
        "marginal_errors": marginal_errors, "joint_errors": joint_errors,
        "first_errors": first_errors,
    })

    g2_details = []
    strong_configs = 0
    strong_families: set[str] = set()
    for config_id in sorted(perf_df["config_id"].unique()) if not perf_df.empty else []:
        pair = _pair_decoder_rows(perf_df, config_id, "lsgrand_certified", f"latent_osd_order{int(cfg['performance']['default_osd_order'])}")
        if pair.empty:
            continue
        eligible = pair[pair["a_certified"].astype(bool) & ~pair["a_cap_hit"].astype(bool) & ~pair["b_cap_hit"].astype(bool)]
        ratio, rlo, rhi, rn = paired_bootstrap_median_ratio(
            pd.to_numeric(eligible["b_state_codeword_likelihoods"], errors="coerce").to_numpy(),
            pd.to_numeric(eligible["a_state_codeword_likelihoods"], errors="coerce").to_numpy(),
            rng,
            resamples=int(th.get("bootstrap_resamples", 1000)),
        )
        diff, dlo, dhi, dn = paired_error_difference_interval(
            (~pair["a_decoded_correct"].astype(bool)).astype(int).to_numpy(),
            (~pair["b_decoded_correct"].astype(bool)).astype(int).to_numpy(),
            rng,
            resamples=int(th.get("bootstrap_resamples", 1000)),
        )
        config_rows = perf_df[perf_df["config_id"] == config_id]
        family = str(config_rows["family"].iloc[0])
        qualifies = bool(
            rn >= int(th["g2_min_paired_trials"])
            and ratio >= float(th["g2_median_work_ratio"])
            and rlo > 1.0
            and dhi <= float(th["g2_max_ls_bler_excess"])
        )
        if qualifies:
            strong_configs += 1
            strong_families.add(family)
        g2_details.append({
            "config_id": config_id, "family": family, "eligible_pairs": rn,
            "osd_to_ls_state_eval_median_ratio": ratio, "ratio_ci_low": rlo,
            "ratio_ci_high": rhi, "ls_minus_osd_error_rate": diff,
            "error_diff_ci_low": dlo, "error_diff_ci_high": dhi,
            "qualifies": qualifies,
        })
    g2_pass = bool(strong_configs >= int(th["g2_min_configs"]) and len(strong_families) >= int(th["g2_min_families"]))
    gates.append({
        "gate": "G2_MATCHED_SCALING_ADVANTAGE", "pass": g2_pass,
        "qualifying_configs": strong_configs, "qualifying_families": sorted(strong_families),
        "details": g2_details,
    })

    if cert_agg.empty:
        g3_pass = False
        g3_configs = []
    else:
        cert_agg = cert_agg.copy()
        cert_agg["qualifies"] = (
            (cert_agg["certification_rate"] >= float(th["g3_min_certification_rate"]))
            & (cert_agg["median_certificate_query_overhead"] <= float(th["g3_max_median_query_overhead"]))
            & (cert_agg["median_queue_touch_fraction"] < float(th["g3_max_median_queue_touch_fraction"]))
            & (cert_agg["p90_queue_touch_fraction"] < float(th["g3_max_p90_queue_touch_fraction"]))
        )
        g3_configs = cert_agg.to_dict(orient="records")
        g3_pass = int(cert_agg["qualifies"].sum()) >= int(th["g3_min_configs"])
    gates.append({"gate": "G3_CERTIFICATE_USEFULNESS", "pass": bool(g3_pass), "details": g3_configs})

    mismatch_qualifying = 0
    mismatch_details = []
    for config_id in sorted(mismatch_df["config_id"].unique()) if not mismatch_df.empty else []:
        pair = _pair_decoder_rows(mismatch_df, config_id, "lsgrand_certified", f"latent_osd_order{int(cfg['mismatch']['default_osd_order'])}")
        if pair.empty:
            continue
        cert_rate = float(pair["a_certified"].astype(float).mean())
        diff, dlo, dhi, dn = paired_error_difference_interval(
            (~pair["a_decoded_correct"].astype(bool)).astype(int).to_numpy(),
            (~pair["b_decoded_correct"].astype(bool)).astype(int).to_numpy(),
            rng,
            resamples=int(th.get("bootstrap_resamples", 1000)),
        )
        qualifies = bool(cert_rate >= float(th["g4_min_certification_rate"]) and dhi <= float(th["g4_max_ls_bler_excess"]))
        mismatch_qualifying += int(qualifies)
        mismatch_details.append({
            "config_id": config_id, "paired_trials": dn, "certification_rate": cert_rate,
            "ls_minus_osd_error_rate": diff, "error_diff_ci_low": dlo,
            "error_diff_ci_high": dhi, "qualifies": qualifies,
        })
    if not collision_agg.empty:
        target_collision = collision_agg[~collision_agg["transform"].astype(str).str.startswith("global_")]
        collision_max = float(target_collision["mean_collision_fraction"].max()) if not target_collision.empty else float("nan")
    else:
        collision_max = float("nan")
    g4_pass = bool(mismatch_qualifying >= int(th["g4_min_configs"]) and (not np.isfinite(collision_max) or collision_max <= float(th["g4_max_target_collision_fraction"])))
    gates.append({
        "gate": "G4_IDENTIFIABILITY_AND_ROBUSTNESS", "pass": g4_pass,
        "qualifying_mismatch_configs": mismatch_qualifying,
        "max_mean_target_piecewise_collision_fraction": collision_max,
        "details": mismatch_details,
    })

    expected_files = [
        "exactness_trials.csv", "collision_aggregate.csv", "rank_separation_aggregate.csv",
        "performance_trials.csv", "performance_aggregate.csv", "mismatch_trials.csv",
        "mismatch_aggregate.csv", "certificate_aggregate.csv", "REPRODUCIBILITY_MANIFEST.json",
    ]
    missing_files = [f for f in expected_files if not (run_dir / f).exists()]
    g5_pass = not missing_files
    gates.append({"gate": "G5_REPRODUCIBILITY", "pass": g5_pass, "missing_files": missing_files})

    gate_map = {g["gate"]: bool(g["pass"]) for g in gates}
    profile = cfg["profile"]
    if profile == "smoke":
        verdict = "INCONCLUSIVE_COMPUTE_OR_STATISTICS"
        rationale = "Smoke profile validates installation and logic but is not powered for a field-potential verdict."
    elif not g0_pass:
        verdict = "STOP_CURRENT_FORM"
        rationale = "Exactness or implementation gate G0 failed."
    elif g0_pass and g1_pass and g2_pass and g3_pass and g4_pass and g5_pass:
        verdict = "CONTINUE_FIELD_DEFINING_CANDIDATE"
        rationale = "All mandatory gates and the nontrivial fiber-effect gate passed."
    elif g0_pass and g2_pass and g4_pass and not g3_pass:
        verdict = "PIVOT_THEORETICAL_OR_APPROXIMATE_ONLY"
        rationale = "Candidate search is promising, but the exact stopping certificate is not practically effective."
    elif g0_pass and g2_pass and g5_pass:
        verdict = "CONTINUE_SIGNIFICANT_BUT_NARROW"
        rationale = "A matched-baseline advantage exists, but robustness, identifiability, or fiber significance remains limited."
    else:
        structural = False
        if not rank_agg.empty:
            structural = float(rank_agg["median_log2_coordinate_separation_lower_bound"].max()) >= float(th["structural_min_log2_separation"])
        if structural and g0_pass:
            verdict = "INCONCLUSIVE_COMPUTE_OR_STATISTICS"
            rationale = "The structural premise survives, but matched end-to-end gates do not yet support continuation or rejection."
        else:
            verdict = "STOP_CURRENT_FORM"
            rationale = "The proposal did not establish a matched, robust scaling advantage under the frozen gate."

    result = {
        "schema_version": "1.0",
        "profile": profile,
        "verdict": verdict,
        "rationale": rationale,
        "gates": gates,
        "gate_pass_map": gate_map,
        "claim_warning": "This is an early evidence gate, not proof that future work will be field-defining.",
    }
    gate_df = pd.DataFrame([
        {"gate": g["gate"], "pass": g["pass"], "summary": json.dumps({k: v for k, v in g.items() if k not in {"gate", "pass", "details"}}, default=str)}
        for g in gates
    ])
    return result, gate_df


def markdown_report(verdict: dict[str, Any], run_dir: Path) -> str:
    lines = [
        "# LS-GRAND field-potential gate report",
        "",
        f"**Profile:** `{verdict['profile']}`",
        "",
        f"**Automated verdict:** `{verdict['verdict']}`",
        "",
        verdict["rationale"],
        "",
        "## Gate outcomes",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    for g in verdict["gates"]:
        lines.append(f"| {g['gate']} | {'PASS' if g['pass'] else 'FAIL'} |")
    lines.extend([
        "",
        "## Scientific interpretation",
        "",
        "A positive result is meaningful only if the matched latent-OSD baseline is competitive in BLER and the LS-GRAND advantage remains after state-codeword likelihood operations, membership queries, caps, and wall-clock time are shown separately.  A failed certificate gate means that the exact theorem may remain correct while the practical exact-decoding claim fails.",
        "",
        "## Files to inspect",
        "",
        "- `gate_status.csv`",
        "- `exactness_trials.csv`",
        "- `performance_aggregate.csv`",
        "- `certificate_aggregate.csv`",
        "- `collision_aggregate.csv`",
        "- `rank_separation_aggregate.csv`",
        "- `mismatch_aggregate.csv`",
        "- `REPRODUCIBILITY_MANIFEST.json`",
        "",
        "## Claim discipline",
        "",
        verdict["claim_warning"],
        "",
    ])
    return "\n".join(lines)


def run_campaign(package_root: Path, config_path: Path, output_root: Path, run_name: str | None = None, unit_tests_passed: bool = True) -> Path:
    cfg = load_config(config_path)
    run_name = run_name or f"LS_GRAND_Field_Potential_Gate_{cfg['profile']}_{utc_stamp()}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "FROZEN_CONFIG.json")
    write_json(run_dir / "RUN_STATE.json", {"status": "RUNNING", "started_utc": utc_stamp(), "profile": cfg["profile"]})

    manifest = environment_manifest(package_root, run_dir, cfg)
    write_json(run_dir / "REPRODUCIBILITY_MANIFEST.json", manifest)

    exact_df = run_exactness(cfg, run_dir)
    collision_agg = run_collisions(cfg, run_dir)
    rank_agg = run_rank_separation(cfg, run_dir)
    perf_df = run_performance_stage(cfg, run_dir, "performance", "performance_trials.csv")
    perf_agg = aggregate_decoder_trials(perf_df)
    atomic_csv(perf_agg, run_dir / "performance_aggregate.csv")
    cert_agg = certificate_aggregate(perf_df)
    atomic_csv(cert_agg, run_dir / "certificate_aggregate.csv")
    mismatch_df = run_performance_stage(cfg, run_dir, "mismatch", "mismatch_trials.csv")
    mismatch_agg = aggregate_decoder_trials(mismatch_df)
    atomic_csv(mismatch_agg, run_dir / "mismatch_aggregate.csv")

    save_figures(perf_agg, cert_agg, rank_agg, collision_agg, run_dir / "figures")
    verdict, gate_df = adjudicate(
        cfg, run_dir, exact_df, perf_df, mismatch_df, cert_agg, rank_agg,
        collision_agg, unit_tests_passed=unit_tests_passed,
    )
    atomic_csv(gate_df, run_dir / "gate_status.csv")
    write_json(run_dir / "FINAL_GATE_VERDICT.json", verdict)
    (run_dir / "FINAL_GATE_REPORT.md").write_text(markdown_report(verdict, run_dir))
    write_json(run_dir / "RUN_STATE.json", {"status": "COMPLETE", "completed_utc": utc_stamp(), "profile": cfg["profile"], "verdict": verdict["verdict"]})
    return run_dir
