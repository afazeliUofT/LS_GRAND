from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stats import wilson_interval


def environment_manifest(package_root: Path, run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    def cmd(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
        except Exception:
            return None

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_commit": cmd(["git", "rev-parse", "HEAD"]),
        "git_branch": cmd(["git", "branch", "--show-current"]),
        "git_remote": cmd(["git", "remote", "get-url", "origin"]),
        "package_root": str(package_root.resolve()),
        "run_dir": str(run_dir.resolve()),
        "config": config,
    }


def aggregate_decoder_trials(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    keys = [
        c for c in [
            "experiment", "config_id", "n", "k", "rate", "family", "snr_db",
            "true_slip_prob", "model_slip_prob", "forced_location_fraction",
            "two_slip_fraction", "decoder"
        ] if c in df.columns
    ]
    rows: list[dict[str, Any]] = []
    for group_key, g in df.groupby(keys, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(keys, group_key))
        trials = len(g)
        errors = int((~g["decoded_correct"].astype(bool)).sum())
        lo, hi = wilson_interval(errors, trials)
        row.update(
            trials=trials,
            errors=errors,
            bler=errors / trials,
            bler_wilson_low=lo,
            bler_wilson_high=hi,
            cap_rate=float(g["cap_hit"].astype(float).mean()),
            certification_rate=float(g["certified"].astype(float).mean()),
        )
        for col in [
            "membership_queries", "residual_patterns_generated", "latent_queues_touched",
            "queue_touch_fraction", "valid_witnesses", "unique_codewords_seen",
            "complete_marginal_scores", "state_codeword_likelihoods", "wall_seconds",
            "certificate_query_overhead"
        ]:
            if col in g:
                vals = pd.to_numeric(g[col], errors="coerce")
                row[f"median_{col}"] = float(vals.median())
                row[f"p90_{col}"] = float(vals.quantile(0.9))
                row[f"mean_{col}"] = float(vals.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def save_figures(
    performance_agg: pd.DataFrame,
    certificate_agg: pd.DataFrame,
    rank_agg: pd.DataFrame,
    collision_agg: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not performance_agg.empty:
        for config_id, g in performance_agg.groupby("config_id"):
            fig, ax = plt.subplots(figsize=(7.0, 4.4))
            gg = g.sort_values("decoder")
            x = np.arange(len(gg))
            ax.bar(x, gg["bler"].to_numpy())
            ax.set_xticks(x)
            ax.set_xticklabels(gg["decoder"].tolist(), rotation=30, ha="right")
            ax.set_ylabel("Block error rate")
            ax.set_title(str(config_id))
            fig.tight_layout()
            fig.savefig(figures_dir / f"bler_{config_id}.png", dpi=180)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7.0, 4.4))
            metric = "median_state_codeword_likelihoods"
            vals = np.maximum(gg.get(metric, pd.Series(np.nan, index=gg.index)).to_numpy(dtype=float), 1.0)
            ax.bar(x, vals)
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(gg["decoder"].tolist(), rotation=30, ha="right")
            ax.set_ylabel("Median state-codeword likelihood operations")
            ax.set_title(str(config_id))
            fig.tight_layout()
            fig.savefig(figures_dir / f"work_{config_id}.png", dpi=180)
            plt.close(fig)

    if not certificate_agg.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        x = np.arange(len(certificate_agg))
        ax.bar(x, certificate_agg["median_queue_touch_fraction"].to_numpy())
        ax.set_xticks(x)
        ax.set_xticklabels(certificate_agg["config_id"].tolist(), rotation=30, ha="right")
        ax.set_ylabel("Median latent-queue touch fraction")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(figures_dir / "certificate_queue_touch_fraction.png", dpi=180)
        plt.close(fig)

    if not rank_agg.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        for label, g in rank_agg.groupby("location_label"):
            gg = g.sort_values("n")
            ax.plot(gg["n"], gg["median_log2_coordinate_separation_lower_bound"], marker="o", label=str(label))
        ax.set_xlabel("Blocklength n (bits)")
        ax.set_ylabel("Median log2 coordinate-separation lower bound")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "rank_coordinate_separation.png", dpi=180)
        plt.close(fig)

    if not collision_agg.empty:
        top = collision_agg.sort_values("mean_collision_fraction", ascending=False).head(24)
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        labels = (top["family"].astype(str) + ":" + top["transform"].astype(str)).tolist()
        x = np.arange(len(top))
        ax.bar(x, top["mean_collision_fraction"].to_numpy())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=70, ha="right")
        ax.set_ylabel("Mean affine-orbit collision fraction")
        fig.tight_layout()
        fig.savefig(figures_dir / "affine_orbit_collisions.png", dpi=180)
        plt.close(fig)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
