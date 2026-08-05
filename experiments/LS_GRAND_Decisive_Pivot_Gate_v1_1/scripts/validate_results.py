#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REQUIRED = [
    "FINAL_DECISIVE_VERDICT.json", "FINAL_DECISIVE_REPORT.md", "gate_status.csv",
    "exact_fiber_trials.csv", "exact_fiber_aggregate.csv", "exact_fiber_summary.json",
    "rank_probe_trials.csv", "rank_probe_aggregate.csv",
    "orbit_transform_trials.csv", "orbit_code_summary.csv", "orbit_aggregate.csv",
    "performance_trials.csv", "performance_aggregate.csv", "performance_comparisons.csv",
    "certificate_comparisons.csv", "REPRODUCIBILITY_MANIFEST.json",
    "RESULT_SHA256_MANIFEST.json", "RUN_STATE.json", "FROZEN_CONFIG.json",
    "figures/actual_noisy_rank_ratio.png", "figures/target_orbit_safe_fraction.png",
    "figures/candidate_frontier.png", "figures/certificate_overhead.png",
]
SCHEMAS = {
    "exact_fiber_trials.csv": {"trial_uid", "cert_matches_marginal", "marginal_error", "first_error"},
    "rank_probe_trials.csv": {"trial_uid", "latent_cap_hit", "plain_cap_hit", "rank_ratio_lower_bound"},
    "orbit_code_summary.csv": {"case_id", "code_replicate", "orbit_safe", "max_collision_fraction"},
    "performance_trials.csv": {"case_id", "trial_uid", "decoder", "decoded_correct", "wall_seconds"},
    "gate_status.csv": {"gate", "pass"},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_results.py RUN_DIRECTORY", file=sys.stderr)
        return 2
    run = Path(sys.argv[1]).resolve()
    problems: list[str] = []
    missing = [name for name in REQUIRED if not (run / name).is_file()]
    problems.extend(f"missing:{name}" for name in missing)
    if not missing:
        state = json.loads((run / "RUN_STATE.json").read_text())
        verdict = json.loads((run / "FINAL_DECISIVE_VERDICT.json").read_text())
        if state.get("status") != "COMPLETE":
            problems.append(f"run_state:{state.get('status')}")
        if not verdict.get("verdict"):
            problems.append("empty_verdict")
        if verdict.get("field_defining_verdict_authorized") is not False:
            problems.append("unsafe_field_defining_authorization")
        manifest = json.loads((run / "RESULT_SHA256_MANIFEST.json").read_text())
        for item in manifest.get("files", []):
            p = run / item["path"]
            if not p.is_file():
                problems.append(f"manifest_missing:{item['path']}")
            elif p.stat().st_size != int(item["bytes"]):
                problems.append(f"size_mismatch:{item['path']}")
            elif sha256_file(p) != item["sha256"]:
                problems.append(f"hash_mismatch:{item['path']}")
        for name, required_cols in SCHEMAS.items():
            df = pd.read_csv(run / name)
            absent = required_cols - set(df.columns)
            if absent:
                problems.append(f"schema:{name}:{sorted(absent)}")
            if len(df) == 0:
                problems.append(f"empty:{name}")
        gates = pd.read_csv(run / "gate_status.csv")
        expected_gates = {
            "F0_EXACTNESS", "F1_MARGINAL_FIBER_RELEVANCE", "F2_EMPIRICAL_RANK_COMPRESSION",
            "F3_APPROXIMATE_CANDIDATE_FRONTIER", "F4_EXACT_CERTIFICATE_USEFULNESS",
            "F5_TARGET_ORBIT_IDENTIFIABILITY", "F6_REPRODUCIBILITY",
        }
        if set(gates["gate"].astype(str)) != expected_gates:
            problems.append("gate_set_mismatch")
        repro = json.loads((run / "REPRODUCIBILITY_MANIFEST.json").read_text())
        if not repro.get("git_commit_at_run_start"):
            problems.append("missing_git_commit_at_run_start")
        if not repro.get("config"):
            problems.append("missing_frozen_config_in_manifest")
    report = {
        "valid": not problems,
        "run": str(run),
        "problems": problems,
        "verdict": None if missing else json.loads((run / "FINAL_DECISIVE_VERDICT.json").read_text()).get("verdict"),
    }
    (run / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
