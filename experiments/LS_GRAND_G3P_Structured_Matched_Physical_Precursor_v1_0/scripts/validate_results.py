#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_results.py RUN_DIR", file=sys.stderr)
        return 2
    run = Path(sys.argv[1]).resolve()
    required = [
        "FINAL_G3P_REPORT.md",
        "FINAL_G3P_VERDICT.json",
        "FROZEN_CONFIG.json",
        "REPRODUCIBILITY_MANIFEST.json",
        "RESULT_SHA256_MANIFEST.json",
        "RUN_STATE.json",
        "G0_CHAIN_LOCK.json",
        "G0_CLAIM_MATRIX_FROZEN.csv",
        "code_orbit_summary.csv",
        "code_orbit_transforms.csv",
        "detector_calibration.csv",
        "detector_thresholds.json",
        "detector_summary.csv",
        "performance_trials.csv",
        "performance_aggregate.csv",
        "performance_comparisons.csv",
        "no_slip_overhead.csv",
        "physical_mixture.csv",
        "gate_status.csv",
    ]
    problems: list[str] = []
    for name in required:
        path = run / name
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"missing or empty: {name}")
    verdict = {}
    cfg = {}
    if not problems:
        verdict = json.loads((run / "FINAL_G3P_VERDICT.json").read_text())
        cfg = json.loads((run / "FROZEN_CONFIG.json").read_text())
        state = json.loads((run / "RUN_STATE.json").read_text())
        if state.get("status") != "COMPLETE":
            problems.append("RUN_STATE is not COMPLETE")
        allowed = {
            "INCONCLUSIVE_SMOKE_ONLY",
            "STOP_CLAIM_CHAIN_VIOLATION",
            "STOP_CODE_IDENTIFIABILITY",
            "STOP_REPRODUCIBILITY",
            "CONTINUE_TO_FINAL_PHYSICAL_MATCHED_GATE",
            "CONTINUE_CONDITIONAL_ON_EXTERNAL_ALARM_ONLY",
            "NARROW_TO_APPLICATION_ONLY",
            "STOP_APPROXIMATE_ROUTE",
        }
        if verdict.get("verdict") not in allowed:
            problems.append(f"unexpected verdict: {verdict.get('verdict')}")
        if verdict.get("field_defining_verdict_authorized") is not False:
            problems.append("field-defining authorization must remain false")
        chain = json.loads((run / "G0_CHAIN_LOCK.json").read_text())
        if chain.get("source_commit") != "0d2866b091576fe07521172378ded33da79ed545":
            problems.append("G0 chain source commit mismatch")
        if chain.get("source_verdict") != "PASS_NARROWED_APPROXIMATE_CLAIM_FREEZE":
            problems.append("G0 chain source verdict mismatch")
        gate_by_name = {str(g.get("gate")): g for g in verdict.get("gates", [])}
        if not bool(gate_by_name.get("G0_CHAIN_LOCK_AND_CLAIM_DISCIPLINE", {}).get("pass")):
            problems.append("G0 chain/claim gate did not pass")
        for csv_name in [
            "G0_CLAIM_MATRIX_FROZEN.csv", "code_orbit_summary.csv", "detector_summary.csv",
            "performance_trials.csv", "performance_aggregate.csv", "performance_comparisons.csv",
            "no_slip_overhead.csv", "physical_mixture.csv", "gate_status.csv",
        ]:
            try:
                if pd.read_csv(run / csv_name).empty:
                    problems.append(f"empty table: {csv_name}")
            except Exception as exc:
                problems.append(f"cannot parse {csv_name}: {exc}")
        comparisons = pd.read_csv(run / "performance_comparisons.csv")
        min_eligible = int(cfg["thresholds"]["min_eligible_work_pairs"])
        for gate in verdict.get("gates", []):
            if gate.get("gate") == "C3_APPROXIMATE_STRUCTURED_CODE_FRONTIER":
                for row in gate.get("qualifying_rows", []):
                    if int(row.get("eligible_work_pairs", -1)) < min_eligible:
                        problems.append("C3 qualifying row violates eligible-work-pair threshold")
            if gate.get("gate") == "C4_CODE_SPECIFIC_BASELINES":
                for row in gate.get("details", []):
                    if bool(row.get("qualifies")) and int(row.get("eligible_work_pairs", -1)) < min_eligible:
                        problems.append("C4 qualifying row violates eligible-work-pair threshold")
            if gate.get("gate") == "C6_SYNTHETIC_PHYSICAL_MIXTURE_SCREEN":
                expected = {
                    "BCH64_45": "event_state_sweep_chase_bch",
                    "POLAR64_48": "event_state_sweep_polar_scflip",
                }
                for row in gate.get("details", []):
                    code_id = str(row.get("code_id"))
                    if code_id in expected and row.get("baseline") != expected[code_id]:
                        problems.append(f"C6 uses a non-code-specific baseline for {code_id}")
                    if abs(float(row.get("tail_quantile", 0.0)) - float(cfg["thresholds"].get("physical_tail_quantile", 0.999))) > 1e-12:
                        problems.append(f"C6 tail quantile mismatch for {code_id}")
                    candidate = str(row.get("candidate"))
                    for prior_gate in [
                        "C3_APPROXIMATE_STRUCTURED_CODE_FRONTIER",
                        "C4_CODE_SPECIFIC_BASELINES",
                        "C5_NO_SLIP_NEUTRALITY",
                    ]:
                        modes = gate_by_name.get(prior_gate, {}).get("qualifying_modes_by_code", {}).get(code_id, [])
                        if candidate not in modes:
                            problems.append(f"C6 candidate {candidate} for {code_id} did not pass {prior_gate}")
        if list(run.rglob(".venv")):
            problems.append(".venv unexpectedly present in run directory")
        manifest = json.loads((run / "RESULT_SHA256_MANIFEST.json").read_text())
        for rel, meta in manifest.items():
            path = run / rel
            if not path.is_file():
                problems.append(f"hash-manifest file missing: {rel}")
                continue
            if sha256_file(path) != meta.get("sha256"):
                problems.append(f"hash mismatch: {rel}")
    report = {
        "valid": not problems,
        "run": str(run),
        "problems": problems,
        "verdict": verdict.get("verdict"),
    }
    (run / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
