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
        "FINAL_G4_REPORT.md", "FINAL_G4_VERDICT.json", "FROZEN_CONFIG.json",
        "REPRODUCIBILITY_MANIFEST.json", "RESULT_SHA256_MANIFEST.json",
        "RUN_STATE.json", "G0_CHAIN_LOCK.json", "G3P_CHAIN_LOCK.json",
        "G0_CLAIM_MATRIX_FROZEN.csv", "TRACE_PROVENANCE_AND_SUMMARY.json",
        "code_orbit_summary.csv", "code_orbit_transforms.csv",
        "detector_calibration.csv", "detector_thresholds.json", "detector_summary.csv",
        "performance_trials.csv", "performance_aggregate.csv", "performance_comparisons.csv",
        "no_slip_overhead.csv", "observed_physical_mixture.csv",
        "rate_accounted_controls.csv", "gate_status.csv",
    ]
    problems: list[str] = []
    for name in required:
        path = run / name
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"missing or empty: {name}")
    verdict: dict = {}
    if not problems:
        verdict = json.loads((run / "FINAL_G4_VERDICT.json").read_text())
        cfg = json.loads((run / "FROZEN_CONFIG.json").read_text())
        state = json.loads((run / "RUN_STATE.json").read_text())
        if state.get("status") != "COMPLETE":
            problems.append("RUN_STATE is not COMPLETE")
        allowed = {
            "INCONCLUSIVE_SMOKE_ONLY", "STOP_CLAIM_CHAIN_VIOLATION",
            "INCONCLUSIVE_TRACE_EVENT_COUNT", "STOP_CODE_IDENTIFIABILITY",
            "READY_FOR_FOCUSED_SUBSTANTIAL_PAPER_PROGRAM",
            "CONTINUE_WITH_EXTERNAL_ALARM_ONLY", "NARROW_TO_ONE_STRUCTURED_CODE_PAPER",
            "STOP_APPROXIMATE_ROUTE",
        }
        if verdict.get("verdict") not in allowed:
            problems.append(f"unexpected verdict: {verdict.get('verdict')}")
        if verdict.get("field_defining_verdict_authorized") is not False:
            problems.append("field-defining authorization must remain false")
        if verdict.get("exact_foundational_route") != "STOPPED":
            problems.append("exact foundational route was incorrectly revived")
        g0 = json.loads((run / "G0_CHAIN_LOCK.json").read_text())
        g3p = json.loads((run / "G3P_CHAIN_LOCK.json").read_text())
        if g0.get("source_commit") != "0d2866b091576fe07521172378ded33da79ed545":
            problems.append("G0 commit mismatch")
        if g3p.get("source_commit") != "5edb4083ec30d44061f0020a2701fd189f87df23":
            problems.append("G3P commit mismatch")
        tables = [
            "G0_CLAIM_MATRIX_FROZEN.csv", "code_orbit_summary.csv", "detector_summary.csv",
            "performance_trials.csv", "performance_aggregate.csv", "performance_comparisons.csv",
            "no_slip_overhead.csv", "observed_physical_mixture.csv",
            "rate_accounted_controls.csv", "gate_status.csv",
        ]
        for name in tables:
            try:
                if pd.read_csv(run / name).empty:
                    problems.append(f"empty table: {name}")
            except Exception as exc:
                problems.append(f"cannot parse {name}: {exc}")
        trace = json.loads((run / "TRACE_PROVENANCE_AND_SUMMARY.json").read_text())
        if trace.get("source_anchor") is None:
            problems.append("trace source anchor is missing")
        if trace.get("observed_one_slip_frames", 0) < 0:
            problems.append("invalid trace event count")
        gate_by = {str(g.get("gate")): g for g in verdict.get("gates", [])}
        if not bool(gate_by.get("G0_CHAIN_LOCK", {}).get("pass")):
            problems.append("chain lock gate failed")
        if verdict.get("verdict") == "READY_FOR_FOCUSED_SUBSTANTIAL_PAPER_PROGRAM":
            for gate in [
                "P1_STANDARD_DERIVED_TRACE", "P2_ORBIT_SAFETY", "P3_EVENT_TRIGGER",
                "P4_STRONG_MATCHED_RECOVERY", "P5_NO_SLIP_NEUTRALITY",
                "P6_OBSERVED_TRACE_MIXTURE", "P7_RATE_ACCOUNTED_CONTROLS", "P8_REPRODUCIBILITY",
            ]:
                if not bool(gate_by.get(gate, {}).get("pass")):
                    problems.append(f"ready verdict issued although {gate} failed")
        if list(run.rglob(".venv")):
            problems.append(".venv unexpectedly present in run directory")
        manifest = json.loads((run / "RESULT_SHA256_MANIFEST.json").read_text())
        for rel, meta in manifest.items():
            path = run / rel
            if not path.is_file():
                problems.append(f"hash-manifest file missing: {rel}")
            elif sha256_file(path) != meta.get("sha256"):
                problems.append(f"hash mismatch: {rel}")
    report = {"valid": not problems, "run": str(run), "problems": problems, "verdict": verdict.get("verdict")}
    (run / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
