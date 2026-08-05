#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = [
    "FINAL_GATE_VERDICT.json", "FINAL_GATE_REPORT.md", "gate_status.csv",
    "exactness_trials.csv", "performance_aggregate.csv", "certificate_aggregate.csv",
    "collision_aggregate.csv", "rank_separation_aggregate.csv", "mismatch_aggregate.csv",
    "REPRODUCIBILITY_MANIFEST.json", "RUN_STATE.json",
]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_results.py RUN_DIRECTORY", file=sys.stderr)
        return 2
    run = Path(sys.argv[1]).resolve()
    missing = [name for name in REQUIRED if not (run / name).is_file()]
    if missing:
        print(json.dumps({"valid": False, "missing": missing}, indent=2))
        return 1
    state = json.loads((run / "RUN_STATE.json").read_text())
    verdict = json.loads((run / "FINAL_GATE_VERDICT.json").read_text())
    valid = state.get("status") == "COMPLETE" and bool(verdict.get("verdict"))
    print(json.dumps({"valid": valid, "run": str(run), "verdict": verdict.get("verdict")}, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
