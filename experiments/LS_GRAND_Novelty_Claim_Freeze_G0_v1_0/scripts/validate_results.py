from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

REQUIRED = [
    "FINAL_G0_NOVELTY_VERDICT.json",
    "FINAL_G0_NOVELTY_REPORT.md",
    "CLAIM_MATRIX.csv",
    "SOURCE_MATRIX.csv",
    "SEARCH_QUERY_REGISTER.csv",
    "ALLOWED_CLAIMS.md",
    "BLOCKED_CLAIMS.md",
    "NEXT_GATE_CONTRACT.md",
    "HUMAN_REVIEW_CHECKLIST.md",
    "FROZEN_AUDIT_INPUT.json",
    "REPRODUCIBILITY_MANIFEST.json",
    "RUN_STATE.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_results.py RUN_DIR")
    run = Path(sys.argv[1]).resolve()
    problems = [f"missing {name}" for name in REQUIRED if not (run / name).is_file()]
    verdict = None
    if not problems:
        v = json.loads((run / "FINAL_G0_NOVELTY_VERDICT.json").read_text())
        verdict = v.get("verdict")
        if verdict not in {"PASS_NARROWED_APPROXIMATE_CLAIM_FREEZE", "PREFLIGHT_SCHEMA_ONLY"}:
            problems.append(f"unexpected verdict {verdict}")
        if v.get("field_defining_program_authorized") is not False:
            problems.append("field-defining authorization must be false")
        if v.get("patent_freedom_to_operate_determined") is not False:
            problems.append("FTO must remain undetermined")
        with (run / "CLAIM_MATRIX.csv").open(newline="", encoding="utf-8") as f:
            claims = list(csv.DictReader(f))
        if len(claims) < 10:
            problems.append("claim matrix too small")
        if not any(c["claim_id"] == "C02" and c["classification"] == "ANTICIPATED_BY_THRESHOLD_ALGORITHM" for c in claims):
            problems.append("threshold-algorithm novelty boundary missing")
        if not any(c["claim_id"] == "C06" and c["disposition"] == "ALLOW_PROVISIONALLY" for c in claims):
            problems.append("narrow first-valid claim freeze missing")
        with (run / "SOURCE_MATRIX.csv").open(newline="", encoding="utf-8") as f:
            sources = list(csv.DictReader(f))
        if len(sources) < 15:
            problems.append("source matrix too small")
    hashes = {}
    for path in sorted(run.iterdir()):
        if path.is_file() and path.name not in {"RESULT_SHA256_MANIFEST.json", "VALIDATION_REPORT.json"}:
            hashes[path.name] = {"sha256":sha256(path), "bytes":path.stat().st_size}
    (run / "RESULT_SHA256_MANIFEST.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    report = {"valid":not problems, "run":str(run), "problems":problems, "verdict":verdict}
    (run / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
