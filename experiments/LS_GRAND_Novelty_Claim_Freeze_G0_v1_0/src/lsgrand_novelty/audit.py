from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PACKAGE_DIRNAME = "LS_GRAND_Novelty_Claim_Freeze_G0_v1_0"
EXPECTED_BASE_COMMIT = "ed5a4b4be5dbabeed98691043dd64db6d0415dae"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_capture(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def load_inputs(root: Path) -> dict[str, Any]:
    return {
        "sources": read_json(root / "data" / "sources.json"),
        "claims": read_json(root / "data" / "claims.json"),
        "language": read_json(root / "data" / "claim_language.json"),
        "queries": read_json(root / "data" / "search_queries.json"),
        "scope": read_json(root / "data" / "frozen_scope.json"),
        "next_gate": read_json(root / "data" / "next_gate_contract.json"),
    }


def validate_inputs(inputs: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    sources = inputs["sources"].get("sources", [])
    claims = inputs["claims"].get("claims", [])
    source_ids = [s.get("source_id") for s in sources]
    claim_ids = [c.get("claim_id") for c in claims]
    if len(source_ids) != len(set(source_ids)):
        problems.append("duplicate source_id")
    if len(claim_ids) != len(set(claim_ids)):
        problems.append("duplicate claim_id")
    if len(sources) < 15:
        problems.append("source register is too small")
    if len(claims) < 10:
        problems.append("claim register is too small")
    known = set(source_ids)
    for claim in claims:
        refs = claim.get("closest_sources", [])
        if not refs:
            problems.append(f"claim {claim.get('claim_id')} has no closest source")
        missing = sorted(set(refs) - known)
        if missing:
            problems.append(f"claim {claim.get('claim_id')} references missing sources {missing}")
    blocked = inputs["language"].get("blocked_phrases", [])
    if len(blocked) < 8:
        problems.append("blocked claim-language register is incomplete")
    if inputs["scope"].get("base_commit") != EXPECTED_BASE_COMMIT:
        problems.append("frozen scope base commit mismatch")
    return problems


def write_claim_matrix(path: Path, claims: list[dict[str, Any]]) -> None:
    fields = ["claim_id", "claim", "classification", "disposition", "closest_sources", "reason"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in claims:
            row = dict(c)
            row["closest_sources"] = ";".join(c.get("closest_sources", []))
            w.writerow({k: row.get(k, "") for k in fields})


def write_source_matrix(path: Path, sources: list[dict[str, Any]]) -> None:
    fields = ["source_id", "type", "year", "title", "authors", "identifier", "url", "primary", "technical_relevance", "novelty_effect"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in sources:
            w.writerow({k: s.get(k, "") for k in fields})


def write_query_register(path: Path, queries: list[dict[str, Any]]) -> None:
    fields = ["query_id", "domain", "query"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for q in queries:
            w.writerow({k: q.get(k, "") for k in fields})


def classify(inputs: dict[str, Any]) -> dict[str, Any]:
    claims = inputs["claims"]["claims"]
    by_id = {c["claim_id"]: c for c in claims}
    exact_block = all(by_id[c]["disposition"].startswith("BLOCK") for c in ["C01", "C02", "C11"])
    narrow_ok = all(by_id[c]["disposition"] in {"ALLOW_PROVISIONALLY", "ALLOW_ONLY_AS_NARROW_COMBINATION", "ALLOW_AS_SUPPORTING_CONTRIBUTION"} for c in ["C05", "C06", "C07", "C08"])
    patent_sources = [s for s in inputs["sources"]["sources"] if s["type"] == "patent"]
    patent_flag = len(patent_sources) >= 6
    language_ok = len(inputs["language"].get("blocked_phrases", [])) >= 8 and bool(inputs["language"].get("primary_claim"))
    gate_ready = len(inputs["next_gate"].get("matched_baselines", [])) >= 5 and len(inputs["next_gate"].get("stop_conditions", [])) >= 5
    gates = [
        {"gate":"N0_INPUT_AND_BASE_EVIDENCE_INTEGRITY", "pass":True, "base_commit":EXPECTED_BASE_COMMIT},
        {"gate":"N1_FOUNDATIONAL_EXACT_NOVELTY", "pass":False, "scientific_disposition":"STOP_AS_PRIMARY_NOVELTY_ROUTE", "reason":"C01/C02/C11 are anticipated or rejected."},
        {"gate":"N2_NARROW_APPROXIMATE_COMBINATION", "pass":bool(narrow_ok), "scientific_disposition":"CONTINUE_ONE_BOUNDED_GATE" if narrow_ok else "STOP"},
        {"gate":"N3_PATENT_PROXIMITY_AND_FTO", "pass":False, "patent_sources":len(patent_sources), "freedom_to_operate":"NOT_DETERMINED", "reason":"Close active/known patent families require counsel."},
        {"gate":"N4_CLAIM_DISCIPLINE", "pass":bool(language_ok)},
        {"gate":"N5_NEXT_GATE_READINESS", "pass":bool(gate_ready)},
        {"gate":"N6_REPRODUCIBILITY", "pass":True, "validator_required":True},
    ]
    verdict = "PASS_NARROWED_APPROXIMATE_CLAIM_FREEZE" if exact_block and narrow_ok and patent_flag and language_ok and gate_ready else "STOP_OR_REPAIR_CLAIM_FREEZE"
    return {
        "schema_version":"1.0",
        "verdict":verdict,
        "rationale":"The generic exact marginalization/stopping thesis is not a defensible novelty pillar, but the precise LS-FV/LS-A2 orbit-screened sparse latent-cause combination remains a provisional narrow candidate for one matched physical gate.",
        "base_commit":EXPECTED_BASE_COMMIT,
        "field_defining_program_authorized":False,
        "exact_foundational_paper_authorized":False,
        "next_bounded_numerical_gate_authorized":bool(verdict == "PASS_NARROWED_APPROXIMATE_CLAIM_FREEZE"),
        "patent_freedom_to_operate_determined":False,
        "frozen_practical_focus":"event-triggered LS-FV with adaptively invoked LS-A2 on orbit-screened codes",
        "gates":gates,
    }


def write_markdown_outputs(run_dir: Path, inputs: dict[str, Any], verdict: dict[str, Any]) -> None:
    lang = inputs["language"]
    decision = inputs["scope"]["scientific_decision"]
    report = f"""# LS-GRAND G0 novelty and claim-freeze report

## Final classification

**{verdict['verdict']}**

{verdict['rationale']}

## Signed project decision

- Foundational exact route: **{decision['foundational_exact_route']}**
- Current exact certificate route: **{decision['current_exact_certificate_route']}**
- Narrow approximate route: **{decision['narrow_approximate_route']}**
- Field-defining program: **{decision['field_defining_program']}**
- Patent freedom to operate: **{decision['patent_freedom_to_operate']}**

## Why the broad novelty thesis is stopped

1. Hidden-state marginalization is established by BCJR/sum-product inference.
2. The original sum-of-queue-heads exact stop is a Threshold Algorithm specialization.
3. Code-aided phase-hypothesis/cycle-slip resolution has substantial literature and patent prior art.
4. List construction followed by complete channel/state rescoring is established.
5. The current exact certificate failed the practical-usefulness gate at the base commit.

## Narrow claim that remains authorized

{lang['primary_claim']}

### Required qualifier

{lang['novelty_qualifier']}

## What happens next

Exactly one structured-code, optimized matched-baseline, physically anchored gate is authorized.  Broad theory,
custom code design, hardware, multi-user generalization, and field-defining language remain blocked.

## Legal limitation

This report is a scientific novelty screen.  It is not legal advice, a patent-validity opinion, or a
freedom-to-operate opinion.
"""
    (run_dir / "FINAL_G0_NOVELTY_REPORT.md").write_text(report, encoding="utf-8")
    allowed = "# Allowed claims\n\n## Primary claim\n\n" + lang["primary_claim"] + "\n\n## Required qualifier\n\n" + lang["novelty_qualifier"] + "\n\n## Supporting statements\n\n" + "".join(f"- {x}\n" for x in lang["allowed_supporting_statements"])
    (run_dir / "ALLOWED_CLAIMS.md").write_text(allowed, encoding="utf-8")
    blocked = "# Blocked claims and phrases\n\n" + "".join(f"- {x}\n" for x in lang["blocked_phrases"])
    (run_dir / "BLOCKED_CLAIMS.md").write_text(blocked, encoding="utf-8")
    ng = inputs["next_gate"]
    next_text = "# Next gate contract\n\n" + f"**Gate:** `{ng['gate_name']}`\n\n**Objective:** {ng['objective']}\n\n## Algorithms\n\n" + "".join(f"- {x}\n" for x in ng["algorithms"]["primary"]) + "\n## Matched baselines\n\n" + "".join(f"- {x}\n" for x in ng["matched_baselines"]) + "\n## Pass conditions\n\n" + "".join(f"- {x}\n" for x in ng["pass_conditions"]) + "\n## Stop conditions\n\n" + "".join(f"- {x}\n" for x in ng["stop_conditions"])
    (run_dir / "NEXT_GATE_CONTRACT.md").write_text(next_text, encoding="utf-8")
    checklist = """# Human review checklist

- [ ] Confirm C01/C02/C11 are not used as novelty claims.
- [ ] Independently inspect the closest phase/FEC patent claims.
- [ ] Search citation and patent families for C06/C07 synonyms.
- [ ] Confirm the allowed primary statement remains technically accurate.
- [ ] Confirm LS-FV and LS-A2 are the only practical flagship modes.
- [ ] Confirm the next gate includes event-triggered BCJR and code-specific baselines.
- [ ] Confirm a physical cycle-slip model or trace exists before launching the next simulation package.
- [ ] Record one decision: accept narrow freeze, narrow further, or stop.
"""
    (run_dir / "HUMAN_REVIEW_CHECKLIST.md").write_text(checklist, encoding="utf-8")


def run_gate(output_root: Path, run_name: str, profile: str = "gate") -> Path:
    root = package_root()
    inputs = load_inputs(root)
    problems = validate_inputs(inputs)
    if problems:
        raise RuntimeError("input validation failed: " + "; ".join(problems))
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    verdict = classify(inputs)
    if profile == "smoke":
        verdict = dict(verdict)
        verdict["verdict"] = "PREFLIGHT_SCHEMA_ONLY"
        verdict["next_bounded_numerical_gate_authorized"] = False
        verdict["rationale"] = "Smoke/preflight mode validates schemas and rendering only."
    write_json(run_dir / "FINAL_G0_NOVELTY_VERDICT.json", verdict)
    write_claim_matrix(run_dir / "CLAIM_MATRIX.csv", inputs["claims"]["claims"])
    write_source_matrix(run_dir / "SOURCE_MATRIX.csv", inputs["sources"]["sources"])
    write_query_register(run_dir / "SEARCH_QUERY_REGISTER.csv", inputs["queries"]["queries"])
    write_json(run_dir / "FROZEN_AUDIT_INPUT.json", inputs)
    write_markdown_outputs(run_dir, inputs, verdict)
    manifest = {
        "created_utc":utc_stamp(),
        "profile":profile,
        "python":sys.version,
        "platform":platform.platform(),
        "package_root":str(root),
        "run_dir":str(run_dir),
        "git_commit_at_run_start":git_capture(["git","rev-parse","HEAD"], root),
        "git_branch":git_capture(["git","branch","--show-current"], root),
        "git_remote":git_capture(["git","remote","get-url","origin"], root),
        "required_base_commit":EXPECTED_BASE_COMMIT,
        "source_count":len(inputs["sources"]["sources"]),
        "claim_count":len(inputs["claims"]["claims"]),
    }
    write_json(run_dir / "REPRODUCIBILITY_MANIFEST.json", manifest)
    write_json(run_dir / "RUN_STATE.json", {"status":"COMPLETE","profile":profile,"run_name":run_name,"verdict":verdict["verdict"]})
    return run_dir
