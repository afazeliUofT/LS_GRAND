from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .experiments import run_campaign


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_tests(root: Path, log_path: Path | None = None) -> bool:
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-v"]
    proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text)
    else:
        print(text, end="")
    return proc.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LS-GRAND field-potential gate")
    sub = parser.add_subparsers(dest="command", required=True)

    p_test = sub.add_parser("test", help="run deterministic mathematical/unit audit")
    p_test.add_argument("--log", type=Path)

    p_run = sub.add_parser("run", help="run a frozen campaign")
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--output-root", type=Path, required=True)
    p_run.add_argument("--run-name")
    p_run.add_argument("--skip-tests", action="store_true")

    args = parser.parse_args(argv)
    root = package_root()
    if args.command == "test":
        return 0 if run_tests(root, args.log) else 1
    if args.command == "run":
        args.output_root.mkdir(parents=True, exist_ok=True)
        test_log = args.output_root / "LAST_UNIT_TEST_LOG.txt"
        tests_ok = True if args.skip_tests else run_tests(root, test_log)
        if not tests_ok:
            print(f"Unit audit failed; see {test_log}", file=sys.stderr)
            return 2
        run_dir = run_campaign(root, args.config.resolve(), args.output_root.resolve(), args.run_name, unit_tests_passed=tests_ok)
        print(json.dumps({"status": "COMPLETE", "run_dir": str(run_dir)}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
