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
    if log_path is None:
        print(text, end="")
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text)
    return proc.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LS-GRAND decisive pivot gate v1.1")
    sub = parser.add_subparsers(dest="command", required=True)
    p_test = sub.add_parser("test")
    p_test.add_argument("--log", type=Path)
    p_run = sub.add_parser("run")
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--output-root", type=Path, required=True)
    p_run.add_argument("--run-name")
    p_run.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    root = package_root()
    if args.command == "test":
        return 0 if run_tests(root, args.log) else 1
    tests_ok = True if args.skip_tests else run_tests(root, args.output_root / "LAST_V11_UNIT_TEST_LOG.txt")
    if not tests_ok:
        print("Unit tests failed; campaign not started.", file=sys.stderr)
        return 2
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_campaign(root, args.config.resolve(), args.output_root.resolve(), args.run_name, unit_tests_passed=tests_ok)
    print(json.dumps({"status": "COMPLETE", "run_dir": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
