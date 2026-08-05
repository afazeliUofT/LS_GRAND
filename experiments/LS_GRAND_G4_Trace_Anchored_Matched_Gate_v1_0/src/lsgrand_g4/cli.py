from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path

from .campaign import run_campaign


def run_tests(package_root: Path, log_path: Path | None = None) -> bool:
    loader = unittest.TestLoader()
    suite = loader.discover(str(package_root / "tests"), pattern="test_*.py")
    if log_path is None:
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        print(log_path.read_text(), end="")
    return bool(result.wasSuccessful())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lsgrand-g4")
    sub = parser.add_subparsers(dest="command", required=True)
    ptest = sub.add_parser("test")
    ptest.add_argument("--package-root", type=Path, required=True)
    ptest.add_argument("--log", type=Path)
    prun = sub.add_parser("run")
    prun.add_argument("--package-root", type=Path, required=True)
    prun.add_argument("--config", type=Path, required=True)
    prun.add_argument("--output-root", type=Path, required=True)
    prun.add_argument("--run-name", required=True)
    prun.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "test":
        return 0 if run_tests(args.package_root, args.log) else 1
    tests_ok = True if args.skip_tests else run_tests(args.package_root)
    if not tests_ok:
        return 2
    run_dir = run_campaign(args.package_root, args.config, args.output_root, args.run_name, unit_tests_passed=tests_ok)
    verdict = json.loads((run_dir / "FINAL_G4_VERDICT.json").read_text())
    print(json.dumps({"status": "COMPLETE", "run_dir": str(run_dir), "verdict": verdict["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
