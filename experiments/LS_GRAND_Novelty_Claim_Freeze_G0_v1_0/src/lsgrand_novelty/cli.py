from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .audit import run_gate


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("test")
    t.add_argument("--log", required=True)
    r = sub.add_parser("run")
    r.add_argument("--profile", choices=["smoke", "gate"], default="gate")
    r.add_argument("--output-root", required=True)
    r.add_argument("--run-name", required=True)
    args = p.parse_args()
    if args.cmd == "test":
        root = Path(__file__).resolve().parents[2]
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-v"],
            cwd=root, text=True, capture_output=True
        )
        text = proc.stdout + proc.stderr
        Path(args.log).write_text(text, encoding="utf-8")
        print(text, end="")
        return proc.returncode
    run_dir = run_gate(Path(args.output_root), args.run_name, args.profile)
    print(f'{{"status":"COMPLETE","run_dir":"{run_dir}"}}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
