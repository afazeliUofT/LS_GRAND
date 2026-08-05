#!/usr/bin/env python3
"""Single WSL wrapper for LS-GRAND G4 Trace-Anchored Matched Gate v1.0.

Place this file directly in /home/afazeli2006/LS_GRAND and run it from the
existing VS Code WSL terminal. It never executes a terminal-close command and
never stages the repository virtual environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PACKAGE_FILENAME = "LS_GRAND_G4_Trace_Anchored_Matched_Gate_v1_0.zip"
PACKAGE_DIRNAME = "LS_GRAND_G4_Trace_Anchored_Matched_Gate_v1_0"
EXPECTED_SHA256 = "b7ec31f256ee2ec65ea166caec425e70edd7654d07a1019bbb1a849a11d0f85a"
EXPECTED_REPOSITORY = "afazeliUofT/LS_GRAND"
EXPECTED_REPO_PATH = Path("/home/afazeli2006/LS_GRAND")
REQUIRED_BASE_COMMIT = "5edb4083ec30d44061f0020a2701fd189f87df23"
VALID_PROFILES = {"smoke", "gate", "stress"}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_remote(url: str) -> str:
    value = url.strip().removesuffix(".git")
    if value.startswith("git@github.com:"):
        value = value.split(":", 1)[1]
    elif "github.com/" in value:
        value = value.split("github.com/", 1)[1]
    return value.strip("/")


class Runner:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("")

    def note(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def run(
        self,
        command: Iterable[str | Path],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        check: bool = True,
    ) -> int:
        cmd = [str(x) for x in command]
        self.note("+ " + " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=None,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        with self.log_path.open("a", encoding="utf-8") as log:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
        rc = proc.wait()
        self.note(f"return code: {rc}")
        if check and rc != 0:
            raise RuntimeError(f"command failed with return code {rc}: {' '.join(cmd)}")
        return rc

    def capture(self, command: Iterable[str | Path], *, cwd: Path, check: bool = True) -> str:
        cmd = [str(x) for x in command]
        proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        return proc.stdout.strip()


def discover_repo(runner: Runner) -> Path:
    cwd = Path.cwd().resolve()
    try:
        root = Path(runner.capture(["git", "rev-parse", "--show-toplevel"], cwd=cwd)).resolve()
    except Exception as exc:
        raise RuntimeError(
            f"Run this wrapper from {EXPECTED_REPO_PATH}. Current directory: {cwd}"
        ) from exc
    if root != EXPECTED_REPO_PATH.resolve():
        raise RuntimeError(f"Expected repository {EXPECTED_REPO_PATH}, found {root}.")
    wrapper = Path(__file__).resolve()
    if wrapper.parent != root:
        raise RuntimeError(f"Place the wrapper directly in {root}; current path is {wrapper}.")
    remote = runner.capture(["git", "remote", "get-url", "origin"], cwd=root)
    if normalize_remote(remote).lower() != EXPECTED_REPOSITORY.lower():
        raise RuntimeError(f"Unexpected origin {remote!r}; expected {EXPECTED_REPOSITORY}.")
    runner.run(["git", "cat-file", "-e", f"{REQUIRED_BASE_COMMIT}^{{commit}}"], cwd=root)
    runner.run(["git", "merge-base", "--is-ancestor", REQUIRED_BASE_COMMIT, "HEAD"], cwd=root)
    return root


def candidate_download_dirs() -> list[Path]:
    out: list[Path] = []
    override = os.environ.get("LSGRAND_G4_DOWNLOADS")
    if override:
        out.append(Path(override).expanduser())
    out.extend([
        Path("/mnt/c/Users/alifa/Downloads"),
        Path("/mnt/c/Users/afazeli2006/Downloads"),
        Path.home() / "Downloads",
    ])
    users = Path("/mnt/c/Users")
    if users.is_dir():
        out.extend(sorted(users.glob("*/Downloads")))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in out:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def locate_package(runner: Runner) -> Path:
    found = [d / PACKAGE_FILENAME for d in candidate_download_dirs() if (d / PACKAGE_FILENAME).is_file()]
    if not found:
        searched = "\n  - ".join(str(d / PACKAGE_FILENAME) for d in candidate_download_dirs())
        raise FileNotFoundError(
            f"Could not find {PACKAGE_FILENAME}. Searched:\n  - {searched}\n"
            "Download the ZIP without renaming it, or set LSGRAND_G4_DOWNLOADS."
        )
    valid: list[Path] = []
    for path in sorted(found, key=lambda p: p.stat().st_mtime, reverse=True):
        digest = sha256_file(path)
        runner.note(f"Candidate ZIP: {path} | SHA-256 {digest}")
        if digest == EXPECTED_SHA256:
            valid.append(path)
    if not valid:
        raise RuntimeError(
            "A ZIP with the expected name was found, but none matched the frozen SHA-256. "
            "Delete stale copies and download the supplied ZIP again."
        )
    return valid[0]


def safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (destination / info.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe ZIP path: {info.filename}")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"symbolic links are not permitted: {info.filename}")
        zf.extractall(destination)


def verify_installed_tree(package_dir: Path) -> None:
    manifest = package_dir / "SHA256SUMS"
    if not manifest.is_file():
        raise RuntimeError(f"missing package manifest: {manifest}")
    failures: list[str] = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*").removeprefix("./")
        path = package_dir / rel
        if not path.is_file() or sha256_file(path) != digest:
            failures.append(rel)
    if failures:
        raise RuntimeError(f"package tree hash mismatch: {failures[:12]}")


def install_package_tree(runner: Runner, repo: Path, zip_path: Path) -> Path:
    experiments = repo / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    package_dir = experiments / PACKAGE_DIRNAME
    marker = package_dir / ".installed_package_sha256"
    if package_dir.exists():
        try:
            verify_installed_tree(package_dir)
            marker.write_text(EXPECTED_SHA256 + "\n")
            runner.note(f"Reusing verified package tree: {package_dir}")
            return package_dir
        except Exception:
            backup = experiments / f"{PACKAGE_DIRNAME}.previous_{utc_stamp()}"
            runner.note(f"Preserving nonmatching existing package tree at {backup}")
            package_dir.rename(backup)
    temp = experiments / f".extract_{PACKAGE_DIRNAME}_{utc_stamp()}"
    if temp.exists():
        shutil.rmtree(temp)
    safe_extract(zip_path, temp)
    extracted = temp / PACKAGE_DIRNAME
    if not extracted.is_dir():
        raise RuntimeError("ZIP does not contain the expected single top-level directory")
    extracted.rename(package_dir)
    shutil.rmtree(temp, ignore_errors=True)
    verify_installed_tree(package_dir)
    marker.write_text(EXPECTED_SHA256 + "\n")
    runner.note(f"Installed verified package at {package_dir}")
    return package_dir


def ensure_root_gitignore(repo: Path) -> None:
    path = repo / ".gitignore"
    marker = "# LS-GRAND G4 trace-gate local execution"
    block = f"""
{marker}
.venv/
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
experiments/{PACKAGE_DIRNAME}/.installed_package_sha256
experiments/{PACKAGE_DIRNAME}.previous_*/
experiments/.extract_{PACKAGE_DIRNAME}_*/
""".lstrip()
    current = path.read_text() if path.exists() else ""
    if marker not in current:
        if current and not current.endswith("\n"):
            current += "\n"
        path.write_text(current + block)


def ensure_venv(runner: Runner, repo: Path, package_dir: Path) -> tuple[Path, dict[str, str]]:
    venv = repo / ".venv"
    python = venv / "bin" / "python"
    if not python.is_file():
        runner.note(f"Creating virtual environment: {venv}")
        runner.run([sys.executable, "-m", "venv", venv], cwd=repo)
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = str(venv / "bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    runner.run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], cwd=repo, env=env)
    runner.run([python, "-m", "pip", "install", "-e", package_dir], cwd=repo, env=env)
    return python, env


def check_clean_staging_area(runner: Runner, repo: Path) -> None:
    staged = runner.capture(["git", "diff", "--cached", "--name-only"], cwd=repo)
    if staged.strip():
        raise RuntimeError(
            "The repository already has staged files. Commit or unstage them before running this wrapper.\n"
            f"Pre-existing staged paths:\n{staged}"
        )


def validate_profile() -> str:
    profile = os.environ.get("LSGRAND_G4_PROFILE", "gate").strip().lower()
    if profile not in VALID_PROFILES:
        raise ValueError(f"LSGRAND_G4_PROFILE must be one of {sorted(VALID_PROFILES)}, got {profile!r}")
    return profile


def stage_review_artifacts(
    runner: Runner,
    repo: Path,
    package_dir: Path,
    run_dir: Path,
    wrapper_path: Path,
) -> None:
    rel = package_dir.relative_to(repo)
    package_allowlist = [
        rel / "README.md",
        rel / "PACKAGE_METADATA.json",
        rel / "pyproject.toml",
        rel / "requirements.txt",
        rel / "SHA256SUMS",
        rel / ".gitignore",
        rel / "configs",
        rel / "data",
        rel / "docs",
        rel / "theory",
        rel / "src",
        rel / "tests",
        rel / "scripts",
        rel / "preflight_results",
    ]
    paths: list[Path] = [repo / ".gitignore", wrapper_path, run_dir]
    paths.extend(repo / p for p in package_allowlist)
    existing = [p.relative_to(repo) if p.is_absolute() else p for p in paths if p.exists()]
    runner.run(["git", "add", "--", *existing], cwd=repo)
    staged = runner.capture(["git", "diff", "--cached", "--name-only"], cwd=repo)
    bad = [
        line for line in staged.splitlines()
        if line == ".venv" or line.startswith(".venv/")
        or "/__pycache__/" in f"/{line}/" or line.endswith(".pyc")
        or ".installed_package_sha256" in line
    ]
    if bad:
        runner.run(["git", "restore", "--staged", "--", *bad], cwd=repo, check=False)
        raise RuntimeError(f"local environment files reached staging: {bad}")
    runner.note("Staged review allowlist:\n" + staged)


def regenerate_result_manifest(run_dir: Path) -> None:
    manifest: dict[str, dict[str, int | str]] = {}
    excluded = {"RESULT_SHA256_MANIFEST.json", "VALIDATION_REPORT.json"}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in excluded:
            manifest[str(path.relative_to(run_dir))] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    (run_dir / "RESULT_SHA256_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def commit_and_push(runner: Runner, repo: Path, run_dir: Path) -> str | None:
    staged = runner.capture(["git", "diff", "--cached", "--name-only"], cwd=repo)
    if not staged.strip():
        runner.note("No new review artifacts required a commit.")
        return runner.capture(["git", "rev-parse", "HEAD"], cwd=repo, check=False) or None
    if not runner.capture(["git", "config", "--get", "user.name"], cwd=repo, check=False):
        runner.run(["git", "config", "user.name", "Ali Fazeli"], cwd=repo)
    if not runner.capture(["git", "config", "--get", "user.email"], cwd=repo, check=False):
        runner.run(["git", "config", "user.email", "afazeliUofT@users.noreply.github.com"], cwd=repo)
    branch = runner.capture(["git", "branch", "--show-current"], cwd=repo, check=False).strip()
    if not branch:
        branch = "main"
        runner.run(["git", "checkout", "-b", branch], cwd=repo)
    runner.run(["git", "commit", "-m", f"LS-GRAND G4 trace-anchored matched gate: {run_dir.name}"], cwd=repo)
    commit = runner.capture(["git", "rev-parse", "HEAD"], cwd=repo)
    if os.environ.get("LSGRAND_G4_SKIP_PUSH", "0") == "1":
        runner.note(f"Push skipped by environment; local commit: {commit}")
        return commit
    runner.run(["git", "push", "-u", "origin", branch], cwd=repo)
    return commit


def main() -> int:
    stamp = utc_stamp()
    log_path = EXPECTED_REPO_PATH / "results" / "_wrapper_logs" / f"g4_trace_gate_wrapper_{stamp}.log"
    runner = Runner(log_path)
    failure_path = log_path.with_name(f"G4_TRACE_GATE_WRAPPER_FAILURE_{stamp}.json")
    try:
        runner.note("=" * 78)
        runner.note("LS-GRAND G4 Trace-Anchored Matched Gate v1.0")
        runner.note("=" * 78)
        repo = discover_repo(runner)
        check_clean_staging_area(runner, repo)
        profile = validate_profile()
        runner.note(f"Frozen run profile: {profile}")
        zip_path = locate_package(runner)
        package_dir = install_package_tree(runner, repo, zip_path)
        ensure_root_gitignore(repo)
        python, env = ensure_venv(runner, repo, package_dir)

        unit_log = repo / "results" / "_wrapper_logs" / f"g4_trace_gate_unit_tests_{stamp}.log"
        runner.run([
            python, "-m", "lsgrand_g4.cli", "test",
            "--package-root", package_dir,
            "--log", unit_log,
        ], cwd=package_dir, env=env)

        run_name = f"LS_GRAND_G4_Trace_Anchored_Matched_Gate_{profile}_{stamp}"
        output_root = repo / "results"
        config = package_dir / "configs" / f"{profile}.json"
        runner.run([
            python, "-m", "lsgrand_g4.cli", "run",
            "--package-root", package_dir,
            "--config", config,
            "--output-root", output_root,
            "--run-name", run_name,
            "--skip-tests",
        ], cwd=package_dir, env=env)
        run_dir = output_root / run_name

        execution_manifest = {
            "wrapper_version": "1.0.0",
            "package_filename": PACKAGE_FILENAME,
            "package_sha256": EXPECTED_SHA256,
            "package_source": str(zip_path),
            "profile": profile,
            "run_name": run_name,
            "repository": EXPECTED_REPOSITORY,
            "required_base_commit": REQUIRED_BASE_COMMIT,
            "git_head_before_commit": runner.capture(["git", "rev-parse", "HEAD"], cwd=repo),
            "unit_test_log_source": str(unit_log.relative_to(repo)),
            "completed_utc": utc_stamp(),
        }
        (run_dir / "WRAPPER_EXECUTION_MANIFEST.json").write_text(
            json.dumps(execution_manifest, indent=2, sort_keys=True) + "\n"
        )
        shutil.copy2(unit_log, run_dir / "UNIT_TEST_LOG.txt")
        shutil.copy2(log_path, run_dir / "WRAPPER_LOG_THROUGH_CAMPAIGN.txt")
        regenerate_result_manifest(run_dir)
        runner.run([
            python, package_dir / "scripts" / "validate_results.py", run_dir
        ], cwd=package_dir, env=env)
        shutil.copy2(log_path, run_dir / "WRAPPER_LOG_THROUGH_VALIDATION.txt")
        regenerate_result_manifest(run_dir)
        runner.run([
            python, package_dir / "scripts" / "validate_results.py", run_dir
        ], cwd=package_dir, env=env)

        wrapper_path = Path(__file__).resolve()
        stage_review_artifacts(runner, repo, package_dir, run_dir, wrapper_path)
        commit = commit_and_push(runner, repo, run_dir)

        verdict = json.loads((run_dir / "FINAL_G4_VERDICT.json").read_text())
        runner.note("=" * 78)
        runner.note(f"RUN COMPLETE: {run_dir}")
        runner.note(f"AUTOMATED G4 CLASSIFICATION: {verdict.get('verdict')}")
        runner.note(f"GIT COMMIT: {commit}")
        runner.note("The VS Code terminal remains open. Return the commit hash or run directory for scientific adjudication.")
        return 0
    except KeyboardInterrupt:
        payload = {"status": "INTERRUPTED", "utc": utc_stamp(), "traceback": traceback.format_exc()}
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(payload, indent=2) + "\n")
        runner.note(f"Interrupted. Failure record: {failure_path}")
        runner.note("The VS Code terminal remains open.")
        return 130
    except Exception as exc:
        payload = {
            "status": "FAILED",
            "utc": utc_stamp(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(payload, indent=2) + "\n")
        runner.note("ERROR: " + str(exc))
        runner.note(traceback.format_exc())
        runner.note(f"Failure record: {failure_path}")
        runner.note("No terminal-close command was issued; the VS Code terminal remains open.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
