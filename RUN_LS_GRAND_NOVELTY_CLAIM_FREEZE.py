#!/usr/bin/env python3
"""Single WSL wrapper for LS-GRAND Novelty and Claim-Freeze Gate G0 v1.0.

Drop this file directly into /home/afazeli2006/LS_GRAND and run it from the
existing VS Code WSL terminal. The script never closes the terminal.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import site
import stat
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PACKAGE_FILENAME = "LS_GRAND_Novelty_Claim_Freeze_G0_v1_0.zip"
PACKAGE_DIRNAME = "LS_GRAND_Novelty_Claim_Freeze_G0_v1_0"
EXPECTED_SHA256 = "6fb29339517487eecbd60f6ac1479a08a12d881741b3dea6fcc62ea79132b96d"
EXPECTED_REPOSITORY = "afazeliUofT/LS_GRAND"
EXPECTED_REPO_PATH = Path("/home/afazeli2006/LS_GRAND")
REQUIRED_BASE_COMMIT = "ed5a4b4be5dbabeed98691043dd64db6d0415dae"
VALID_PROFILES = {"gate", "smoke"}


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
        self.log_path.write_text("", encoding="utf-8")

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

    def capture(
        self,
        command: Iterable[str | Path],
        *,
        cwd: Path,
        check: bool = True,
    ) -> str:
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
            f"Run this wrapper from the initialized WSL repository {EXPECTED_REPO_PATH}. "
            f"Current directory: {cwd}"
        ) from exc
    if root != EXPECTED_REPO_PATH.resolve():
        raise RuntimeError(f"Expected repository {EXPECTED_REPO_PATH}, found {root}.")
    wrapper = Path(__file__).resolve()
    if wrapper.parent != root:
        raise RuntimeError(f"Drop the wrapper directly in {root}; current path is {wrapper}.")
    remote = runner.capture(["git", "remote", "get-url", "origin"], cwd=root)
    if normalize_remote(remote).lower() != EXPECTED_REPOSITORY.lower():
        raise RuntimeError(f"Unexpected origin {remote!r}; expected {EXPECTED_REPOSITORY}.")
    runner.run(["git", "cat-file", "-e", f"{REQUIRED_BASE_COMMIT}^{{commit}}"], cwd=root)
    runner.run(["git", "merge-base", "--is-ancestor", REQUIRED_BASE_COMMIT, "HEAD"], cwd=root)
    return root


def candidate_download_dirs() -> list[Path]:
    out: list[Path] = []
    override = os.environ.get("LSGRAND_G0_DOWNLOADS")
    if override:
        out.append(Path(override).expanduser())
    out.extend(
        [
            Path("/mnt/c/Users/alifa/Downloads"),
            Path("/mnt/c/Users/afazeli2006/Downloads"),
            Path.home() / "Downloads",
        ]
    )
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
            "Download the ZIP without renaming it, or set LSGRAND_G0_DOWNLOADS."
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
            "Delete stale copies and download the supplied package again."
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
    for line in manifest.read_text(encoding="utf-8").splitlines():
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
            marker.write_text(EXPECTED_SHA256 + "\n", encoding="utf-8")
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
    marker.write_text(EXPECTED_SHA256 + "\n", encoding="utf-8")
    runner.note(f"Installed verified package at {package_dir}")
    return package_dir


def ensure_root_gitignore(repo: Path) -> None:
    path = repo / ".gitignore"
    marker = "# LS-GRAND novelty/claim-freeze local execution"
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
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in current:
        if current and not current.endswith("\n"):
            current += "\n"
        path.write_text(current + block, encoding="utf-8")


def ensure_venv_and_install(
    runner: Runner,
    repo: Path,
    package_dir: Path,
) -> tuple[Path, dict[str, str]]:
    """Create/activate .venv and install the standard-library package via .pth.

    A .pth installation avoids network access and never stages the virtual
    environment. It is sufficient because the package has no external runtime
    dependencies.
    """
    venv = repo / ".venv"
    python = venv / "bin" / "python"
    if not python.is_file():
        runner.note(f"Creating virtual environment: {venv}")
        runner.run([sys.executable, "-m", "venv", venv], cwd=repo)
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = str(venv / "bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    site_dir_text = runner.capture(
        [python, "-c", "import site; print(site.getsitepackages()[0])"], cwd=repo
    )
    site_dir = Path(site_dir_text)
    site_dir.mkdir(parents=True, exist_ok=True)
    pth = site_dir / "lsgrand_novelty_g0.pth"
    pth.write_text(str(package_dir / "src") + "\n", encoding="utf-8")
    runner.note(f"Activated venv and installed package path through {pth}")
    runner.run(
        [python, "-c", "import lsgrand_novelty; print(lsgrand_novelty.__version__)"],
        cwd=repo,
        env=env,
    )
    return python, env


def check_clean_staging_area(runner: Runner, repo: Path) -> None:
    staged = runner.capture(["git", "diff", "--cached", "--name-only"], cwd=repo)
    if staged.strip():
        raise RuntimeError(
            "The repository already has staged files. Commit or unstage them before running this wrapper.\n"
            f"Pre-existing staged paths:\n{staged}"
        )


def validate_profile() -> str:
    profile = os.environ.get("LSGRAND_G0_PROFILE", "gate").strip().lower()
    if profile not in VALID_PROFILES:
        raise ValueError(f"LSGRAND_G0_PROFILE must be one of {sorted(VALID_PROFILES)}, got {profile!r}")
    return profile


def write_wrapper_manifest(
    run_dir: Path,
    *,
    profile: str,
    repo: Path,
    package_path: Path,
    git_head_before: str,
    unit_log: Path,
) -> None:
    obj = {
        "wrapper_version": "1.0.0",
        "package_filename": PACKAGE_FILENAME,
        "package_sha256": EXPECTED_SHA256,
        "package_source": str(package_path),
        "profile": profile,
        "run_name": run_dir.name,
        "repository": EXPECTED_REPOSITORY,
        "required_base_commit": REQUIRED_BASE_COMMIT,
        "git_head_before_commit": git_head_before,
        "unit_test_log_source": str(unit_log.relative_to(repo)),
        "completed_utc_before_git": utc_stamp(),
    }
    (run_dir / "WRAPPER_EXECUTION_MANIFEST.json").write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def stage_review_artifacts(
    runner: Runner,
    repo: Path,
    package_dir: Path,
    run_dir: Path,
    wrapper_path: Path,
) -> None:
    rel = package_dir.relative_to(repo)
    package_allowlist = [
        rel / ".gitignore",
        rel / "README.md",
        rel / "PACKAGE_METADATA.json",
        rel / "SHA256SUMS",
        rel / "pyproject.toml",
        rel / "requirements.txt",
        rel / "data",
        rel / "docs",
        rel / "src",
        rel / "scripts",
        rel / "tests",
        rel / "preflight_results",
    ]
    paths: list[Path] = [repo / ".gitignore", wrapper_path, run_dir]
    paths.extend(repo / p for p in package_allowlist)
    existing = [p.relative_to(repo) if p.is_absolute() else p for p in paths if p.exists()]
    runner.run(["git", "add", "--", *existing], cwd=repo)
    staged = runner.capture(["git", "diff", "--cached", "--name-only"], cwd=repo)
    bad: list[str] = []
    allowed_prefixes = {
        ".gitignore",
        wrapper_path.name,
        str(rel),
        str(run_dir.relative_to(repo)),
    }
    for line in staged.splitlines():
        if line == ".venv" or line.startswith(".venv/") or "/__pycache__/" in f"/{line}/":
            bad.append(line)
            continue
        if not any(line == p or line.startswith(p + "/") for p in allowed_prefixes):
            bad.append(line)
    if bad:
        runner.run(["git", "reset"], cwd=repo)
        raise RuntimeError(f"Refusing to commit unexpected paths: {bad}")
    if not staged.strip():
        raise RuntimeError("No review artifacts were staged.")
    runner.note("Staged review allowlist:\n" + staged)


def main() -> int:
    stamp = utc_stamp()
    fallback_log = EXPECTED_REPO_PATH / "results" / "_wrapper_logs" / f"novelty_g0_wrapper_{stamp}.log"
    runner = Runner(fallback_log)
    run_dir: Path | None = None
    repo: Path | None = None
    try:
        runner.note("=" * 78)
        runner.note("LS-GRAND Novelty and Claim-Freeze Gate G0 v1.0")
        runner.note("=" * 78)
        repo = discover_repo(runner)
        check_clean_staging_area(runner, repo)
        profile = validate_profile()
        runner.note(f"Frozen run profile: {profile}")
        package_path = locate_package(runner)
        package_dir = install_package_tree(runner, repo, package_path)
        ensure_root_gitignore(repo)
        python, env = ensure_venv_and_install(runner, repo, package_dir)
        git_head_before = runner.capture(["git", "rev-parse", "HEAD"], cwd=repo)

        unit_log = repo / "results" / "_wrapper_logs" / f"novelty_g0_unit_tests_{stamp}.log"
        runner.run(
            [python, "-m", "lsgrand_novelty.cli", "test", "--log", unit_log],
            cwd=repo,
            env=env,
        )

        run_name = f"LS_GRAND_Novelty_Claim_Freeze_G0_{profile}_{stamp}"
        run_dir = repo / "results" / run_name
        runner.run(
            [
                python,
                "-m",
                "lsgrand_novelty.cli",
                "run",
                "--profile",
                profile,
                "--output-root",
                repo / "results",
                "--run-name",
                run_name,
            ],
            cwd=repo,
            env=env,
        )
        shutil.copy2(unit_log, run_dir / "UNIT_TEST_LOG.txt")
        write_wrapper_manifest(
            run_dir,
            profile=profile,
            repo=repo,
            package_path=package_path,
            git_head_before=git_head_before,
            unit_log=unit_log,
        )
        shutil.copy2(runner.log_path, run_dir / "WRAPPER_LOG_BEFORE_VALIDATION.txt")

        validator = package_dir / "scripts" / "validate_results.py"
        runner.run([python, validator, run_dir], cwd=repo, env=env)
        shutil.copy2(runner.log_path, run_dir / "WRAPPER_LOG_THROUGH_VALIDATION.txt")
        runner.run([python, validator, run_dir], cwd=repo, env=env)

        validation = json.loads((run_dir / "VALIDATION_REPORT.json").read_text(encoding="utf-8"))
        if not validation.get("valid"):
            raise RuntimeError(f"result validation failed: {validation}")
        verdict = json.loads((run_dir / "FINAL_G0_NOVELTY_VERDICT.json").read_text(encoding="utf-8"))

        wrapper_path = Path(__file__).resolve()
        stage_review_artifacts(runner, repo, package_dir, run_dir, wrapper_path)
        runner.run(
            ["git", "commit", "-m", f"LS-GRAND novelty claim freeze: {run_name}"],
            cwd=repo,
        )
        branch = runner.capture(["git", "branch", "--show-current"], cwd=repo)
        if not branch:
            raise RuntimeError("Cannot push from a detached HEAD.")
        runner.run(["git", "push", "-u", "origin", branch], cwd=repo)
        commit = runner.capture(["git", "rev-parse", "HEAD"], cwd=repo)

        runner.note("=" * 78)
        runner.note(f"RUN COMPLETE: {run_dir}")
        runner.note(f"AUTOMATED CLASSIFICATION: {verdict['verdict']}")
        runner.note(f"GIT COMMIT: {commit}")
        runner.note("The VS Code terminal remains open. Return the commit hash or run directory for review.")
        return 0
    except Exception as exc:
        runner.note("=" * 78)
        runner.note(f"RUN FAILED: {type(exc).__name__}: {exc}")
        with runner.log_path.open("a", encoding="utf-8") as f:
            f.write("\n" + traceback.format_exc())
        if run_dir is not None and run_dir.exists():
            try:
                shutil.copy2(runner.log_path, run_dir / "WRAPPER_FAILURE_LOG.txt")
            except Exception:
                pass
        print(
            f"\nFailure log: {runner.log_path}\n"
            "The VS Code terminal remains open; correct the reported issue and rerun the same wrapper.",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
