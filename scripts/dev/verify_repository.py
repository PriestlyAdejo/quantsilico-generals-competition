#!/usr/bin/env python3
"""Validate repository structure and safety constraints for bootstrap."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_PATHS = [
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".python-version",
    ".env.example",
    "LICENSE",
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "requirements-dev.txt",
    "Makefile",
    "src/generals_bot/__init__.py",
    "scripts/dev/bootstrap.ps1",
    "scripts/dev/bootstrap.sh",
    "scripts/dev/verify_environment.py",
    "scripts/dev/verify_repository.py",
    "third_party/README.md",
    "third_party/generals-bots",
    "docs/architecture/0001-repository-boundary.md",
    "docs/architecture/0002-official-engine-pinning.md",
    "docs/architecture/0003-hgb-psro-research-direction.md",
    "docs/research/trading-game-analogy.md",
    "docs/private/README.md",
    "experiments/reproducibility.yml",
    "tests/unit/test_package_import.py",
]

PRIVATE_EVIDENCE_DIRS = [
    "docs/private",
    "experiments/raw",
    "experiments/checkpoints",
    "replays/private",
]

FORBIDDEN_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".pem", ".key")
FORBIDDEN_NAMES = {".env"}


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def report(ok: bool, message: str) -> None:
    prefix = "OK  " if ok else "FAIL"
    print(f"[{prefix}] {message}")


def check_required_paths() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_PATHS:
        path = REPO_ROOT / rel
        exists = path.exists()
        report(exists, f"required path: {rel}")
        if not exists:
            failures.append(f"missing required path: {rel}")
    return failures


def check_submodule() -> list[str]:
    failures: list[str] = []
    status = run_git("submodule", "status")
    report(status.returncode == 0, "git submodule status")
    if status.returncode != 0:
        failures.append(status.stderr.strip() or "git submodule status failed")
        return failures

    line = status.stdout.strip().splitlines()[0] if status.stdout.strip() else ""
    print(f"         submodule: {line or '(empty)'}")
    if "third_party/generals-bots" not in line:
        failures.append("generals-bots submodule not registered")
        report(False, "submodule presence")
    else:
        report(True, "submodule presence")

    engine_readme = REPO_ROOT / "third_party" / "generals-bots" / "README.md"
    if not engine_readme.exists():
        failures.append("submodule working tree appears empty")
        report(False, "submodule checkout")
    else:
        report(True, "submodule checkout")
    return failures


def check_working_tree(require_clean: bool) -> list[str]:
    failures: list[str] = []
    status = run_git("status", "--short")
    report(status.returncode == 0, "git status --short")
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    if dirty:
        print("         working tree is dirty (expected during bootstrap before commit):")
        for line in dirty[:40]:
            print(f"           {line}")
        if len(dirty) > 40:
            print(f"           ... {len(dirty) - 40} more")
        if require_clean:
            failures.append("working tree is not clean (--require-clean)")
            report(False, "require-clean")
        else:
            report(True, "working-tree state reported (clean not required)")
    else:
        report(True, "working tree clean")
    return failures


def check_private_evidence_dirs() -> list[str]:
    failures: list[str] = []
    for rel in PRIVATE_EVIDENCE_DIRS:
        path = REPO_ROOT / rel
        ok = path.is_dir()
        report(ok, f"private evidence dir: {rel}")
        if not ok:
            failures.append(f"missing private evidence dir: {rel}")
    return failures


def check_forbidden_committed_files() -> list[str]:
    failures: list[str] = []
    listed = run_git("ls-files")
    if listed.returncode != 0:
        failures.append("git ls-files failed")
        report(False, "forbidden committed secret files")
        return failures

    bad: list[str] = []
    for rel in listed.stdout.splitlines():
        name = Path(rel).name
        if name in FORBIDDEN_NAMES or name.endswith(FORBIDDEN_SUFFIXES):
            # Allow .env.example explicitly
            if name == ".env.example":
                continue
            bad.append(rel)
    if bad:
        for item in bad:
            print(f"         forbidden tracked file: {item}")
        failures.append("forbidden secret/weight files are tracked")
        report(False, "forbidden committed secret files")
    else:
        report(True, "forbidden committed secret files")
    return failures


def _file_uses_lf(path: Path) -> bool:
    data = path.read_bytes()
    if b"\r\n" in data:
        return False
    return True


def check_shell_scripts() -> list[str]:
    """Check LF endings and Git executable bit for bootstrap.sh and */run.sh|build.sh."""
    failures: list[str] = []
    patterns = ("bootstrap.sh", "run.sh", "build.sh")
    candidates = [
        p
        for p in REPO_ROOT.rglob("*")
        if p.is_file()
        and p.name in patterns
        and "third_party" not in p.parts
        and ".venv" not in p.parts
    ]

    if not candidates:
        report(False, "shell scripts present")
        failures.append("expected at least scripts/dev/bootstrap.sh")
        return failures

    staged = run_git("ls-files", "--stage")
    stage_map: dict[str, str] = {}
    if staged.returncode == 0:
        for line in staged.stdout.splitlines():
            # mode sha stage\tpath
            try:
                meta, path = line.split("\t", 1)
                mode = meta.split()[0]
                stage_map[path.replace("\\", "/")] = mode
            except ValueError:
                continue

    for path in sorted(candidates):
        rel = path.relative_to(REPO_ROOT).as_posix()
        lf_ok = _file_uses_lf(path)
        report(lf_ok, f"LF endings: {rel}")
        if not lf_ok:
            failures.append(f"CRLF line endings in {rel}")

        mode = stage_map.get(rel)
        if mode is None:
            # Not yet staged/tracked: report informational, do not fail by default
            report(True, f"executable bit not yet in index (untracked ok during bootstrap): {rel}")
            continue
        exec_ok = mode == "100755"
        report(exec_ok, f"git executable bit (100755): {rel} [{mode}]")
        if not exec_ok:
            failures.append(f"missing executable bit for {rel} (mode={mode})")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="Fail if the working tree has tracked or untracked changes.",
    )
    args = parser.parse_args()

    print(f"Repository root: {REPO_ROOT}")
    failures: list[str] = []
    failures.extend(check_required_paths())
    failures.extend(check_submodule())
    failures.extend(check_working_tree(require_clean=args.require_clean))
    failures.extend(check_private_evidence_dirs())
    failures.extend(check_forbidden_committed_files())
    failures.extend(check_shell_scripts())

    print()
    if failures:
        print("Repository verification FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("Repository verification PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
