"""Read-only repository status DTO — no skipped generics, no Git mutation."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dashboard.backend.app.paths import REPO_ROOT
from dashboard.backend.app.readers.evidence import manifest, submitted_package_dto


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(cwd or REPO_ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _status(value: str) -> str:
    allowed = {
        "PASS",
        "FAIL",
        "NOT_RUN",
        "NOT_RECORDED",
        "NOT_CONFIGURED",
        "NOT_APPLICABLE",
        "UNKNOWN",
    }
    return value if value in allowed else "UNKNOWN"


def repository_status_dto() -> dict[str, Any]:
    engine = REPO_ROOT / "third_party" / "generals-bots"
    branch = _git("branch", "--show-current") or "NOT_RECORDED"
    commit = _git("rev-parse", "HEAD") or "NOT_RECORDED"
    dirty = bool(_git("status", "--porcelain"))
    remote_url = _git("remote", "get-url", "origin")
    upstream = _git("rev-parse", "--abbrev-ref", "@{upstream}")
    ahead_behind = _git("rev-list", "--left-right", "--count", "@{upstream}...HEAD")
    ahead = behind = None
    if ahead_behind and "\t" in ahead_behind:
        # format: behind\tahead for left-right count of upstream...HEAD
        parts = ahead_behind.split()
        if len(parts) == 2:
            behind, ahead = int(parts[0]), int(parts[1])

    engine_commit = _git("rev-parse", "HEAD", cwd=engine)
    engine_dirty = bool(_git("status", "--porcelain", cwd=engine))

    pkg = submitted_package_dto()
    linux = manifest("linux_parity_report_preppo.json") or manifest("linux_parity_report.json")
    if linux is None:
        linux_status = _status("NOT_RUN")
    elif linux.get("passed") is True:
        linux_status = _status("PASS")
    elif linux.get("passed") is False:
        linux_status = _status("FAIL")
    else:
        linux_status = _status("NOT_RECORDED")

    windows = pkg.get("windows_validation")
    if windows is True or (isinstance(windows, dict) and windows.get("passed") is True):
        windows_status = _status("PASS")
    elif windows is False or (isinstance(windows, dict) and windows.get("passed") is False):
        windows_status = _status("FAIL")
    elif windows is None:
        windows_status = _status("NOT_RECORDED")
    else:
        windows_status = _status("UNKNOWN")

    fe_info_path = REPO_ROOT / "dashboard" / "frontend" / "dist" / "build-info.json"
    frontend_build = None
    if fe_info_path.is_file():
        try:
            frontend_build = json.loads(fe_info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            frontend_build = None

    log = _git("log", "-8", "--pretty=format:%H%x09%cI%x09%an%x09%s")
    commits: list[dict[str, Any]] = []
    if log:
        for line in log.splitlines():
            parts = line.split("\t", 3)
            if len(parts) == 4:
                commits.append(
                    {
                        "sha": parts[0],
                        "committed_at": parts[1],
                        "author": parts[2],
                        "message": parts[3],
                        "availability": "RECORDED",
                    }
                )

    locks = []
    for name, path in [
        ("pyproject.toml", REPO_ROOT / "pyproject.toml"),
        ("package.json", REPO_ROOT / "dashboard" / "frontend" / "package.json"),
        ("pnpm-lock.yaml", REPO_ROOT / "dashboard" / "frontend" / "pnpm-lock.yaml"),
        ("competition/requirements.txt", engine / "competition" / "requirements.txt"),
    ]:
        if path.is_file():
            locks.append(
                {
                    "name": name,
                    "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "sha256": _file_sha256(path),
                    "locked_by": "repository",
                    "reason": "Immutable dependency / project lock file",
                    "availability": "RECORDED",
                }
            )

    gh_workflows = REPO_ROOT / ".github" / "workflows"
    ci_workflows = []
    if gh_workflows.is_dir():
        for p in sorted(gh_workflows.glob("*.yml")) + sorted(gh_workflows.glob("*.yaml")):
            ci_workflows.append({"name": p.name, "path": str(p.relative_to(REPO_ROOT)).replace("\\", "/")})
    ci_status = _status("NOT_CONFIGURED" if not ci_workflows else "NOT_RECORDED")

    return {
        "schema_version": 1,
        "kind": "REPOSITORY_STATUS",
        "branch": branch,
        "commit": commit,
        "dirty": dirty,
        "remote": {"name": "origin", "url": remote_url or "NOT_RECORDED"},
        "upstream": upstream or "NOT_RECORDED",
        "ahead": ahead,
        "behind": behind,
        "engine_commit": engine_commit or "NOT_RECORDED",
        "engine_dirty": engine_dirty,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "frontend_build": frontend_build,
        "backend_commit": commit,
        "lockfiles": locks,
        "package_lifecycle": pkg.get("lifecycle") or "NOT_RECORDED",
        "windows_validation": windows_status,
        "linux_parity": linux_status,
        "linux_parity_source": "experiments/manifests/linux_parity_report_preppo.json"
        if linux
        else None,
        "dashboard_tests": _status("NOT_RUN"),
        "training_tests": _status("NOT_RUN"),
        "latest_test_timestamp": None,
        "recent_commits": commits,
        "ci_workflows": ci_workflows,
        "ci_runs": [],
        "ci_runs_status": ci_status,
        "hardware": {
            "cpu": platform.processor() or "NOT_RECORDED",
            "machine": platform.machine(),
            "gpu": "NOT_RECORDED",
            "note": "GPU telemetry is not collected by this read-only endpoint.",
        },
        "mutations": {"enabled": False, "reason": "The console is read-only for repository state."},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
