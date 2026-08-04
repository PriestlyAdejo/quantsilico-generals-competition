"""Repository path helpers and containment checks."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

REPO_ROOT = Path(__file__).resolve().parents[3]

ALLOWED_ROOTS = {
    REPO_ROOT / "replays" / "private",
    REPO_ROOT / "experiments",
    REPO_ROOT / "models",
    REPO_ROOT / "submission" / "packages",
    REPO_ROOT / "var" / "dashboard",
}

JOBS_DIR = REPO_ROOT / "var" / "dashboard" / "jobs"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix().replace("\\", "/")


def assert_allowlisted_path(path: Path) -> Path:
    resolved = path.resolve()
    for root in ALLOWED_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise HTTPException(400, "path not allowlisted")


def safe_replay_id(replay_id: str) -> str:
    if "/" in replay_id or "\\" in replay_id or ".." in replay_id:
        raise HTTPException(400, "invalid replay id")
    return replay_id
