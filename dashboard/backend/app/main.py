"""QuantSilico Generals research dashboard API.

Binds to 127.0.0.1 only. Jobs and paths are allowlisted.
Runs from .venv-training. CLI remains independent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_ROOTS = {
    REPO_ROOT / "replays" / "private",
    REPO_ROOT / "experiments",
    REPO_ROOT / "models",
    REPO_ROOT / "submission" / "packages",
    REPO_ROOT / "var" / "dashboard",
}
ALLOWED_CANDIDATES = {
    "pass",
    "pass_bot",
    "legal_random",
    "heuristic_v0",
    "heuristic_v1",
    "heuristic_v2_qualifier",
    "heuristic_aggressive",
    "heuristic_defensive",
    "heuristic_castle",
    "heuristic_deathtouch",
    "expander",
}
ALLOWED_JOBS = {"MATCH", "SUBMISSION_VALIDATE"}

app = FastAPI(title="QuantSilico Generals Research Console", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8765",
        "http://localhost:8765",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class MatchRequest(BaseModel):
    job_type: Literal["MATCH"] = "MATCH"
    candidate: str
    opponent: str
    seed: int = 0
    max_turns: int | None = Field(default=50, ge=1, le=1200)
    record_replay: bool = True


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(cwd or REPO_ROOT), text=True
    ).strip()


def _assert_allowlisted_path(path: Path) -> Path:
    resolved = path.resolve()
    for root in ALLOWED_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise HTTPException(400, "path not allowlisted")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "bind": "127.0.0.1",
        "repo": str(REPO_ROOT),
        "env": "training",
        "schema_version": 1,
    }


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "engine_commit": _git(
            "rev-parse", "HEAD", cwd=REPO_ROOT / "third_party" / "generals-bots"
        ),
        "champion": "heuristic_v1",
        "champion_status": "PACKAGED",
        "package": "submission/packages/heuristic_v1_packaged.zip",
        "active_jobs": [],
        "latest_experiment": None,
    }


@app.get("/api/repository")
def repository() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "remote": "origin",
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "recent_commits": _git("log", "-5", "--oneline").splitlines(),
        "engine_commit": _git(
            "rev-parse", "HEAD", cwd=REPO_ROOT / "third_party" / "generals-bots"
        ),
        "privacy": "private",
    }


@app.get("/api/experiments")
def experiments() -> dict[str, Any]:
    root = REPO_ROOT / "experiments" / "manifests"
    items = []
    if root.exists():
        for path in sorted(root.glob("*.json")):
            items.append(json.loads(path.read_text(encoding="utf-8")))
    return {"schema_version": 1, "experiments": items}


@app.get("/api/models")
def models() -> dict[str, Any]:
    root = REPO_ROOT / "models" / "registry"
    items = []
    if root.exists():
        for path in sorted(root.glob("*.json")):
            items.append(json.loads(path.read_text(encoding="utf-8")))
    return {
        "schema_version": 1,
        "models": items,
        "champion": "heuristic_v1",
        "challengers": [],
    }


@app.get("/api/submission")
def submission() -> dict[str, Any]:
    pkg = REPO_ROOT / "submission" / "packages" / "heuristic_v1_packaged.zip"
    report = REPO_ROOT / "submission" / "packages" / "heuristic_v1_packaged.report.json"
    parity = REPO_ROOT / "experiments" / "manifests" / "linux_parity_report.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "candidate": "heuristic_v1",
        "status": "PACKAGED",
        "upload_ready": False,
        "package_exists": pkg.exists(),
        "package_path": str(pkg) if pkg.exists() else None,
        "manual_upload_instructions": "submission/MANUAL_UPLOAD.md",
        "notes": ["No upload button; UPLOAD_READY requires Linux parity"],
    }
    if report.exists():
        rep = json.loads(report.read_text(encoding="utf-8"))
        payload["report"] = rep
        payload["status"] = rep.get("status", payload["status"])
        payload["upload_ready"] = bool(rep.get("upload_ready"))
        payload["package_hash"] = rep.get("sha256")
        payload["zip_size"] = rep.get("zip_size")
        payload["unpacked_size"] = rep.get("unpacked_size")
        payload["file_count"] = rep.get("file_count")
        payload["windows_validation"] = rep.get("windows_validation")
        payload["linux_parity"] = rep.get("linux_parity")
    if parity.exists():
        payload["linux_parity_report"] = json.loads(parity.read_text(encoding="utf-8"))
    return payload


@app.get("/api/replays")
def list_replays() -> dict[str, Any]:
    root = REPO_ROOT / "replays" / "private"
    items = []
    if root.exists():
        for path in sorted(root.glob("*.json")):
            items.append({"id": path.stem, "name": path.name})
    return {"schema_version": 1, "replays": items}


@app.get("/api/replays/{replay_id}")
def get_replay(replay_id: str) -> dict[str, Any]:
    if "/" in replay_id or "\\" in replay_id or ".." in replay_id:
        raise HTTPException(400, "invalid replay id")
    path = _assert_allowlisted_path(
        REPO_ROOT / "replays" / "private" / f"{replay_id}.json"
    )
    if not path.exists():
        raise HTTPException(404, "replay not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = data.get("schema_version", 1)
    data["privileged_label"] = "TRAINING / DEBUG PRIVILEGED VIEW — NOT AVAILABLE TO THE POLICY"
    return data


@app.get("/api/jobs/allowlist")
def job_allowlist() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "jobs": sorted(ALLOWED_JOBS),
        "candidates": sorted(ALLOWED_CANDIDATES),
        "cli_map": {
            "MATCH": "python -m generals_bot.cli.main match ...",
            "SUBMISSION_VALIDATE": "python -m generals_bot.cli.main submission validate ...",
        },
    }


@app.post("/api/jobs/match")
def run_match(req: MatchRequest) -> dict[str, Any]:
    if req.job_type not in ALLOWED_JOBS:
        raise HTTPException(400, "job type not allowlisted")
    if req.candidate not in ALLOWED_CANDIDATES or req.opponent not in ALLOWED_CANDIDATES:
        raise HTTPException(400, "candidate/opponent not allowlisted")
    # Prefer competition .venv python for matches when available.
    competition_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    python = str(competition_py if competition_py.exists() else sys.executable)
    cmd = [
        python,
        "-m",
        "generals_bot.cli.main",
        "match",
        "--candidate",
        req.candidate,
        "--opponent",
        req.opponent,
        "--seed",
        str(req.seed),
    ]
    if req.max_turns is not None:
        cmd.extend(["--max-turns", str(req.max_turns)])
    if req.record_replay:
        cmd.append("--record-replay")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise HTTPException(
            500,
            detail={"stderr": result.stderr[-2000:], "stdout": result.stdout[-2000:]},
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"invalid match output: {exc}") from exc
    payload["schema_version"] = 1
    payload["cli_command"] = " ".join(cmd)
    return payload


# Empty-state pages — real data only, no synthetic metrics.
@app.get("/api/training")
def training() -> dict[str, Any]:
    return {"schema_version": 1, "campaigns": [], "active": None}


@app.get("/api/population")
def population() -> dict[str, Any]:
    return {"schema_version": 1, "population": [], "payoff_matrix": None}


@app.get("/api/explainability")
def explainability() -> dict[str, Any]:
    return {"schema_version": 1, "explanations": []}


@app.get("/api/qualification")
def qualification() -> dict[str, Any]:
    """Qualification view: portal observations + local W/D/L suites (real traces only)."""
    portal = REPO_ROOT / "experiments" / "manifests" / "official_portal_results_2026-08-03.json"
    suites_dir = REPO_ROOT / "experiments" / "manifests"
    suite_files = sorted(suites_dir.glob("qualification_*.json"))
    phase9q = REPO_ROOT / "experiments" / "manifests" / "phase_9q_milestone.json"
    reward = REPO_ROOT / "experiments" / "manifests" / "ppo_reward_audit.json"
    draw_diag = REPO_ROOT / "experiments" / "manifests" / "expander_draw_diagnostics_9q.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "QUALIFICATION_DASHBOARD",
        "phase": "9Q",
        "champion_until_promoted": "heuristic_v1",
        "candidate_lineage": "heuristic_v2_qualifier",
        "milestones": ["turn_800_deathtouch", "turn_1050_draw_avoidance", "turn_1150", "turn_1200"],
        "portal": None,
        "phase_9q": None,
        "local_suites": [],
        "development_groups": {},
        "draw_diagnostics": None,
        "reward_audit": None,
        "note": "Wire real traces only; score_rate alone is insufficient.",
    }
    if portal.exists():
        payload["portal"] = json.loads(portal.read_text(encoding="utf-8"))
    if phase9q.exists():
        payload["phase_9q"] = json.loads(phase9q.read_text(encoding="utf-8"))
    if draw_diag.exists():
        payload["draw_diagnostics"] = json.loads(draw_diag.read_text(encoding="utf-8"))
    for path in suite_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        item = {"id": path.stem, "path": str(path.relative_to(REPO_ROOT)), "data": data}
        payload["local_suites"].append(item)
        if path.stem.startswith("qualification_development_") or "development_" in path.stem:
            payload["development_groups"][path.stem] = data.get("policies", {})
    if reward.exists():
        payload["reward_audit"] = json.loads(reward.read_text(encoding="utf-8"))
    return payload


@app.get("/api/competition")
def competition() -> dict[str, Any]:
    return {"schema_version": 1, "submissions": [], "note": "manual records only"}


STATIC_DIR = Path(__file__).resolve().parent / "static"
FRONTEND_DIST = REPO_ROOT / "dashboard" / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
elif STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "dashboard.backend.app.main:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )


if __name__ == "__main__":
    main()
