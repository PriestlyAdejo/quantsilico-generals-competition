"""Minimal FastAPI research dashboard (Arena + Replay Lab).

Bind to 127.0.0.1 only. Jobs are allowlisted; no arbitrary shell/paths/Git.
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
ALLOWED_CANDIDATES = {"pass", "pass_bot", "legal_random", "heuristic_v0", "expander"}
ALLOWED_JOBS = {"MATCH"}

app = FastAPI(title="QuantSilico Generals Research Console", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8765", "http://localhost:8765", "http://127.0.0.1:5173"],
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


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "bind": "127.0.0.1", "repo": str(REPO_ROOT)}


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    import subprocess as sp

    branch = sp.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip()
    commit = sp.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty_out = sp.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
    dirty = dirty_out.strip() != ""
    engine = sp.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT / "third_party" / "generals-bots",
        text=True,
    ).strip()
    return {
        "branch": branch,
        "commit": commit,
        "dirty": dirty,
        "engine_commit": engine,
        "champion": "heuristic_v0",
        "schema_version": 1,
    }


@app.get("/api/replays")
def list_replays() -> dict[str, Any]:
    root = REPO_ROOT / "replays" / "private"
    items = []
    if root.exists():
        for path in sorted(root.glob("*.json")):
            items.append({"id": path.stem, "path": str(path), "name": path.name})
    return {"schema_version": 1, "replays": items}


@app.get("/api/replays/{replay_id}")
def get_replay(replay_id: str) -> dict[str, Any]:
    # Path allowlist: only private replay JSON by stem
    if "/" in replay_id or "\\" in replay_id or ".." in replay_id:
        raise HTTPException(400, "invalid replay id")
    path = REPO_ROOT / "replays" / "private" / f"{replay_id}.json"
    if not path.exists():
        raise HTTPException(404, "replay not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = data.get("schema_version", 1)
    data["privileged_label"] = None
    return data


@app.post("/api/jobs/match")
def run_match(req: MatchRequest) -> dict[str, Any]:
    if req.job_type not in ALLOWED_JOBS:
        raise HTTPException(400, "job type not allowlisted")
    if req.candidate not in ALLOWED_CANDIDATES or req.opponent not in ALLOWED_CANDIDATES:
        raise HTTPException(400, "candidate/opponent not allowlisted")
    cmd = [
        sys.executable,
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
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
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


STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
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
