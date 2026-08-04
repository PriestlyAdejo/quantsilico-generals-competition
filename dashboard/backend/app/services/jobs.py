"""Dashboard-facing JobService with shared lifecycle vocabulary."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from dashboard.backend.app.paths import JOBS_DIR, REPO_ROOT, rel

JobState = Literal["QUEUED", "RUNNING", "PAUSED", "COMPLETED", "FAILED", "CANCELLED"]


@dataclass
class JobRecord:
    schema_version: int = 1
    job_id: str = ""
    job_type: str = ""
    state: JobState = "QUEUED"
    created_at: str = ""
    updated_at: str = ""
    candidate: str | None = None
    opponent: str | None = None
    seed: int | None = None
    max_turns: int | None = None
    record_replay: bool = False
    match_record: dict[str, Any] | None = None
    replay_id: str | None = None
    replay_status: str | None = None  # RECORDED | REPLAY_NOT_RECORDED
    error: str | None = None
    argv: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobService(Protocol):
    def create_match_job(
        self,
        *,
        candidate: str,
        opponent: str,
        seed: int,
        max_turns: int | None,
        record_replay: bool,
        python_exe: str,
    ) -> JobRecord: ...

    def get_job(self, job_id: str) -> JobRecord | None: ...

    def list_jobs(self) -> list[JobRecord]: ...

    def cancel_job(self, job_id: str) -> JobRecord: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_candidate_allowlist() -> set[str]:
    """Prefer the live policy registry; always include verified submitted ID."""
    from generals_bot.candidate_identity import (
        EXECUTABLE_REGISTRY_ID,
        canonicalize_candidate_id,
    )

    base = {
        "pass",
        "pass_bot",
        "legal_random",
        "expander",
        "official_expander",
        "hunter",
        "official_hunter",
        "heuristic_v0",
        "heuristic_v1",
        "heuristic_v2_qualifier",
        EXECUTABLE_REGISTRY_ID,
        "heuristic_aggressive",
        "heuristic_defensive",
        "heuristic_castle",
        "heuristic_deathtouch",
    }
    try:
        from generals_bot.selector import list_policies

        base.update(list_policies())
    except Exception:
        pass
    # Drop dashboard typo alias — not a distinct executable policy.
    base.discard("heuristic_v2f_plus_planner_terminal_form")
    base.add(EXECUTABLE_REGISTRY_ID)
    # Normalise any accidental alias insertions.
    return {canonicalize_candidate_id(x) for x in base}

class FilesystemJobService:
    """Local filesystem job store under var/dashboard/jobs/."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or JOBS_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, job_id: str) -> Path:
        if "/" in job_id or "\\" in job_id or ".." in job_id:
            raise ValueError("invalid job id")
        return self.root / f"{job_id}.json"

    def _write(self, job: JobRecord) -> None:
        job.updated_at = _now()
        self._path(job.job_id).write_text(json.dumps(job.to_dict(), indent=2) + "\n", encoding="utf-8")

    def get_job(self, job_id: str) -> JobRecord | None:
        path = self._path(job_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return JobRecord(**{k: v for k, v in data.items() if k in JobRecord.__dataclass_fields__})

    def list_jobs(self) -> list[JobRecord]:
        jobs: list[JobRecord] = []
        for path in sorted(self.root.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            jobs.append(JobRecord(**{k: v for k, v in data.items() if k in JobRecord.__dataclass_fields__}))
        return jobs

    def cancel_job(self, job_id: str) -> JobRecord:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.state in {"COMPLETED", "FAILED", "CANCELLED"}:
            job.notes.append("cancel ignored: job already terminal")
            self._write(job)
            return job
        job.state = "CANCELLED"
        job.notes.append("Cancelled via dashboard JobService (best-effort; evaluator may already have finished).")
        self._write(job)
        return job

    def create_match_job(
        self,
        *,
        candidate: str,
        opponent: str,
        seed: int,
        max_turns: int | None,
        record_replay: bool,
        python_exe: str,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        cmd = [
            python_exe,
            "-m",
            "generals_bot.cli.main",
            "match",
            "--candidate",
            candidate,
            "--opponent",
            opponent,
            "--seed",
            str(seed),
        ]
        if max_turns is not None:
            cmd.extend(["--max-turns", str(max_turns)])
        if record_replay:
            cmd.append("--record-replay")

        job = JobRecord(
            job_id=job_id,
            job_type="MATCH",
            state="QUEUED",
            created_at=_now(),
            updated_at=_now(),
            candidate=candidate,
            opponent=opponent,
            seed=seed,
            max_turns=max_turns,
            record_replay=record_replay,
            argv=cmd,
        )
        self._write(job)

        job.state = "RUNNING"
        self._write(job)

        result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
        if result.returncode != 0:
            job.state = "FAILED"
            job.error = (result.stderr or result.stdout or "match failed")[-2000:]
            self._write(job)
            return job

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            job.state = "FAILED"
            job.error = f"invalid match output: {exc}"
            self._write(job)
            return job

        payload["schema_version"] = payload.get("schema_version", 1)
        job.match_record = payload
        replay_id = payload.get("replay_id") or payload.get("replay") or None
        if isinstance(replay_id, dict):
            replay_id = replay_id.get("id")
        if replay_id:
            job.replay_id = str(replay_id)
            job.replay_status = "RECORDED"
        else:
            job.replay_id = None
            job.replay_status = "REPLAY_NOT_RECORDED"
            job.notes.append("MATCH COMPLETE; REPLAY NOT RECORDED")
        job.state = "COMPLETED"
        self._write(job)
        return job


def default_python() -> str:
    competition_py = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if competition_py.exists():
        return str(competition_py)
    return sys.executable


_service: FilesystemJobService | None = None


def get_job_service() -> FilesystemJobService:
    global _service
    if _service is None:
        _service = FilesystemJobService()
    return _service
