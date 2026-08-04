"""Core read/write API routes for the research console."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dashboard.backend.app.capabilities import build_capabilities
from dashboard.backend.app.paths import REPO_ROOT, assert_allowlisted_path, rel, safe_replay_id
from dashboard.backend.app.readers.evidence import (
    GRAPH_LATENCY_WARNING,
    manifest,
    profile_snapshot_dto,
    submitted_package_dto,
)
from dashboard.backend.app.services.jobs import (
    default_python,
    get_job_service,
    resolve_candidate_allowlist,
)

router = APIRouter()

ALLOWED_JOBS = {"MATCH", "SUBMISSION_VALIDATE"}


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


@router.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "bind": "127.0.0.1",
        "repo": ".",
        "env": "training",
        "schema_version": 1,
    }


@router.get("/api/capabilities")
def capabilities() -> dict[str, Any]:
    return build_capabilities(arena_match=True)


@router.get("/api/overview")
def overview() -> dict[str, Any]:
    pkg = submitted_package_dto()
    obs = manifest("official_portal_observation_heuristic_v2_preppo_2026-08-04.json") or {}
    readiness = manifest("learning_readiness_gate.json") or {}
    return {
        "schema_version": 1,
        "kind": "OVERVIEW",
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "engine_commit": _git(
            "rev-parse", "HEAD", cwd=REPO_ROOT / "third_party" / "generals-bots"
        ),
        "research_phase": "Phase 4 console + Phase 5/6 smoke passed; Phase 7 next",
        "active_submitted_package": pkg,
        "heuristic_baseline": pkg.get("candidate"),
        "learned_champion": None,
        "learned_champion_note": "NO LEARNED CHAMPION",
        "learned_challenger": None,
        "gate_status": obs.get("gate_status_at_observation")
        or {
            "HEURISTIC_DEVELOPMENT_GATE": "FAIL",
            "PRE_PPO_SUBMISSION_GATE": "PASS",
            "PORTAL_SUBMISSION_GATE": "PASS",
            "LEARNING_READINESS_GATE": readiness.get("decision", "UNKNOWN"),
            "LEARNED_PROMOTION_GATE": "NONE",
        },
        "active_jobs": [
            j.to_dict()
            for j in get_job_service().list_jobs()
            if j.state in {"QUEUED", "RUNNING", "PAUSED"}
        ],
        "learning_smoke": {
            "readiness": readiness.get("decision"),
            "bridge": (manifest("jax_pytorch_bridge_benchmark.json") or {}).get("decision"),
            "note": "Smoke/infrastructure results only — not competitive performance.",
        },
        # Deliberately omit live Elo/rank here.
    }


@router.get("/api/repository")
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
        "mutations": {"enabled": False, "reason": "The console is read-only for repository state."},
    }


@router.get("/api/experiments")
def experiments() -> dict[str, Any]:
    root = REPO_ROOT / "experiments" / "manifests"
    items = []
    if root.exists():
        for path in sorted(root.glob("*.json")):
            if path.name.startswith("_"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            items.append(
                {
                    "id": path.stem,
                    "path": rel(path),
                    "schema_version": data.get("schema_version", 1),
                    "kind": data.get("kind") or data.get("status") or "MANIFEST",
                    "data": data,
                }
            )
    return {"schema_version": 1, "experiments": items}


@router.get("/api/experiments/{experiment_id}")
def experiment_detail(experiment_id: str) -> dict[str, Any]:
    if "/" in experiment_id or ".." in experiment_id:
        raise HTTPException(400, "invalid experiment id")
    path = REPO_ROOT / "experiments" / "manifests" / f"{experiment_id}.json"
    if not path.is_file():
        raise HTTPException(404, "experiment not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "id": experiment_id,
        "path": rel(path),
        "data": data,
    }


@router.get("/api/models")
def models() -> dict[str, Any]:
    pkg = submitted_package_dto()
    cpu = manifest("official_venv_cpu_load.json") or {}
    models_out = [
        {
            "id": "heuristic_v2f_plus_planner_terminal_form",
            "architecture": "heuristic",
            "lifecycle": "EVALUATED",
            "competitive_role": "BASELINE",
            "delivery_status": "SUBMITTED",
            "compatibility": "OFFICIAL_ENV_COMPATIBLE",
            "notes": ["Active submitted portal package"],
        },
        {
            "id": "recurrent_mlp_v1",
            "architecture": "recurrent_mlp_bridge",
            "lifecycle": "SMOKE_TESTED",
            "competitive_role": "NONE",
            "delivery_status": "NOT_APPLICABLE",
            "compatibility": "CPU_COMPATIBLE" if cpu.get("all_ok") else "UNKNOWN",
            "notes": ["Engineering bridge control; not a competitive champion"],
        },
        {
            "id": "recurrent_cnn_v2",
            "architecture": "recurrent_cnn",
            "lifecycle": "SMOKE_TESTED",
            "competitive_role": "NONE",
            "delivery_status": "NOT_APPLICABLE",
            "compatibility": "CPU_COMPATIBLE" if cpu.get("all_ok") else "UNKNOWN",
            "notes": ["Learned control smoke only"],
        },
        {
            "id": "recurrent_graph_belief_v2_pure_torch",
            "architecture": "recurrent_graph_belief",
            "lifecycle": "SMOKE_TESTED",
            "competitive_role": "NONE",
            "delivery_status": "NOT_APPLICABLE",
            "compatibility": "CPU_COMPATIBLE" if cpu.get("all_ok") else "UNKNOWN",
            "graph_latency_warning": GRAPH_LATENCY_WARNING,
            "notes": ["Principal challenger smoke; latency not competition-validated"],
        },
    ]
    return {
        "schema_version": 1,
        "models": models_out,
        "learned_champion": None,
        "learned_champion_note": "NO LEARNED CHAMPION",
        "heuristic_baseline": pkg.get("candidate"),
        "pyg_needed": False,
        "graph_latency_warning": GRAPH_LATENCY_WARNING,
    }


@router.get("/api/submission")
def submission() -> dict[str, Any]:
    pkg = submitted_package_dto()
    obs = manifest("official_portal_observation_heuristic_v2_preppo_2026-08-04.json")
    parity = manifest("linux_parity_report_preppo.json")
    return {
        "schema_version": 1,
        "kind": "SUBMISSION_DASHBOARD",
        "package": pkg,
        "active_portal_submission": (obs or {}).get("active_submission"),
        "gate_status": (obs or {}).get("gate_status_at_observation"),
        "recording": (obs or {}).get("recording"),
        "linux_parity_report": parity,
        "upload_enabled": False,
        "upload_note": "Uploads are manual by design. Credentials never enter this application.",
    }


@router.get("/api/replays")
def list_replays() -> dict[str, Any]:
    root = REPO_ROOT / "replays" / "private"
    items = []
    if root.exists():
        for path in sorted(root.glob("*.json")):
            items.append({"id": path.stem, "name": path.name, "path": rel(path)})
    return {"schema_version": 1, "replays": items}


@router.get("/api/replays/{replay_id}")
def get_replay(replay_id: str) -> dict[str, Any]:
    rid = safe_replay_id(replay_id)
    path = assert_allowlisted_path(REPO_ROOT / "replays" / "private" / f"{rid}.json")
    if not path.exists():
        raise HTTPException(404, "replay not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = data.get("schema_version", 1)
    data["id"] = rid
    data["map_key"] = data.get("map_key") or data.get("seed") or rid
    data["privileged_label"] = (
        "TRAINING / DEBUG PRIVILEGED VIEW — NOT AVAILABLE TO THE POLICY"
    )
    return data


@router.get("/api/jobs/allowlist")
def job_allowlist() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "jobs": sorted(ALLOWED_JOBS),
        "candidates": sorted(resolve_candidate_allowlist()),
        "cli_map": {
            "MATCH": "python -m generals_bot.cli.main match ...",
            "SUBMISSION_VALIDATE": "python -m generals_bot.cli.main submission validate ...",
        },
    }


@router.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "jobs": [j.to_dict() for j in get_job_service().list_jobs()],
    }


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    if "/" in job_id or ".." in job_id:
        raise HTTPException(400, "invalid job id")
    job = get_job_service().get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job.to_dict()


@router.post("/api/jobs/match")
def run_match(req: MatchRequest) -> dict[str, Any]:
    allow = resolve_candidate_allowlist()
    if req.job_type not in ALLOWED_JOBS:
        raise HTTPException(400, "job type not allowlisted")
    if req.candidate not in allow or req.opponent not in allow:
        raise HTTPException(400, "candidate/opponent not allowlisted")
    job = get_job_service().create_match_job(
        candidate=req.candidate,
        opponent=req.opponent,
        seed=req.seed,
        max_turns=req.max_turns,
        record_replay=req.record_replay,
        python_exe=default_python(),
    )
    return job.to_dict()


@router.get("/api/training")
def training() -> dict[str, Any]:
    readiness = manifest("learning_readiness_gate.json")
    bridge = manifest("jax_pytorch_bridge_benchmark.json")
    bc = manifest("bc_tiny.json")
    equal = manifest("equal_budget_dev_comparison.json")
    cpu = manifest("official_venv_cpu_load.json")
    ppo = []
    root = REPO_ROOT / "experiments" / "manifests"
    if root.exists():
        for path in sorted(root.glob("ppo_smoke*.json")):
            ppo.append({"id": path.stem, "path": rel(path), "data": json.loads(path.read_text(encoding="utf-8"))})
    return {
        "schema_version": 1,
        "kind": "TRAINING_SMOKE_DASHBOARD",
        "campaigns": [],
        "active": None,
        "smoke": {
            "learning_readiness": readiness,
            "bridge": {"decision": (bridge or {}).get("decision"), "path": "experiments/manifests/jax_pytorch_bridge_benchmark.json"},
            "bc_tiny": bc,
            "ppo": ppo,
            "equal_budget_dev_comparison": equal,
            "official_venv_cpu_load": cpu,
        },
        "labels": {
            "bc_accuracies": "smoke-training accuracy, not competitive game performance",
            "ppo_smoke": "pipeline ran with legal actions and resume; not a win-rate claim",
        },
        "graph_latency_warning": GRAPH_LATENCY_WARNING,
        "launch_enabled": False,
    }


@router.get("/api/population")
def population() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "population": [],
        "payoff_matrix": None,
        "state": "POPULATION DEVELOPMENT NOT YET RECORDED",
        "note": "Phase 7 PFSP work has not produced recorded population evidence yet.",
    }


@router.get("/api/explainability")
def explainability() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "explanations": [],
        "state": "NO EXPLANATION RECORD",
        "note": "Phase 8 explainability records from frozen checkpoints are not yet generated.",
    }


@router.get("/api/qualification")
def qualification() -> dict[str, Any]:
    portal_v2 = manifest("official_portal_observation_heuristic_v2_preppo_2026-08-04.json")
    readiness = manifest("learning_readiness_gate.json")
    preppo = manifest("phase_9q_pre_ppo_submission_gate.json")
    return {
        "schema_version": 1,
        "kind": "QUALIFICATION_DASHBOARD",
        "champion_until_promoted": "heuristic_v2f_plus_planner_terminal_form",
        "gates": {
            "HEURISTIC_DEVELOPMENT_GATE": "FAIL",
            "PRE_PPO_SUBMISSION_GATE": "PASS",
            "PORTAL_SUBMISSION_GATE": "PASS",
            "LEARNING_READINESS_GATE": (readiness or {}).get("decision", "UNKNOWN"),
            "LEARNED_PROMOTION_GATE": "NONE",
        },
        "gate_names": {
            "HEURISTIC_DEVELOPMENT_GATE": "internal research Expander discovery/conversion suite",
            "PRE_PPO_SUBMISSION_GATE": "local comparison vs previously submitted package",
            "PORTAL_SUBMISSION_GATE": "portal Expander 3-game gate (QUALIFIED ≠ final tournament)",
            "LEARNING_READINESS_GATE": "engineering readiness before PPO campaigns",
            "LEARNED_PROMOTION_GATE": "learned model may replace heuristic champion",
        },
        "portal_active_v2": portal_v2,
        "pre_ppo_submission_gate": preppo,
        "learning_readiness": readiness,
        "note": "Never use unqualified QUALIFIED without naming PORTAL_SUBMISSION_GATE.",
    }


@router.get("/api/competition")
def competition() -> dict[str, Any]:
    obs = manifest("official_portal_observation_heuristic_v2_preppo_2026-08-04.json")
    probe = manifest("portal_public_attribution_probe.json")
    return {
        "schema_version": 1,
        "kind": "COMPETITION_PORTAL_MANUAL",
        "active_submission": (obs or {}).get("active_submission"),
        "profile_snapshot": profile_snapshot_dto(),
        "gate_status": (obs or {}).get("gate_status_at_observation"),
        "attribution_probe": probe,
        "matches": [],
        "note": "manual records only; no portal mutation endpoints; snapshots are not live",
    }


@router.get("/api/environment")
def environment() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "READ_ONLY",
        "state": "OFFICIAL SESSION NOT REGISTERED",
        "capabilities": {
            "inspect": True,
            "reset": False,
            "step": False,
        },
        "reason": "No safe official-environment session service is currently registered.",
        "note": "Use Replay Lab for board inspection derived from recorded matches.",
    }


@router.get("/api/champion")
def champion() -> dict[str, Any]:
    pkg = submitted_package_dto()
    return {
        "schema_version": 1,
        "active_submitted_package": pkg,
        "heuristic_baseline": pkg.get("candidate"),
        "local_champion": pkg.get("candidate"),
        "learned_champion": None,
        "learned_challenger": None,
        "learned_champion_note": "NO LEARNED CHAMPION",
        "promotion_checklist": {
            "LEARNED_PROMOTION_GATE": "NONE",
            "competitive_evaluation": "NOT_EVALUATED",
            "official_cpu_packaging": "NOT_EVALUATED",
            "portal_ready": False,
            "incomplete": True,
        },
    }


@router.get("/api/documentation")
def documentation() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sections": [
            {"id": "startup", "title": "Startup"},
            {"id": "gates", "title": "Qualification gates"},
            {"id": "attribution", "title": "Portal attribution"},
            {"id": "manual-upload", "title": "Manual upload"},
            {"id": "training-smoke", "title": "Training smoke evidence"},
            {"id": "troubleshooting", "title": "Troubleshooting"},
        ],
    }
