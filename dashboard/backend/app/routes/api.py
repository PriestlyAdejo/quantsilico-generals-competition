"""Core read/write API routes for the research console."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from generals_bot.candidate_identity import candidate_identity_record
from dashboard.backend.app.capabilities import build_capabilities
from dashboard.backend.app.gates import gate_status_dto, legacy_gate_board_from_current
from dashboard.backend.app.paths import REPO_ROOT, assert_allowlisted_path, rel, safe_replay_id
from dashboard.backend.app.readers.documentation import documentation_index, documentation_section
from dashboard.backend.app.readers.evidence import (
    GRAPH_LATENCY_WARNING,
    manifest,
    profile_snapshot_dto,
    submitted_package_dto,
)
from dashboard.backend.app.readers.qualification import qualification_dashboard_dto
from dashboard.backend.app.readers.repository_status import repository_status_dto
from dashboard.backend.app.services import env_sessions
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
    payload: dict[str, Any] = {
        "status": "ok",
        "bind": "127.0.0.1",
        "repo": ".",
        "env": "training",
        "schema_version": 1,
        "service": "QuantSilico Generals Research Console",
    }
    try:
        payload["branch"] = _git("branch", "--show-current")
        payload["commit"] = _git("rev-parse", "HEAD")
    except Exception:
        payload["branch"] = None
        payload["commit"] = None
    build = _frontend_build_info()
    if build:
        payload["frontend_build"] = build
        if payload.get("commit") and build.get("commit") and str(build["commit"]) != str(payload["commit"]):
            payload["frontend_build_mismatch"] = True
    return payload


def _frontend_build_info() -> dict[str, Any] | None:
    path = REPO_ROOT / "dashboard" / "frontend" / "dist" / "build-info.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@router.get("/api/build-info")
def build_info() -> dict[str, Any]:
    repo_branch = None
    repo_commit = None
    try:
        repo_branch = _git("branch", "--show-current")
        repo_commit = _git("rev-parse", "HEAD")
    except Exception:
        pass
    frontend = _frontend_build_info()
    mismatch = bool(
        frontend
        and repo_commit
        and frontend.get("commit")
        and str(frontend.get("commit")) != str(repo_commit)
    )
    return {
        "schema_version": 1,
        "kind": "BUILD_INFO",
        "repository": {"branch": repo_branch, "commit": repo_commit},
        "frontend": frontend,
        "mismatch": mismatch,
        "warning": (
            "dashboard/frontend/dist was built from a different commit than repository HEAD"
            if mismatch
            else None
        ),
    }


@router.get("/api/capabilities")
def capabilities() -> dict[str, Any]:
    return build_capabilities(arena_match=True)


@router.get("/api/overview")
def overview() -> dict[str, Any]:
    pkg = submitted_package_dto()
    readiness = manifest("learning_readiness_gate.json") or {}
    latency = manifest("competition_size_latency_gate.json") or {}
    development = manifest("bounded_development_ppo.json") or {}
    gates = gate_status_dto()
    current = gates["current"]
    classification = latency.get("classification") or {}
    return {
        "schema_version": 1,
        "kind": "OVERVIEW",
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "engine_commit": _git(
            "rev-parse", "HEAD", cwd=REPO_ROOT / "third_party" / "generals-bots"
        ),
        "research_phase": "Phase 7 evidence recorded; console fidelity corrective rebuild",
        "active_submitted_package": pkg,
        "heuristic_baseline": pkg.get("candidate"),
        "learned_champion": None,
        "learned_champion_note": "NO LEARNED CHAMPION",
        "learned_challenger": None,
        "gate_status": gates,
        # Convenience board for chips — ALWAYS derived from current, never upload-time.
        "gate_board": legacy_gate_board_from_current(current),
        "metrics": {
            "submitted_candidate": pkg.get("candidate"),
            "learning_readiness": current["learning_readiness"],
            "cnn_latency": classification.get("recurrent_cnn_v2", "NOT RECORDED"),
            "graph_latency": classification.get("recurrent_graph_belief_v2", "NOT RECORDED"),
            "learned_promotion": current["learned_promotion"],
            "development_campaign": development.get("kind"),
            "development_elapsed_s": development.get("elapsed_s"),
            "development_stopped_reason": development.get("stopped_reason"),
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
        "candidate_identity": candidate_identity_record(),
        "frontend_build": _frontend_build_info(),
    }


@router.get("/api/repository")
def repository() -> dict[str, Any]:
    return repository_status_dto()


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
    latency = manifest("competition_size_latency_gate.json") or {}
    classification = latency.get("classification") or {}
    cnn_lat = classification.get("recurrent_cnn_v2", "NOT RECORDED")
    graph_lat = classification.get("recurrent_graph_belief_v2", "NOT RECORDED")
    graph_deploy = (
        "ELIGIBLE_FOR_DEV"
        if graph_lat == "PASS"
        else (
            "BLOCKED_FOR_DEPLOYMENT"
            if graph_lat in {"PARTIAL", "FAIL"}
            else "NOT RECORDED"
        )
    )
    models_out = [
        {
            "id": "heuristic_v2f_plus_planner_terminal_fix",
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
            "competition_size_latency": cnn_lat,
            "notes": ["Learned control smoke only"],
        },
        {
            "id": "recurrent_graph_belief_v2_pure_torch",
            "architecture": "recurrent_graph_belief",
            "lifecycle": "SMOKE_TESTED",
            "competitive_role": "NONE",
            "delivery_status": "NOT_APPLICABLE",
            "compatibility": "CPU_COMPATIBLE" if cpu.get("all_ok") else "UNKNOWN",
            "competition_size_latency": graph_lat,
            "deployment_status": graph_deploy,
            "graph_latency_warning": GRAPH_LATENCY_WARNING,
            "notes": [
                "Principal challenger smoke",
                "Blank 8x8 ~139ms warning is not the competition-size gate",
            ],
        },
    ]
    return {
        "schema_version": 1,
        "models": models_out,
        "learned_champion": None,
        "learned_champion_note": "NO LEARNED CHAMPION",
        "heuristic_baseline": pkg.get("candidate"),
        "pyg_needed": False,
        "competition_size_latency_gate": latency or None,
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
    from generals_bot.training.telemetry_schema import annotate_history

    readiness = manifest("learning_readiness_gate.json")
    bridge = manifest("jax_pytorch_bridge_benchmark.json")
    bc = manifest("bc_tiny.json")
    equal = manifest("equal_budget_dev_comparison.json")
    cpu = manifest("official_venv_cpu_load.json")
    latency = manifest("competition_size_latency_gate.json")
    development = manifest("bounded_development_ppo.json")
    ppo = []
    charts = []
    root = REPO_ROOT / "experiments" / "manifests"
    if root.exists():
        for path in sorted(root.glob("ppo_smoke*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            telem = data.get("telemetry") or annotate_history(
                data.get("history"), producer=f"manifest:{path.stem}"
            )
            ppo.append({"id": path.stem, "path": rel(path), "data": data, "telemetry": telem})
            charts.append(
                {
                    "id": path.stem,
                    "title": f"PPO metrics — {path.stem}",
                    "producer": telem.get("producer"),
                    "schema_version": telem.get("schema_version"),
                    "series_keys": ["loss", "policy_loss", "value_loss", "entropy"],
                    "points": telem.get("points") or [],
                    "missing": telem.get("missing") or [],
                    "note": telem.get("note"),
                }
            )
    if development and isinstance(development.get("arms"), dict):
        for arm_id, arm in development["arms"].items():
            if not isinstance(arm, dict) or "history" not in arm:
                continue
            telem = arm.get("telemetry") or annotate_history(
                arm.get("history"), producer=f"development:{arm_id}"
            )
            charts.append(
                {
                    "id": f"dev_{arm_id}",
                    "title": f"DEVELOPMENT PPO — {arm_id}",
                    "producer": telem.get("producer"),
                    "schema_version": telem.get("schema_version"),
                    "series_keys": ["loss", "policy_loss", "value_loss", "entropy"],
                    "points": telem.get("points") or [],
                    "missing": telem.get("missing") or [],
                    "note": telem.get("note"),
                }
            )
    campaigns = []
    if development:
        campaigns.append(
            {
                "id": "bounded_development_ppo",
                "kind": development.get("kind"),
                "limits": development.get("limits"),
                "graph_training_allowed": development.get("graph_training_allowed"),
                "graph_deployment_status": development.get("graph_deployment_status"),
                "stopped_reason": development.get("stopped_reason"),
                "elapsed_s": development.get("elapsed_s"),
                "path": "experiments/manifests/bounded_development_ppo.json",
            }
        )
    return {
        "schema_version": 1,
        "kind": "TRAINING_SMOKE_DASHBOARD",
        "campaigns": campaigns,
        "active": None,
        "charts": charts,
        "smoke": {
            "learning_readiness": readiness,
            "bridge": {
                "decision": (bridge or {}).get("decision"),
                "path": "experiments/manifests/jax_pytorch_bridge_benchmark.json",
            },
            "bc_tiny": bc,
            "ppo": ppo,
            "equal_budget_dev_comparison": equal,
            "official_venv_cpu_load": cpu,
            "competition_size_latency_gate": latency,
            "bounded_development_ppo": development,
        },
        "labels": {
            "bc_accuracies": "smoke-training accuracy, not competitive game performance",
            "ppo_smoke": "pipeline ran with legal actions and resume; not a win-rate claim",
            "charts": "Only producer-emitted points are shown; missing fields are NOT RECORDED",
        },
        "graph_latency_warning": GRAPH_LATENCY_WARNING,
        "launch_enabled": False,
    }


@router.get("/api/population")
def population() -> dict[str, Any]:
    payoff = manifest("payoff_population_smoke.json") or manifest("payoff_pfsp_development.json")
    pfsp = manifest("pfsp_empirical.json")
    psro = manifest("psro_lightweight.json")
    if not payoff:
        return {
            "schema_version": 1,
            "population": [],
            "payoff_matrix": None,
            "state": "POPULATION DEVELOPMENT NOT YET RECORDED",
            "note": "Phase 7 PFSP work has not produced recorded population evidence yet.",
        }
    return {
        "schema_version": 1,
        "population": payoff.get("labels") or [],
        "payoff_matrix": payoff,
        "pfsp": pfsp,
        "psro": psro,
        "state": "POPULATION EVIDENCE RECORDED",
        "note": "Empirical payoff / PFSP only — not a promotion claim.",
        "games_total": payoff.get("games_total"),
        "synthetic": payoff.get("synthetic", False),
    }


@router.get("/api/explainability")
def explainability() -> dict[str, Any]:
    report = manifest("explainability_report.json")
    heuristic = manifest("heuristic_trace_smoke.json")
    if not report and not heuristic:
        return {
            "schema_version": 1,
            "explanations": [],
            "state": "NO EXPLANATION RECORD",
            "note": "Phase 8 explainability records from frozen checkpoints are not yet generated.",
        }
    return {
        "schema_version": 1,
        "explanations": [x for x in (report, heuristic) if x],
        "state": "EXPLANATION RECORD PRESENT",
        "note": "Frozen-checkpoint / heuristic smoke explanations — not a promotion claim.",
        "learned_promotion_gate": "NONE",
    }


@router.get("/api/qualification")
def qualification() -> dict[str, Any]:
    return qualification_dashboard_dto()


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
        "mode": "OFFICIAL_SESSIONS",
        "state": "OFFICIAL SESSION SERVICE REGISTERED",
        "capabilities": {
            "inspect": True,
            "reset": True,
            "step": True,
            "sessions": True,
        },
        "limits": {
            "max_concurrent": env_sessions.MAX_CONCURRENT,
            "default_ttl_s": env_sessions.DEFAULT_TTL_S,
            "max_ttl_s": env_sessions.MAX_TTL_S,
            "max_actions": env_sessions.MAX_ACTIONS,
        },
        "reason": None,
        "note": "Official sessions use the pinned GeneralsEnv. DEMO mode remains an explicit frontend opt-in.",
    }


class EnvSessionCreate(BaseModel):
    seed: int = 0
    map_preset: str = "standard"
    ttl_s: int | None = Field(default=None, ge=60, le=3600)


class EnvSessionStep(BaseModel):
    src_row: int = Field(ge=0)
    src_col: int = Field(ge=0)
    dst_row: int = Field(ge=0)
    dst_col: int = Field(ge=0)


@router.post("/api/environment/sessions")
def env_create_session(req: EnvSessionCreate) -> dict[str, Any]:
    try:
        session = env_sessions.create_session(seed=req.seed, map_preset=req.map_preset, ttl_s=req.ttl_s)
    except RuntimeError as exc:
        raise HTTPException(429, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"session create failed: {exc}") from exc
    return session.public_dict()


@router.get("/api/environment/sessions")
def env_list_sessions() -> dict[str, Any]:
    rows = env_sessions.list_sessions()
    return {
        "schema_version": 1,
        "kind": "ENV_SESSION_LIST",
        "sessions": rows,
        "active_count": len(rows),
        "max_concurrent": env_sessions.MAX_CONCURRENT,
    }


@router.get("/api/environment/sessions/{session_id}")
def env_get_session(session_id: str) -> dict[str, Any]:
    session = env_sessions.get_session(session_id)
    if session is None:
        raise HTTPException(404, "session not found or expired")
    return session.public_dict()


@router.post("/api/environment/sessions/{session_id}/reset")
def env_reset_session(session_id: str, req: EnvSessionCreate | None = None) -> dict[str, Any]:
    try:
        seed = req.seed if req is not None else None
        session = env_sessions.reset_session(session_id, seed=seed)
    except KeyError as exc:
        raise HTTPException(404, "session not found or expired") from exc
    except RuntimeError as exc:
        raise HTTPException(429, str(exc)) from exc
    return session.public_dict()


@router.post("/api/environment/sessions/{session_id}/step")
def env_step_session(session_id: str, req: EnvSessionStep) -> dict[str, Any]:
    try:
        session = env_sessions.step_session(
            session_id,
            src_row=req.src_row,
            src_col=req.src_col,
            dst_row=req.dst_row,
            dst_col=req.dst_col,
        )
    except KeyError as exc:
        raise HTTPException(404, "session not found or expired") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return session.public_dict()


@router.get("/api/environment/sessions/{session_id}/legal-actions")
def env_legal_actions(session_id: str) -> dict[str, Any]:
    try:
        return env_sessions.legal_actions(session_id)
    except KeyError as exc:
        raise HTTPException(404, "session not found or expired") from exc


@router.delete("/api/environment/sessions/{session_id}")
def env_delete_session(session_id: str) -> dict[str, Any]:
    ok = env_sessions.close_session(session_id)
    if not ok:
        raise HTTPException(404, "session not found")
    return {"schema_version": 1, "closed": True, "session_id": session_id}


@router.get("/api/champion")
def champion() -> dict[str, Any]:
    pkg = submitted_package_dto()
    precheck = manifest("learned_promotion_precheck.json") or {}
    return {
        "schema_version": 1,
        "active_submitted_package": pkg,
        "heuristic_baseline": pkg.get("candidate"),
        "local_champion": pkg.get("candidate"),
        "learned_champion": None,
        "learned_challenger": None,
        "learned_champion_note": "NO LEARNED CHAMPION",
        "learned_promotion_precheck": precheck or None,
        "promotion_checklist": {
            "LEARNED_PROMOTION_GATE": precheck.get("decision", "NONE"),
            "blocked": precheck.get("blocked", True),
            "reasons": precheck.get("reasons")
            or [
                "DEVELOPMENT only; no INITIAL/qualification campaign",
                "No learned package build authorised",
            ],
            "competitive_evaluation": "NOT_EVALUATED",
            "official_cpu_packaging": "NOT_EVALUATED",
            "portal_ready": False,
            "incomplete": True,
        },
    }


@router.get("/api/documentation")
def documentation() -> dict[str, Any]:
    return documentation_index()


@router.get("/api/documentation/{section_id}")
def documentation_by_id(section_id: str) -> dict[str, Any]:
    section = documentation_section(section_id)
    if section is None:
        raise HTTPException(404, "documentation section not found")
    return section
