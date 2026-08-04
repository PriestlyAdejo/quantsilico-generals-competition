"""Safe JSON evidence readers for dashboard DTOs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dashboard.backend.app.paths import REPO_ROOT, rel


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def manifest(name: str) -> dict[str, Any] | None:
    return read_json(REPO_ROOT / "experiments" / "manifests" / name)


def package_report(stem: str) -> dict[str, Any] | None:
    return read_json(REPO_ROOT / "submission" / "packages" / f"{stem}.report.json")


def submitted_package_dto() -> dict[str, Any]:
    report = package_report("heuristic_v2_preppo_8f7405fe9834161c_packaged") or {}
    obs = manifest("official_portal_observation_heuristic_v2_preppo_2026-08-04.json") or {}
    active = obs.get("active_submission") or {}
    return {
        "schema_version": 1,
        "candidate": report.get("candidate")
        or active.get("candidate_id")
        or "heuristic_v2f_plus_planner_terminal_form",
        "package_path": "submission/packages/heuristic_v2_preppo_8f7405fe9834161c_packaged.zip",
        "package_sha256": report.get("sha256")
        or active.get("package_sha256")
        or "e1237f77dee469935fc3a60811b9a34522b83dd37bf4d76fa2555e6107a8edfa",
        "config_hash": active.get("config_hash") or "8f7405fe9834161c",
        "authoritative_policy_source_commit": "027ff5d",
        "embedded_bot_commit": "ee06778",
        "embedded_metadata_status": "STALE",
        "repository_completion_commit": "26954e6",
        "windows_validation": report.get("windows_validation"),
        "linux_parity": report.get("linux_parity"),
        "lifecycle": report.get("status") or "SUBMITTED",
        "portal_verdict": "QUALIFIED",
        "portal_gate_name": "PORTAL_SUBMISSION_GATE",
        "upload_record": "submission/UPLOAD_RECORD_heuristic_v2_preppo.md",
        "manual_upload_instructions": "submission/MANUAL_UPLOAD_heuristic_v2_preppo.md",
        "metadata_note": (
            "Package policy bytes match commit 027ff5d. "
            "The embedded bot_commit contains stale metadata (ee06778) and is retained for auditability only."
        ),
        "learned_model_included": False,
        "report_path": rel(
            REPO_ROOT / "submission" / "packages" / "heuristic_v2_preppo_8f7405fe9834161c_packaged.report.json"
        )
        if (REPO_ROOT / "submission" / "packages" / "heuristic_v2_preppo_8f7405fe9834161c_packaged.report.json").exists()
        else None,
    }


def profile_snapshot_dto() -> dict[str, Any] | None:
    obs = manifest("official_portal_observation_heuristic_v2_preppo_2026-08-04.json")
    if not obs:
        return None
    snap = obs.get("public_leaderboard_snapshot") or {}
    return {
        "schema_version": 1,
        "kind": "PORTAL_PROFILE_SNAPSHOT",
        "label": "PORTAL PROFILE SNAPSHOT",
        "observed_at": obs.get("observed_at"),
        "source_reference": obs.get("public_profile_url")
        or "https://www.generals.bot/player?name=QuantSilico&id=88151",
        "provenance": obs.get("recording") or "MANUALLY_RECORDED",
        "attribution_method": (obs.get("active_submission") or {}).get(
            "attribution_method", "MANUAL_OPERATOR_ASSIGNMENT"
        ),
        "rank": snap.get("rank"),
        "of": snap.get("of"),
        "elo": snap.get("elo"),
        "record": snap.get("record"),
        "games": snap.get("games"),
        "displayed_raw_win_rate": snap.get("displayed_raw_win_rate"),
        "score_rate_draws_half": snap.get("score_rate_draws_half"),
        "live": False,
        "note": "Manually recorded snapshot — not live Elo/rank.",
    }


GRAPH_LATENCY_WARNING = (
    "~139 ms measured on blank 8×8 smoke input; competition-sized p99 not yet validated."
)
