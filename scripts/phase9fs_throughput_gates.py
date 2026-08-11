"""RESOURCE_ISOLATION_GATE + Tier-1 throughput labelling (Candidate C path)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _active_processes() -> list[dict]:
    try:
        import psutil  # type: ignore

        out = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent"]):
            name = (p.info.get("name") or "").lower()
            if any(k in name for k in ("python", "node", "uvicorn", "dashboard")):
                out.append(
                    {
                        "pid": p.info.get("pid"),
                        "name": p.info.get("name"),
                        "cpu_percent": p.info.get("cpu_percent"),
                    }
                )
        return out[:50]
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    # Prefer package gate PASS before Tier-1 pursuit
    pkg = {}
    pkg_path = REPO / "experiments" / "manifests" / "phase9fs_first_submission_package_gate.json"
    if pkg_path.exists():
        pkg = json.loads(pkg_path.read_text(encoding="utf-8"))

    procs = _active_processes()
    isolation = {
        "schema_version": 1,
        "kind": "RESOURCE_ISOLATION_GATE",
        "created_at": now,
        "dashboard_polling_stopped": "OPERATOR_CONFIRM_REQUIRED",
        "replay_collectors_stopped": "OPERATOR_CONFIRM_REQUIRED",
        "no_concurrent_trainer": True,
        "no_concurrent_evaluator": True,
        "active_processes_sample": procs,
        "gate_status": "PASS_WITH_OPERATOR_CONFIRMATION"
        if pkg.get("gate_status") == "PASS"
        else "BLOCKED_UNTIL_FIRST_PACKAGE",
        "note": "Stop dashboard polling and replay collectors before latency/SPS benches.",
    }

    # Historical Phase 9F collection speed reference (not valid PPO learning TPS)
    overnight = REPO / "experiments" / "phase9f_overnight_ppo" / "rl_control" / "overnight_report.json"
    baseline_collection_tps = None
    if overnight.exists():
        od = json.loads(overnight.read_text(encoding="utf-8"))
        # Prefer explicit fields if present; else leave null
        for key in (
            "collection_transitions_per_second",
            "transitions_per_second",
            "sps",
        ):
            if key in od and od[key] is not None:
                baseline_collection_tps = float(od[key])
                break
        if baseline_collection_tps is None and "history" in od and od["history"]:
            # Cannot invent; leave null and label clearly
            baseline_collection_tps = None

    tier1 = {
        "schema_version": 1,
        "kind": "TIER1_THROUGHPUT_FORMULA",
        "created_at": now,
        "named_metrics": [
            "environment_SPS",
            "policy_decisions_per_second",
            "two_player_agent_transitions_per_second",
            "completed_games_per_second",
            "valid_ppo_learning_transitions_per_second",
        ],
        "requirements": {
            "valid_ppo_learning_transitions_per_second_min": 100,
            "valid_ppo_learning_transitions_per_second_vs_baseline_multiplier": 25,
        },
        "phase9f_baseline_label": "phase9f_collection_transitions_per_second",
        "phase9f_baseline_value": baseline_collection_tps,
        "phase9f_baseline_note": (
            "Historical overnight PPO updates were TRAINER_SEMANTICS_INVALID; "
            "do not label invalid samples as valid PPO learning transitions. "
            "Use collection TPS as historical speed reference when valid learning TPS is unavailable."
        ),
        "candidate_c_smoke_ceilings": {
            "updates": 16,
            "transitions": 4096,
            "wall_minutes": 30,
        },
        "status": "FORMULA_RECORDED_SMOKE_PENDING",
        "first_package_gate": pkg.get("gate_status"),
    }

    # Minimal Candidate C smoke placeholder — only if one-update PASS exists
    one_path = REPO / "experiments" / "manifests" / "phase9fs_one_update_correctness.json"
    c_status = "BLOCKED_UNTIL_ONE_UPDATE"
    if one_path.exists():
        one = json.loads(one_path.read_text(encoding="utf-8"))
        if one.get("gate_status") == "PASS":
            c_status = "READY_FOR_BOUNDED_SMOKE"
        else:
            c_status = "BLOCKED_ONE_UPDATE_FAIL"

    cand_c = {
        "schema_version": 1,
        "kind": "CANDIDATE_C_SMOKE_STATUS",
        "created_at": now,
        "status": c_status,
        "compare_to_frozen_v001": "AFTER_SMOKE_AND_UPLOAD_FREEZE",
        "optional_v002": "ONLY_IF_STRONGER_THAN_FROZEN_V001",
    }

    (REPO / "experiments" / "manifests" / "phase9fs_resource_isolation_gate.json").write_text(
        json.dumps(isolation, indent=2) + "\n", encoding="utf-8"
    )
    (REPO / "experiments" / "manifests" / "phase9fs_tier1_throughput.json").write_text(
        json.dumps(tier1, indent=2) + "\n", encoding="utf-8"
    )
    (REPO / "experiments" / "manifests" / "phase9fs_candidate_c_status.json").write_text(
        json.dumps(cand_c, indent=2) + "\n", encoding="utf-8"
    )
    (REPO / "experiments" / "reports" / "phase9fs_throughput.md").write_text(
        "\n".join(
            [
                "# Tier-1 throughput + Candidate C",
                "",
                f"Created: {now}",
                "",
                f"- Resource isolation: **{isolation['gate_status']}**",
                f"- Tier-1 formula recorded; baseline collection TPS: `{baseline_collection_tps}`",
                f"- Candidate C: **{c_status}**",
                "",
                "Do not call invalid overnight samples valid PPO learning transitions.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "isolation": isolation["gate_status"],
                "candidate_c": c_status,
                "baseline_collection_tps": baseline_collection_tps,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
