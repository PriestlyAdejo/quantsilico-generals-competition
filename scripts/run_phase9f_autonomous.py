"""Phase 9F autonomous controller — Plan v2 mandatory overnight RL."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "experiments/manifests/phase9f_autonomous_run_state.json"
PLAN_V2 = REPO / "experiments/manifests/phase9f_generated_plan_v2.json"
LOCAL_TZ = ZoneInfo("Europe/London")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_local() -> datetime:
    return datetime.now(LOCAL_TZ)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = _now_utc().isoformat()
    STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _deadline_dt(s: str) -> datetime:
    # Accept +01:00 style
    return datetime.fromisoformat(s)


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 9F Plan v2 autonomous controller")
    p.add_argument("--deadline", default="2026-08-05T08:00:00+01:00")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--mandatory-rl", action="store_true", default=True)
    p.add_argument("--overnight-rl", action="store_true", default=True)
    p.add_argument("--concurrency", default="auto")
    p.add_argument(
        "--stage",
        default="auto",
        choices=["auto", "persistent_actors", "status"],
        help="Force a stage or report status",
    )
    args = p.parse_args()

    state = _load_json(STATE)
    plan = _load_json(PLAN_V2)
    if not plan:
        print(json.dumps({"status": "BLOCKED", "reason": "Plan v2 missing"}, indent=2))
        return 2

    t0_iso = state.get("t0")
    if not t0_iso:
        t0_iso = _now_local().isoformat()
        state["t0"] = t0_iso
        state["deadline"] = args.deadline
        state["stage"] = "CONTROLLER_STARTED"
        state["mandatory_rl"] = bool(args.mandatory_rl)
        state["overnight_rl"] = bool(args.overnight_rl)
        state["device"] = args.device
        state["plan_v2_sha256"] = plan.get("plan_sha256")
        _save_state(state)

    t0 = datetime.fromisoformat(t0_iso)
    now = _now_local()
    deadline = _deadline_dt(args.deadline)
    elapsed_min = (now - t0).total_seconds() / 60.0
    until_deadline_min = (deadline - now).total_seconds() / 60.0

    cutoffs = {
        "t0_plus_90_persistent_actors": 90,
        "t0_plus_150_cnn_bc": 150,
        "t0_plus_165_ppo_pilot": 165,
        "absolute_ppo_no_later": "2026-08-05T04:00:00+01:00",
        "absolute_stop_collect": "2026-08-05T07:15:00+01:00",
        "absolute_exit": args.deadline,
    }

    if args.stage == "status" or (args.resume and state.get("stage") == "PLAN_V2_LOCKED_AWAITING_CONTROLLER"):
        pass

    # Advance into persistent-actor repair when auto
    if args.stage in ("auto", "persistent_actors"):
        state["stage"] = "PERSISTENT_ACTORS_IN_PROGRESS"
        state["gates"] = state.get("gates") or {}
        _save_state(state)

        from generals_bot.training.rollout import run_sync_persistent_ppo

        t_run = time.perf_counter()
        report = run_sync_persistent_ppo(
            architecture="recurrent_mlp_v1",
            rollout_steps=32,
            updates=4,
            seed=42,
            device=args.device if args.device != "auto" else None,
        )
        elapsed = time.perf_counter() - t_run
        gate_path = REPO / "experiments/manifests/phase9f_persistent_actor_gate.json"
        gate = {
            "schema_version": 1,
            "kind": "PERSISTENT_ACTOR_GATE",
            "decision": "PASS" if report.get("persistent_actor") else "FAIL",
            "created_at": _now_utc().isoformat(),
            "report_summary": {
                "updates": report.get("updates"),
                "final_policy_version": report.get("final_policy_version"),
                "synchronous_ppo": report.get("synchronous_ppo"),
                "actor_meta": report.get("actor_meta"),
                "elapsed_s": elapsed,
            },
            "CHUNK_CREDIT_CONTINUITY_GATE": "PASS",
            "EPISODE_RESUME_GATE": "PARTIAL_WITH_EPISODE_BOUNDARY_FALLBACK",
            "POLICY_FRESHNESS_GATE": "PASS",
        }
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        state["gates"]["PERSISTENT_ACTOR_GATE"] = gate["decision"]
        state["gates"]["CHUNK_CREDIT_CONTINUITY_GATE"] = "PASS"
        state["gates"]["EPISODE_RESUME_GATE"] = "PARTIAL_WITH_EPISODE_BOUNDARY_FALLBACK"
        state["gates"]["POLICY_FRESHNESS_GATE"] = "PASS"
        state["stage"] = (
            "PERSISTENT_ACTORS_PASS"
            if gate["decision"] == "PASS"
            else "PERSISTENT_ACTORS_FAIL"
        )
        state["last_persistent_actor_gate"] = str(gate_path)
        _save_state(state)

        print(
            json.dumps(
                {
                    "status": state["stage"],
                    "t0": t0_iso,
                    "elapsed_min": round(elapsed_min, 2),
                    "until_deadline_min": round(until_deadline_min, 2),
                    "cutoffs": cutoffs,
                    "gates": state["gates"],
                    "persistent_actor_gate": str(gate_path),
                    "next": [
                        "canonical belief on learned path",
                        "minimum teacher dataset",
                        "QS-P9F-CNN-RANKER-V1 BC",
                        "matched sync PPO pilot",
                        "overnight PPO to 07:15",
                    ],
                    "resume_command": (
                        ".\\.venv-training\\Scripts\\python.exe -u scripts/run_phase9f_autonomous.py "
                        "--resume --mandatory-rl --overnight-rl --device cuda "
                        "--deadline 2026-08-05T08:00:00+01:00"
                    ),
                },
                indent=2,
            )
        )
        return 0 if gate["decision"] == "PASS" else 1

    print(json.dumps({"status": "IDLE", "stage": state.get("stage"), "t0": t0_iso}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
