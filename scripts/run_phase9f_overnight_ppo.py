"""Phase 9F overnight matched CNN-PPO: RL_CONTROL vs RL_CURRICULUM until 07:15 local."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from generals_bot.training.conversion_reward import CONTROL_V1, CURRICULUM_DISCOVERY_V1
from generals_bot.training.rollout import run_sync_persistent_ppo

REPO = Path(__file__).resolve().parents[1]
LOCAL = ZoneInfo("Europe/London")


def seconds_until(iso: str) -> float:
    target = datetime.fromisoformat(iso)
    now = datetime.now(LOCAL)
    if target.tzinfo is None:
        target = target.replace(tzinfo=LOCAL)
    return max(0.0, (target - now).total_seconds())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--stop-at", default="2026-08-05T07:15:00+01:00")
    p.add_argument(
        "--init-checkpoint",
        default=str(REPO / "experiments/phase9f_cnn_ranker_v1/checkpoints/bc/model.json"),
    )
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--max-updates", type=int, default=100000)
    p.add_argument("--arm", choices=["both", "control", "curriculum"], default="both")
    args = p.parse_args()

    stop_s = seconds_until(args.stop_at)
    print(json.dumps({"stop_at": args.stop_at, "seconds_remaining": stop_s, "init": args.init_checkpoint}))
    if stop_s < 60:
        print(json.dumps({"status": "TOO_LATE", "seconds_remaining": stop_s}))
        return 2

    arms = []
    if args.arm in ("both", "control"):
        arms.append(("RL_CONTROL", CONTROL_V1, 42))
    if args.arm in ("both", "curriculum"):
        arms.append(("RL_CURRICULUM", CURRICULUM_DISCOVERY_V1, 42))

    # Split remaining time across arms equally when both
    per_arm = stop_s / max(1, len(arms))
    out_root = REPO / "experiments/phase9f_overnight_ppo"
    out_root.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, reward_cfg, seed in arms:
        remaining = seconds_until(args.stop_at)
        budget = min(per_arm, remaining - 30)  # leave 30s margin between arms
        if budget < 60:
            results[name] = {"status": "SKIPPED_NO_TIME", "remaining_s": remaining}
            continue
        print(json.dumps({"arm": name, "budget_s": budget}))
        arm_out = out_root / name.lower()
        report = run_sync_persistent_ppo(
            architecture="recurrent_cnn_v2",
            rollout_steps=args.rollout_steps,
            updates=args.max_updates,
            seed=seed,
            device=args.device,
            reward_config=reward_cfg,
            init_checkpoint=args.init_checkpoint,
            stop_after_s=budget,
            out_dir=arm_out,
        )
        report.pop("_model", None)
        report.pop("_opt", None)
        report["arm"] = name
        report["candidate_id"] = "QS-P9F-CNN-RANKER-V1"
        report["source"] = "QS-P9F-CNN-RANKER-V1"
        report["stop_at"] = args.stop_at
        (arm_out / "overnight_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        results[name] = {
            "updates": report.get("updates"),
            "final_policy_version": report.get("final_policy_version"),
            "elapsed_s": report.get("elapsed_s"),
            "final_checkpoint": report.get("final_checkpoint"),
            "persistent_actor": report.get("persistent_actor"),
            "synchronous_ppo": report.get("synchronous_ppo"),
        }
        print(json.dumps({"arm_done": name, **results[name]}))

    summary = {
        "schema_version": 1,
        "kind": "PHASE9F_OVERNIGHT_MATCHED_PPO",
        "created_at": datetime.now(LOCAL).isoformat(),
        "stop_at": args.stop_at,
        "init_checkpoint": args.init_checkpoint,
        "arms": results,
        "MANDATORY_RL_GATE": "PASS" if any(v.get("updates", 0) for v in results.values()) else "FAIL_WITH_ATTEMPTED_EVIDENCE",
    }
    man = REPO / "experiments/manifests/phase9f_overnight_matched_ppo.json"
    man.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    state_path = REPO / "experiments/manifests/phase9f_autonomous_run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["stage"] = "OVERNIGHT_PPO_COMPLETE"
    state["gates"]["MANDATORY_RL_GATE"] = summary["MANDATORY_RL_GATE"]
    state["overnight_ppo"] = str(man)
    state["updated_at"] = datetime.utcnow().isoformat() + "+00:00"
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
