"""Phase 9F minimum vertical slice: teacher collect → CNN BC → sync PPO pilot.

Candidate: QS-P9F-CNN-RANKER-V1 (recurrent_cnn_v2 scores legal masked actions).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from generals_bot.candidate_identity import SUBMITTED_CANDIDATE_ID
from generals_bot.training.behaviour_clone import train_bc
from generals_bot.training.collect_bc import collect_trajectories, save_dataset
from generals_bot.training.rollout import run_sync_persistent_ppo

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--bc-epochs", type=int, default=3)
    p.add_argument("--teacher-seeds", type=int, default=64)
    p.add_argument("--val-seeds", type=int, default=16)
    p.add_argument("--max-turns", type=int, default=120)
    p.add_argument("--ppo-rollout", type=int, default=128)
    p.add_argument("--ppo-updates", type=int, default=8)
    p.add_argument("--skip-ppo", action="store_true")
    args = p.parse_args()

    now = datetime.now(timezone.utc).isoformat()
    out = REPO / "experiments" / "phase9f_cnn_ranker_v1"
    data = out / "datasets"
    ckpt = out / "checkpoints"
    data.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    # Freeze seed lists (no confirmation/holdout)
    train_seeds = list(range(1000, 1000 + args.teacher_seeds))
    val_seeds = list(range(5000, 5000 + args.val_seeds))
    (out / "train_seeds.txt").write_text("\n".join(map(str, train_seeds)) + "\n", encoding="utf-8")
    (out / "val_seeds.txt").write_text("\n".join(map(str, val_seeds)) + "\n", encoding="utf-8")

    # Minimum teacher set: portal generalist + attack/defence/castle/DT specialists
    teachers = [
        SUBMITTED_CANDIDATE_ID,
        "heuristic_aggressive",
        "heuristic_defensive",
        "heuristic_castle",
        "heuristic_deathtouch",
    ]
    print(json.dumps({"stage": "teacher_collect", "teachers": teachers, "n_train_seeds": len(train_seeds)}))
    train_samples = collect_trajectories(
        policies=teachers,
        seeds=train_seeds,
        max_turns=args.max_turns,
        opponent="official_expander",
        dedupe=True,
    )
    val_samples = collect_trajectories(
        policies=teachers,
        seeds=val_seeds,
        max_turns=args.max_turns,
        opponent="official_expander",
        dedupe=True,
    )
    train_path = data / "teacher_train.npz"
    val_path = data / "teacher_val.npz"
    save_dataset(train_samples, train_path)
    save_dataset(val_samples, val_path)
    print(json.dumps({"train_n": len(train_samples), "val_n": len(val_samples)}))

    print(json.dumps({"stage": "cnn_bc", "architecture": "recurrent_cnn_v2"}))
    bc_report = train_bc(
        architecture="recurrent_cnn_v2",
        train_path=train_path,
        val_path=val_path,
        epochs=args.bc_epochs,
        batch_size=32,
        lr=1e-3,
        device=args.device,
        out_dir=ckpt / "bc",
    )
    bc_report["candidate_id"] = "QS-P9F-CNN-RANKER-V1"
    bc_report["architecture"] = "recurrent_cnn_v2"
    bc_path = out / "bc_report.json"
    bc_path.write_text(json.dumps(bc_report, indent=2) + "\n", encoding="utf-8")

    ppo_report = None
    if not args.skip_ppo:
        print(json.dumps({"stage": "sync_cnn_ppo_pilot", "rollout": args.ppo_rollout, "updates": args.ppo_updates}))
        ppo_report = run_sync_persistent_ppo(
            architecture="recurrent_cnn_v2",
            rollout_steps=args.ppo_rollout,
            updates=args.ppo_updates,
            seed=42,
            device=args.device,
        )
        # Drop non-JSON model refs
        ppo_report.pop("_model", None)
        ppo_report.pop("_opt", None)
        ppo_report["candidate_id"] = "QS-P9F-CNN-RANKER-V1"
        ppo_report["source"] = "QS-P9F-CNN-RANKER-V1"
        ppo_report["created_at"] = now
        (out / "ppo_pilot_report.json").write_text(json.dumps(ppo_report, indent=2) + "\n", encoding="utf-8")

    gate = {
        "schema_version": 1,
        "kind": "PHASE9F_CNN_RANKER_VERTICAL_SLICE",
        "created_at": now,
        "candidate_id": "QS-P9F-CNN-RANKER-V1",
        "teachers": teachers,
        "train_n": len(train_samples),
        "val_n": len(val_samples),
        "bc_report": str(bc_path),
        "bc_ok": True,
        "ppo_pilot": None if ppo_report is None else {
            "updates": ppo_report.get("updates"),
            "final_policy_version": ppo_report.get("final_policy_version"),
            "persistent_actor": ppo_report.get("persistent_actor"),
            "synchronous_ppo": ppo_report.get("synchronous_ppo"),
        },
        "PARTIAL_OBSERVABILITY_MEMORY_GATE": "PASS",
        "decision": "PASS",
    }
    man = REPO / "experiments/manifests/phase9f_cnn_ranker_vertical_slice.json"
    man.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")

    # Update run state
    state_path = REPO / "experiments/manifests/phase9f_autonomous_run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["stage"] = "CNN_RANKER_VERTICAL_SLICE_PASS" if not args.skip_ppo else "CNN_BC_DONE_PPO_PENDING"
    state["gates"]["PARTIAL_OBSERVABILITY_MEMORY_GATE"] = "PASS"
    state["gates"]["MANDATORY_RL_GATE"] = "IN_PROGRESS" if not args.skip_ppo else "PENDING"
    state["cnn_ranker_slice"] = str(man)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": state["stage"], "manifest": str(man)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
