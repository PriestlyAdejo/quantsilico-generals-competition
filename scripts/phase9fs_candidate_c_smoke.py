"""Bounded Candidate C repaired-PPO smoke (ceilings: ≤16 updates, ≤4096 transitions, ≤30 min)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
BC = REPO / "experiments" / "phase9f_cnn_ranker_v1" / "checkpoints" / "bc" / "model.json"
OUT = REPO / "experiments" / "phase9fs_candidate_c_smoke"
MAX_UPDATES = 4  # keep well under 16 for first smoke
MAX_TRANSITIONS = 256
WALL_S = 30 * 60


def main() -> int:
    one = json.loads(
        (REPO / "experiments" / "manifests" / "phase9fs_one_update_correctness.json").read_text(
            encoding="utf-8"
        )
    )
    if one.get("gate_status") != "PASS":
        raise SystemExit("one-update gate must PASS before Candidate C smoke")

    started = time.perf_counter()
    now = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)

    from generals_bot.models.checkpoint import apply_state_dict, save_checkpoint
    from generals_bot.models.factory import build_model
    from generals_bot.training.actors import PersistentActor
    from generals_bot.training.conversion_reward import CONTROL_V1
    from generals_bot.training.device_policy import resolve_training_device
    from generals_bot.training.rollout import ppo_update_from_fragment

    device_s = resolve_training_device(None, context="phase9fs_candidate_c_smoke")
    device = torch.device(device_s)
    meta = json.loads(BC.read_text(encoding="utf-8"))
    arch = meta["architecture"]
    model = build_model(arch).to(device)
    apply_state_dict(model, BC, map_location=device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    actor = PersistentActor(actor_id="candC", seed=7, reward_config=CONTROL_V1)
    actor.attach_model_state(model, device)

    history = []
    transitions_total = 0
    policy_version = 0
    for u in range(MAX_UPDATES):
        if time.perf_counter() - started > WALL_S:
            break
        if transitions_total >= MAX_TRANSITIONS:
            break
        steps = min(64, MAX_TRANSITIONS - transitions_total)
        frag = actor.collect_fragment(
            model,
            rollout_steps=steps,
            device=device,
            policy_version=policy_version,
            mixture_deterministic=True,
        )
        metrics = ppo_update_from_fragment(
            model,
            opt,
            frag,
            device=device,
            expected_policy_version=policy_version,
            mixture_deterministic=True,
        )
        policy_version += 1
        transitions_total += len(frag.transitions)
        history.append({"update": u, "policy_version": policy_version, **metrics})

    ckpt = OUT / "smoke_final.json"
    save_checkpoint(model, ckpt, architecture=arch, config=meta.get("config"))
    elapsed = time.perf_counter() - started
    valid_tps = transitions_total / max(elapsed, 1e-6)
    doc = {
        "schema_version": 1,
        "kind": "CANDIDATE_C_REPAIRED_PPO_SMOKE",
        "created_at": now,
        "gate_status": "PASS" if history and all(h.get("support_mismatch", 1) == 0 for h in history) else "FAIL",
        "device": device_s,
        "architecture": arch,
        "updates": len(history),
        "transitions": transitions_total,
        "wall_s": elapsed,
        "valid_ppo_learning_transitions_per_second": valid_tps,
        "tier1_note": (
            "This smoke reports valid learning TPS after semantics repair; "
            "Tier-1 thresholds ( >=100 and >=25x baseline ) may still be unmet."
        ),
        "history": history,
        "checkpoint": str(ckpt.as_posix()),
        "compare_to_frozen_v001": "PENDING_UPLOAD_FREEZE",
        "optional_v002_recommendation": "NOT_YET",
    }
    (REPO / "experiments" / "manifests" / "phase9fs_candidate_c_smoke.json").write_text(
        json.dumps(doc, indent=2) + "\n", encoding="utf-8"
    )
    (REPO / "experiments" / "reports" / "phase9fs_candidate_c_smoke.md").write_text(
        "\n".join(
            [
                "# Candidate C repaired PPO smoke",
                "",
                f"Created: {now}",
                f"- Status: **{doc['gate_status']}**",
                f"- Updates: {len(history)} / transitions: {transitions_total}",
                f"- Valid learning TPS: {valid_tps:.2f}",
                f"- Support mismatches: {[h.get('support_mismatch') for h in history]}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    # refresh candidate_c status
    status = {
        "schema_version": 1,
        "kind": "CANDIDATE_C_SMOKE_STATUS",
        "created_at": now,
        "status": "SMOKE_" + doc["gate_status"],
        "valid_ppo_learning_transitions_per_second": valid_tps,
        "compare_to_frozen_v001": "AFTER_UPLOAD_FREEZE",
        "optional_v002": "ONLY_IF_STRONGER_THAN_FROZEN_V001",
    }
    (REPO / "experiments" / "manifests" / "phase9fs_candidate_c_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gate_status": doc["gate_status"], "tps": valid_tps, "updates": len(history)}, indent=2))
    return 0 if doc["gate_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
