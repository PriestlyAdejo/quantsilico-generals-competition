"""Stage 4 — step-zero + one-update correctness from BC (gates Candidate C only)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
BC = REPO / "experiments" / "phase9f_cnn_ranker_v1" / "checkpoints" / "bc" / "model.json"
OUT = REPO / "experiments" / "phase9fs_one_update"
TOL = 1e-4
WALL_S = 15 * 60
MAX_TRANSITIONS = 1024


def main() -> int:
    started = time.perf_counter()
    now = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)

    from generals_bot.models.checkpoint import apply_state_dict, save_checkpoint
    from generals_bot.models.factory import build_model
    from generals_bot.training.actors import PersistentActor
    from generals_bot.training.conversion_reward import CONTROL_V1
    from generals_bot.training.device_policy import resolve_training_device
    from generals_bot.training.rollout import ppo_update_from_fragment

    device_s = resolve_training_device(None, context="phase9fs_one_update")
    device = torch.device(device_s)
    meta = json.loads(BC.read_text(encoding="utf-8"))
    arch = meta["architecture"]
    model = build_model(arch).to(device)
    apply_state_dict(model, BC, map_location=device)
    model.train()

    actor = PersistentActor(actor_id="oneupd", seed=123, reward_config=CONTROL_V1)
    actor.attach_model_state(model, device)
    steps = min(64, MAX_TRANSITIONS)
    frag = actor.collect_fragment(
        model, rollout_steps=steps, device=device, policy_version=0, mixture_deterministic=True
    )

    # Step-zero: lr=0 compute
    state0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt0 = torch.optim.Adam(model.parameters(), lr=0.0)
    step0 = ppo_update_from_fragment(
        model, opt0, frag, device=device, expected_policy_version=0, mixture_deterministic=True
    )
    model.load_state_dict(state0)

    step0_pass = (
        step0["support_mismatch"] == 0.0
        and step0["max_abs_delta_logp"] <= TOL
        and step0["max_abs_ratio_err"] <= TOL
    )

    # One real update
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    one = ppo_update_from_fragment(
        model, opt, frag, device=device, expected_policy_version=0, mixture_deterministic=True
    )
    out_ckpt = OUT / "one_update.json"
    save_checkpoint(model, out_ckpt, architecture=arch, config=meta.get("config"))

    # BC retention smoke: reload BC and confirm load works (weights diverge after update)
    model_bc = build_model(arch).to(device)
    apply_state_dict(model_bc, BC, map_location=device)
    resume_ok = True
    try:
        apply_state_dict(model_bc, out_ckpt, map_location=device)
    except Exception as exc:  # noqa: BLE001
        resume_ok = False
        resume_err = str(exc)
    else:
        resume_err = None

    elapsed = time.perf_counter() - started
    gate = {
        "schema_version": 1,
        "kind": "PHASE9FS_ONE_UPDATE_CORRECTNESS",
        "created_at": now,
        "device": device_s,
        "architecture": arch,
        "bc_checkpoint": str(BC.as_posix()),
        "transitions": steps,
        "wall_s": elapsed,
        "wall_budget_s": WALL_S,
        "tolerance": TOL,
        "step_zero": step0,
        "step_zero_pass": step0_pass,
        "one_update": one,
        "one_update_checkpoint": str(out_ckpt.as_posix()),
        "bc_retention_reload_ok": True,
        "resume_from_one_update_ok": resume_ok,
        "resume_error": resume_err,
        "gate_status": "PASS"
        if step0_pass and resume_ok and elapsed <= WALL_S
        else "FAIL",
        "gates_candidate_c_only": True,
    }
    (REPO / "experiments" / "manifests" / "phase9fs_one_update_correctness.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    (REPO / "experiments" / "reports" / "phase9fs_one_update_correctness.md").write_text(
        "\n".join(
            [
                "# One-update correctness (Candidate C path)",
                "",
                f"Created: {now}",
                "",
                f"- Gate: **{gate['gate_status']}**",
                f"- Step-zero pass: {step0_pass}",
                f"- max|Δlogp|: {step0.get('max_abs_delta_logp')}",
                f"- max|ratio-1|: {step0.get('max_abs_ratio_err')}",
                f"- support_mismatch: {step0.get('support_mismatch')}",
                f"- Device: {device_s}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"gate_status": gate["gate_status"], "step_zero_pass": step0_pass}, indent=2))
    return 0 if gate["gate_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
