from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
from torch.distributions import Categorical

from generals_bot.models.action_index import ACTION_DIM, PASS_INDEX, index_to_action
from generals_bot.models.checkpoint import apply_state_dict
from generals_bot.models.factory import build_model
from generals_bot.models.legal_mask import apply_action_mask, legal_mask_observation
from generals_bot.models.model_forward import adapt_forward_output
from generals_bot.models.observation_encoder import encode_globals_numpy, encode_grids_numpy
from generals_bot.training.bridge_benchmark import extract_numpy_boards
from generals_bot.training.collect_bc import _action_to_jax, _observation_from_arrays
from generals_bot.training.conversion_reward import CONTROL_V1
from generals_bot.training.actors import PersistentActor


def as_float(x: Any) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    return float(x.item())  # torch scalar


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint-json",
        type=str,
        default=str(
            repo
            / "experiments"
            / "phase9f_overnight_ppo"
            / "rl_control"
            / "final.json"
        ),
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--rollout-steps", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    now = datetime.now(timezone.utc).isoformat()

    # Evidence-driven checkpoint: RL_CONTROL final (the one that reported entropy collapse).
    ckpt_json = Path(args.checkpoint_json)
    if not ckpt_json.exists():
        raise FileNotFoundError(str(ckpt_json))

    device_str = args.device
    torch_device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Small rollout: we only need step-zero consistency signal, not competence.
    rollout_steps = args.rollout_steps
    seed = args.seed

    # Numerical tolerance contract (from Phase 9F-R diagnosis plan).
    # CUDA float32:
    #   abs(new_logp - old_logp) <= 1e-4
    #   abs(ratio - 1.0) <= 1e-4
    if torch_device.type == "cuda":
        tol_abs = 1e-4
        tol_ratio = 1e-4
        tolerance = "CUDA_float32"
    else:
        tol_abs = 1e-5
        tol_ratio = 1e-5
        tolerance = "CPU_float32"

    torch.manual_seed(seed)
    np.random.seed(seed)

    reward_cfg = CONTROL_V1  # training_opponent="pass" keeps environment opponent stable/minimal
    reward_cfg.terminal.validate_ordering()

    actor = PersistentActor(
        actor_id="diag_actor0",
        seed=seed,
        reward_config=reward_cfg,
        policy_version=0,
    )
    actor.attach_model_state(build_model("recurrent_cnn_v2").to(torch_device), torch_device)

    # Load checkpoint into a fresh model (so we know the audit uses exactly this checkpoint).
    model = build_model("recurrent_cnn_v2").to(torch_device)
    apply_state_dict(model, ckpt_json, map_location=torch_device)
    model.eval()
    actor.attach_model_state(model, torch_device)

    # Containers (per transition).
    cells_list: list[np.ndarray] = []
    globs_list: list[np.ndarray] = []
    legal_mask_list: list[np.ndarray] = []  # FULL legal-action mask
    acts_list: list[int] = []
    pass_flags: list[bool] = []

    old_logp_list: list[float] = []  # logp under FULL legal mask at collection-time
    entropy_behaviour_list: list[float] = []
    pass_prob_full_list: list[float] = []

    # Support-mismatch-only: recompute logp using update mask on the *same logits*.
    entropy_update_mask_same_logits_list: list[float] = []
    new_logp_update_mask_same_logits_list: list[float] = []

    # Now collect an episode fragment with persistent hidden/cell_mem.
    state_hash_before = actor.state_hash()
    rollout_ended_early = False

    # Precompute opponent action for "pass" opponent.
    opp_a = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)

    actor.action_history = []
    for _ in range(rollout_steps):
        if rollout_ended_early:
            break

        eng = actor.get_obs(actor.state, actor.learner_seat)
        tg, og, ag, _, meta = extract_numpy_boards(eng, actor.h, actor.w)
        cells = encode_grids_numpy(tg, og, ag)
        obs = _observation_from_arrays(tg, og, ag, meta)
        glob = encode_globals_numpy(obs)

        if actor.belief is not None:
            actor.belief.update_visible(obs)

        cell_t = torch.from_numpy(cells).unsqueeze(0).to(torch_device)
        glob_t = torch.from_numpy(glob).unsqueeze(0).to(torch_device)

        with torch.no_grad():
            if actor.cell_mem is not None:
                raw = model.forward_tensors(
                    cell_t, glob_t, actor.hidden, actor.cell_mem, deterministic=False
                )
            else:
                # RecurrentCNNPolicy, but keep generic for safety.
                raw = model.forward_tensors(
                    cell_t, glob_t, actor.hidden, deterministic=False
                )
            fwd = adapt_forward_output(raw)
            if fwd.cell_memory is not None:
                actor.cell_mem = fwd.cell_memory
            actor.hidden = fwd.hidden
            logits = fwd.logits

            legal_mask = legal_mask_observation(obs, device=torch_device).unsqueeze(0)
            masked_full = apply_action_mask(logits, legal_mask)
            dist_full = Categorical(logits=masked_full)

            action = dist_full.sample()
            acts_idx = int(action.item())
            assert bool(legal_mask[0, acts_idx]), "illegal action sampled"

            old_logp = dist_full.log_prob(action)
            entropy_beh = dist_full.entropy().mean()
            # Softmax-normalised pass probability under the full legal mask.
            probs_full = dist_full.probs
            pass_prob = probs_full[0, PASS_INDEX] if probs_full.ndim == 2 else probs_full[PASS_INDEX]
            pass_prob_full = float(pass_prob.item())

            # Update-time support: only {chosen_action, PASS} (exactly what ppo_update_from_fragment does).
            mask_update = torch.zeros_like(legal_mask, dtype=torch.bool)
            mask_update.scatter_(1, torch.tensor([[acts_idx]], device=torch_device), True)
            mask_update[:, PASS_INDEX] = True
            masked_update = apply_action_mask(logits, mask_update)
            dist_update_same = Categorical(logits=masked_update)
            new_logp_same = dist_update_same.log_prob(action)
            entropy_update_mask_same = dist_update_same.entropy().mean()

        # Store for recomputation later.
        cells_list.append(cells)
        globs_list.append(glob)
        legal_mask_list.append(legal_mask[0].detach().cpu().numpy())
        acts_list.append(acts_idx)
        pass_flags.append(acts_idx == PASS_INDEX)
        old_logp_list.append(as_float(old_logp))
        entropy_behaviour_list.append(as_float(entropy_beh))
        pass_prob_full_list.append(pass_prob_full)
        entropy_update_mask_same_logits_list.append(as_float(entropy_update_mask_same))
        new_logp_update_mask_same_logits_list.append(as_float(new_logp_same))

        # Step the environment using the chosen action (and pass opponent).
        agent_action = index_to_action(acts_idx)
        actor.state, info = actor.transition_fn(
            actor.state, jnp.stack([_action_to_jax(agent_action), opp_a])
        )
        actor.turn += 1

        reward = 0.0
        terminated = bool(info.is_done)
        truncated = False
        if terminated:
            rollout_ended_early = True

    state_hash_after = actor.state_hash()

    # Recompute "new_logp" using the update implementation semantics:
    # - logits recomputed in a single forward on the batch
    # - hidden and cell_mem reset to zeros/initial_cell_memory for each transition
    # - policy distribution support restricted to {acts_t, PASS}
    b = len(acts_list)
    if b == 0:
        raise RuntimeError("rollout produced zero transitions")

    cells_np = np.stack(cells_list)
    globs_np = np.stack(globs_list)
    acts_np = np.asarray(acts_list, dtype=np.int64)

    cells_t = torch.from_numpy(cells_np).to(torch_device)
    globs_t = torch.from_numpy(globs_np).to(torch_device)
    acts_t = torch.tensor(acts_np, dtype=torch.long, device=torch_device)

    with torch.no_grad():
        hidden = model.initial_hidden(b, device=torch_device)
        if hasattr(model, "initial_cell_memory"):
            cell_mem = model.initial_cell_memory(b, device=torch_device)
            raw = model.forward_tensors(cells_t, globs_t, hidden, cell_mem, deterministic=False)
        else:
            raw = model.forward_tensors(cells_t, globs_t, hidden, deterministic=False)
        fwd = adapt_forward_output(raw)
        logits_update_impl = fwd.logits

        mask_update = torch.zeros(b, ACTION_DIM, dtype=torch.bool, device=torch_device)
        mask_update.scatter_(1, acts_t.unsqueeze(1), True)
        mask_update[:, PASS_INDEX] = True
        masked = apply_action_mask(logits_update_impl, mask_update)
        dist_impl = Categorical(logits=masked)
        new_logp_impl = dist_impl.log_prob(acts_t)
        entropy_update_impl = dist_impl.entropy().mean()

    old_logp_t = torch.tensor(old_logp_list, dtype=torch.float32, device=torch_device)
    new_logp_impl_t = new_logp_impl.to(torch.float32)
    ratio_update_impl = torch.exp(new_logp_impl_t - old_logp_t)

    abs_diff_new_old = torch.abs(new_logp_impl_t - old_logp_t)
    abs_diff_ratio = torch.abs(ratio_update_impl - 1.0)

    pass_abs = (abs_diff_new_old <= tol_abs).to(torch.bool)
    pass_ratio = (abs_diff_ratio <= tol_ratio).to(torch.bool)

    transitions_passed = int((pass_abs & pass_ratio).sum().item())

    # Entropy figures.
    h_behaviour = float(np.mean(entropy_behaviour_list))
    h_update_mask_same_logits = float(np.mean(entropy_update_mask_same_logits_list))
    h_update_impl = float(entropy_update_impl.item())
    pass_prob_full_mean = float(np.mean(pass_prob_full_list))
    pass_prob_full_min = float(np.min(pass_prob_full_list))

    # Support stats.
    legal_support_sizes = np.asarray([int(m.sum()) for m in legal_mask_list], dtype=np.int32)
    update_support_sizes = np.asarray([1 if a == PASS_INDEX else 2 for a in acts_list], dtype=np.int32)

    results = {
        "schema_version": 1,
        "kind": "PHASE9FR_PPO_STEP_ZERO_RATIO_ENTROPY_AUDIT",
        "created_at": now,
        "checkpoint_json": str(ckpt_json),
        "architecture": "recurrent_cnn_v2",
        "reward_config": reward_cfg.version,
        "rollout_steps_requested": rollout_steps,
        "rollout_steps_collected": b,
        "torch_device": str(torch_device),
        "numerical_tolerance_contract": {
            "tolerance_profile": tolerance,
            "abs_new_minus_old_logp_le": tol_abs,
            "abs_ratio_minus_1_le": tol_ratio,
        },
        "step_zero_ratio_gate": {
            "transitions_passed": transitions_passed,
            "transitions_total": b,
            "pass_rate": transitions_passed / max(1, b),
        },
        "entropy_diagnosis": {
            "behaviour_entropy_full_legal_mean": h_behaviour,
            "update_mask_entropy_same_logits_mean": h_update_mask_same_logits,
            "update_impl_entropy_mean": h_update_impl,
        },
        "support_diagnostics": {
            "support_kind_old": "FULL_ACTION_SPACE_LEGAL_MASK (as used in actors.py)",
            "support_kind_new_update_mask": "HYBRID_EXECUTABLE_SET (chosen action + PASS only; as used in ppo_update_from_fragment)",
            "legal_support_sizes_mean": float(legal_support_sizes.mean()),
            "legal_support_sizes_min": int(legal_support_sizes.min()),
            "legal_support_sizes_max": int(legal_support_sizes.max()),
            "update_support_sizes_unique": sorted(list(set(update_support_sizes.tolist()))),
            "pass_action_fraction": float(np.mean(np.asarray(pass_flags, dtype=np.float32))),
            "pass_probability_full_legal_mean": pass_prob_full_mean,
            "pass_probability_full_legal_min": pass_prob_full_min,
        },
        "hidden_state_and_sequence_provenance": {
            "actor_state_hash_before": state_hash_before,
            "actor_state_hash_after": state_hash_after,
            "note": "Old_logp computed with persistent actor hidden/cell_mem evolving sequentially. New_logp_impl computed by ppo_update_from_fragment semantics which reset hidden/cell_mem to model.initial_* per-transition batch element.",
        },
        "ratios_summary": {
            "ratio_update_impl_mean": float(ratio_update_impl.mean().item()),
            "ratio_update_impl_min": float(ratio_update_impl.min().item()),
            "ratio_update_impl_max": float(ratio_update_impl.max().item()),
            "abs_new_minus_old_logp_mean": float(abs_diff_new_old.mean().item()),
            "abs_new_minus_old_logp_max": float(abs_diff_new_old.max().item()),
        },
        "entropy_update_support_effect": {
            "entropy_update_mask_same_logits_minus_behaviour_mean": h_update_mask_same_logits - h_behaviour
        },
        # For quick manual spot checks, store the first few per-step scalar pairs.
        "step0_samples": [
            {
                "t": i,
                "acts_idx": acts_list[i],
                "old_logp_full": old_logp_list[i],
                "new_logp_impl_update_mask": float(new_logp_impl[i].item()),
                "ratio_update_impl": float(ratio_update_impl[i].item()),
                "abs_delta_logp": float(abs_diff_new_old[i].item()),
                "legal_support_size": int(legal_support_sizes[i]),
                "update_support_size": int(update_support_sizes[i]),
                "entropy_full_legal": entropy_behaviour_list[i],
                "entropy_update_mask_same_logits": entropy_update_mask_same_logits_list[i],
                "pass_probability_full_legal": pass_prob_full_list[i],
            }
            for i in range(min(8, b))
        ],
    }

    ckpt_stem = ckpt_json.stem
    arm_tag = (
        "rl_control"
        if "rl_control" in str(ckpt_json).lower()
        else ("rl_curriculum" if "rl_curriculum" in str(ckpt_json).lower() else "unknown")
    )
    # Include ckpt_stem so intermediate updates don't overwrite each other.
    out_path = repo / "experiments" / "manifests" / f"phase9fr_ppo_ratio_semantics_{arm_tag}_{ckpt_stem}.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    md_path = repo / "experiments" / "reports" / f"phase9fr_ppo_ratio_semantics_{arm_tag}_{ckpt_stem}.md"
    md_lines: list[str] = []
    md_lines.append("# Phase 9F-R PPO ratio semantics audit (step-zero)")
    md_lines.append("")
    md_lines.append(f"Created: {now}")
    md_lines.append("")
    md_lines.append(f"Checkpoint: `{ckpt_json}`")
    md_lines.append("")
    md_lines.append("## Step-zero gate")
    md_lines.append("")
    md_lines.append(
        f"- Pass transitions: {transitions_passed}/{b} ({transitions_passed / max(1, b):.3%})"
    )
    md_lines.append(
        f"- Numerical tolerance: {tolerance} (abs(logp_delta) <= {tol_abs}; abs(ratio-1) <= {tol_ratio})"
    )
    md_lines.append("")
    md_lines.append("## Entropy diagnosis (means)")
    md_lines.append("")
    md_lines.append(f"- Behaviour entropy (FULL legal mask): {h_behaviour:.6f}")
    md_lines.append(f"- Update-mask entropy on same logits: {h_update_mask_same_logits:.6f}")
    md_lines.append(f"- Update-impl entropy (ppo_update_from_fragment semantics): {h_update_impl:.6f}")
    md_lines.append("")
    md_lines.append("## Support mismatch evidence")
    md_lines.append("")
    md_lines.append(f"- Mean legal support size: {legal_support_sizes.mean():.2f} (min={legal_support_sizes.min()}, max={legal_support_sizes.max()})")
    md_lines.append(f"- Update support sizes unique (chosen vs PASS only): {sorted(list(set(update_support_sizes.tolist())))}")
    md_lines.append(f"- PASS action fraction: {np.mean(np.asarray(pass_flags, dtype=np.float32)):.2%}")
    md_lines.append("")
    md_lines.append("## Interpretation constraints")
    md_lines.append("")
    md_lines.append(
        "Old_logp uses persistent sequential hidden/cell_mem evolution. "
        "New_logp_impl uses update semantics which reset hidden/cell_mem per-transition batch element. "
        "Therefore ratio failure can be caused by (1) support mismatch, (2) hidden/state reconstruction mismatch, "
        "or (3) stochastic strategic-mixture sampling differences."
    )
    md_lines.append("")
    md_lines.append("## Samples (first few transitions)")
    md_lines.append("")
    md_lines.append("See JSON for detailed per-step numbers.")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("written", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

