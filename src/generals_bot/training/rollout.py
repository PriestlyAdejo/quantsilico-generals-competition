"""Rollout helpers for persistent-actor PPO fragments."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from generals_bot.models.legal_mask import apply_action_mask
from generals_bot.models.mlp import RecurrentMLPPolicy
from generals_bot.models.model_forward import adapt_forward_output
from generals_bot.training.actors import PersistentActor, RolloutFragment
from generals_bot.training.ppo import _gae


def fragment_to_buffers(fragment: RolloutFragment) -> dict[str, Any]:
    tr = fragment.transitions
    return {
        "cells": np.stack([t.cells for t in tr]),
        "globs": np.stack([t.glob for t in tr]),
        "actions": np.asarray([t.action for t in tr], dtype=np.int64),
        "old_logp": np.asarray([t.logp for t in tr], dtype=np.float32),
        "values": [t.value for t in tr],
        "rewards": [t.reward for t in tr],
        "terminated": [t.terminated for t in tr],
        "legal_masks": np.stack([t.legal_mask for t in tr]).astype(bool),
        "support_hashes": [t.support_hash for t in tr],
        "bootstrap_value": fragment.bootstrap_value,
        "policy_version": fragment.policy_version,
        "episode_id": fragment.episode_id,
    }


def ppo_update_from_fragment(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    fragment: RolloutFragment,
    *,
    device: torch.device,
    clip: float = 0.2,
    lr: float = 3e-4,
    expected_policy_version: int | None = None,
    mixture_deterministic: bool = True,
    burn_in: int = 0,
) -> dict[str, float]:
    """Synchronous on-policy update for one fragment (policy freshness enforced).

    Uses persisted collection-time legal masks (same support as ``old_logp``) and
    sequential recurrent unrolling (reset hidden after terminations).
    """
    del lr  # optimiser already constructed with learning rate
    if expected_policy_version is not None and fragment.policy_version != expected_policy_version:
        raise RuntimeError(
            f"stale fragment policy_version={fragment.policy_version} "
            f"expected={expected_policy_version}"
        )
    buf = fragment_to_buffers(fragment)
    dones = buf["terminated"]
    advantages, returns = _gae(
        buf["rewards"],
        buf["values"],
        bootstrap_value=buf["bootstrap_value"],
        dones=dones,
    )
    adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
    old_logp = torch.tensor(buf["old_logp"], dtype=torch.float32, device=device)
    cells_t = torch.from_numpy(buf["cells"]).to(device)
    globs_t = torch.from_numpy(buf["globs"]).to(device)
    acts_t = torch.tensor(buf["actions"], dtype=torch.long, device=device)
    masks_t = torch.from_numpy(buf["legal_masks"]).to(device=device, dtype=torch.bool)

    # Replay support hash check (quarantine mismatches from policy ratio).
    from generals_bot.training.action_support import support_hash_from_mask

    support_mismatch = 0
    valid_ratio = torch.ones(acts_t.shape[0], dtype=torch.bool, device=device)
    for i, expected_hash in enumerate(buf["support_hashes"]):
        got = support_hash_from_mask(masks_t[i])
        if got != expected_hash:
            support_mismatch += 1
            valid_ratio[i] = False

    b = cells_t.shape[0]
    # Sequential recurrent unroll (training path); not parallel initial_hidden(b).
    new_logps: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    hidden = model.initial_hidden(1, device=device)
    cell_mem = None
    if hasattr(model, "initial_cell_memory"):
        cell_mem = model.initial_cell_memory(1, device=device)

    for t in range(b):
        if t > 0 and bool(dones[t - 1]):
            hidden = model.initial_hidden(1, device=device)
            if hasattr(model, "initial_cell_memory"):
                cell_mem = model.initial_cell_memory(1, device=device)
        cell_t = cells_t[t : t + 1]
        glob_t = globs_t[t : t + 1]
        if cell_mem is not None:
            raw = model.forward_tensors(
                cell_t, glob_t, hidden, cell_mem, deterministic=mixture_deterministic
            )
        else:
            flat = cell_t.reshape(1, -1) if isinstance(model, RecurrentMLPPolicy) else cell_t
            raw = model.forward_tensors(flat, glob_t, hidden, deterministic=mixture_deterministic)
        fwd = adapt_forward_output(raw)
        if fwd.cell_memory is not None:
            cell_mem = fwd.cell_memory
        hidden = fwd.hidden
        logits = fwd.logits
        masked = apply_action_mask(logits, masks_t[t : t + 1])
        dist = torch.distributions.Categorical(logits=masked)
        new_logps.append(dist.log_prob(acts_t[t : t + 1]))
        entropies.append(dist.entropy())
        values.append(fwd.value)
        if t < burn_in:
            valid_ratio[t] = False

    new_logp = torch.cat(new_logps, dim=0)
    entropy = torch.cat(entropies, dim=0)
    value_pred = torch.cat(values, dim=0)

    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * adv_t
    surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_t
    per_t_policy = -torch.min(surr1, surr2)
    if bool(valid_ratio.any()):
        policy_loss = per_t_policy[valid_ratio].mean()
        entropy_term = entropy[valid_ratio].mean()
    else:
        policy_loss = per_t_policy.mean() * 0.0
        entropy_term = entropy.mean() * 0.0
    value_loss = F.mse_loss(value_pred, ret_t)
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_term
    assert torch.isfinite(loss), "NaN/Inf PPO loss"
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 0.5))
    opt.step()
    with torch.no_grad():
        delta_logp = (new_logp - old_logp).abs()
        ratio_err = (ratio - 1.0).abs()
    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy_term.item() if torch.is_tensor(entropy_term) else entropy.mean().item()),
        "grad_norm": grad_norm,
        "bootstrap_value": float(buf["bootstrap_value"]),
        "continuation_mask": float(fragment.continuation_mask),
        "support_mismatch": float(support_mismatch),
        "valid_ratio_frac": float(valid_ratio.float().mean().item()),
        "max_abs_delta_logp": float(delta_logp.max().item()),
        "max_abs_ratio_err": float(ratio_err.max().item()),
    }


def run_sync_persistent_ppo(
    *,
    architecture: str = "recurrent_mlp_v1",
    rollout_steps: int = 64,
    updates: int = 2,
    lr: float = 3e-4,
    clip: float = 0.2,
    seed: int = 0,
    device: str | None = None,
    reward_config: Any = None,
    actor: PersistentActor | None = None,
    init_checkpoint: Any = None,
    stop_after_s: float | None = None,
    out_dir: Any = None,
) -> dict[str, Any]:
    """Synchronous PPO: freeze policy N, collect, pause, update, publish N+1."""
    from pathlib import Path

    from generals_bot.models.checkpoint import apply_state_dict, save_checkpoint
    from generals_bot.models.factory import build_model
    from generals_bot.training.conversion_reward import CONTROL_V1
    from generals_bot.training.device_policy import assert_module_on_cuda, resolve_training_device

    reward_cfg = reward_config or CONTROL_V1
    reward_cfg.terminal.validate_ordering()
    device_s = resolve_training_device(device, context="ppo.run_sync_persistent_ppo")
    torch_device = torch.device(device_s)
    model = build_model(architecture).to(torch_device)
    if init_checkpoint is not None and Path(init_checkpoint).is_file():
        apply_state_dict(model, init_checkpoint, map_location=torch_device)
    model.train()
    assert_module_on_cuda(model, expected=device_s)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    owned_actor = actor is None
    if actor is None:
        actor = PersistentActor(
            actor_id="actor0",
            seed=seed,
            reward_config=reward_cfg,
            policy_version=0,
        )
    actor.attach_model_state(model, torch_device)

    history: list[dict[str, float]] = []
    policy_version = 0
    meta_snapshots: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    completed_updates = 0

    for update in range(updates):
        if stop_after_s is not None and (time.perf_counter() - t0) >= stop_after_s:
            break
        model.eval()
        fragment = actor.collect_fragment(
            model,
            rollout_steps=rollout_steps,
            device=torch_device,
            policy_version=policy_version,
        )
        assert fragment.policy_version == policy_version
        model.train()
        stats = ppo_update_from_fragment(
            model,
            opt,
            fragment,
            device=torch_device,
            clip=clip,
            lr=lr,
            expected_policy_version=policy_version,
        )
        policy_version += 1
        actor.policy_version = policy_version
        stats["update"] = float(update)
        stats["policy_version_after"] = float(policy_version)
        history.append(stats)
        meta_snapshots.append(actor.snapshot_meta())
        completed_updates += 1
        if out_dir is not None and completed_updates % 10 == 0:
            outp = Path(out_dir)
            outp.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                model,
                outp / f"update_{policy_version}",
                architecture=architecture,
                config=model.config_dict(),  # type: ignore[attr-defined]
            )

    result = {
        "architecture": architecture,
        "device": device_s,
        "rollout_steps": rollout_steps,
        "updates_requested": updates,
        "updates": completed_updates,
        "history": history,
        "final_policy_version": policy_version,
        "actor_meta": actor.snapshot_meta(),
        "actor_meta_per_update": meta_snapshots[-5:],
        "owned_actor": owned_actor,
        "bridge_decision": "PASS",
        "persistent_actor": True,
        "synchronous_ppo": True,
        "checkpoint_resume_mode": actor.checkpoint_resume_mode,
        "elapsed_s": time.perf_counter() - t0,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "_model": model,
        "_opt": opt,
    }
    if out_dir is not None:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model,
            outp / "final",
            architecture=architecture,
            config=model.config_dict(),  # type: ignore[attr-defined]
        )
        result["final_checkpoint"] = str(outp / "final.json")
    return result
