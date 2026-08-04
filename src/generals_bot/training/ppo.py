"""Bounded recurrent PPO trainer (bridge smoke + Phase 9D continuation)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from generals import GeneralsEnv
from generals.core import game

from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.models.action_index import ACTION_DIM, index_to_action
from generals_bot.models.checkpoint import apply_state_dict, save_checkpoint
from generals_bot.models.factory import build_model
from generals_bot.models.legal_mask import apply_action_mask
from generals_bot.models.mlp import RecurrentMLPPolicy
from generals_bot.models.model_forward import adapt_forward_output
from generals_bot.models.observation_encoder import encode_globals_numpy, encode_grids_numpy
from generals_bot.training.bridge_benchmark import extract_numpy_boards
from generals_bot.training.collect_bc import _action_to_jax, _observation_from_arrays
from generals_bot.training.conversion_reward import (
    CONTROL_V1,
    RewardConfig,
    assert_no_privileged_keys,
    count_visible_enemy_cells,
)
from generals_bot.training.device_policy import assert_module_on_cuda, resolve_training_device


def _gae(rewards: list[float], values: list[float], gamma: float = 0.99, lam: float = 0.95) -> tuple[list[float], list[float]]:
    advantages: list[float] = []
    gae = 0.0
    values = values + [0.0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1], strict=True)]
    return advantages, returns


def run_bounded_ppo(
    *,
    architecture: str = "recurrent_mlp_v1",
    rollout_steps: int = 64,
    updates: int = 2,
    lr: float = 3e-4,
    clip: float = 0.2,
    seed: int = 0,
    device: str | None = None,
    init_checkpoint: Path | None = None,
    out_dir: Path | None = None,
    reward_config: RewardConfig | None = None,
) -> dict[str, Any]:
    reward_cfg = reward_config or CONTROL_V1
    reward_cfg.terminal.validate_ordering()
    device = resolve_training_device(device, context="ppo.run_bounded_ppo")
    torch_device = torch.device(device)
    model = build_model(architecture).to(torch_device)
    if init_checkpoint and Path(init_checkpoint).is_file():
        apply_state_dict(model, init_checkpoint, map_location=torch_device)
    model.train()
    assert_module_on_cuda(model, expected=device)
    ema = build_model(architecture).to(torch_device)
    ema.load_state_dict(model.state_dict())
    ema.eval()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    env = GeneralsEnv(mode="competition")
    transition = make_transition(env)
    get_obs = game.get_observation
    state = make_board(env, seed)
    h, w = (int(d) for d in state.armies.shape)

    opp_policy = None
    opp_state = None
    if reward_cfg.training_opponent != "pass":
        from generals_bot.observation import GameContext
        from generals_bot.policies.base import TraceLevel
        from generals_bot.selector import create_policy

        opp_policy = create_policy(reward_cfg.training_opponent, seed=seed)
        opp_state = opp_policy.initial_state(GameContext(1, h, w))

    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    legal_rate = 1.0
    episode_shaping = 0.0
    prev_enemy_cells = 0
    discovered = False

    for update in range(updates):
        cells_buf: list[np.ndarray] = []
        glob_buf: list[np.ndarray] = []
        act_buf: list[int] = []
        logp_buf: list[float] = []
        val_buf: list[float] = []
        rew_buf: list[float] = []
        hidden = model.initial_hidden(1, device=torch_device)
        cell_mem = None
        if hasattr(model, "initial_cell_memory"):
            cell_mem = model.initial_cell_memory(1, device=torch_device)

        for _ in range(rollout_steps):
            eng = get_obs(state, 0)
            tg, og, ag, _g, meta = extract_numpy_boards(eng, h, w)
            cells = encode_grids_numpy(tg, og, ag)
            obs = _observation_from_arrays(tg, og, ag, meta)
            glob = encode_globals_numpy(obs)
            cell_t = torch.from_numpy(cells).unsqueeze(0).to(torch_device)
            glob_t = torch.from_numpy(glob).unsqueeze(0).to(torch_device)
            with torch.no_grad():
                if cell_mem is not None:
                    raw = model.forward_tensors(cell_t, glob_t, hidden, cell_mem, deterministic=False)
                else:
                    flat = cell_t.reshape(1, -1) if isinstance(model, RecurrentMLPPolicy) else cell_t
                    raw = model.forward_tensors(flat, glob_t, hidden, deterministic=False)
                fwd = adapt_forward_output(raw)
                if fwd.cell_memory is not None:
                    cell_mem = fwd.cell_memory
                hidden = fwd.hidden
                logits = fwd.logits
                from generals_bot.models.legal_mask import legal_mask_observation

                mask = legal_mask_observation(obs, device=torch_device).unsqueeze(0)
                masked = apply_action_mask(logits, mask)
                dist = torch.distributions.Categorical(logits=masked)
                action = dist.sample()
                logp = dist.log_prob(action)
                value = fwd.value
            idx = int(action.item())
            assert bool(mask[0, idx]), "illegal action sampled"
            cells_buf.append(cells)
            glob_buf.append(glob)
            act_buf.append(idx)
            logp_buf.append(float(logp.item()))
            val_buf.append(float(value.item()))

            agent_action = index_to_action(idx)
            if opp_policy is None:
                opp_a = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
            else:
                from generals_bot.observation import GameContext
                from generals_bot.policies.base import TraceLevel

                eng1 = get_obs(state, 1)
                t1, o1, a1, _, m1 = extract_numpy_boards(eng1, h, w)
                obs1 = _observation_from_arrays(t1, o1, a1, m1)
                d1 = opp_policy.act(obs1, opp_state, deterministic=True, trace=TraceLevel.NONE, deadline=None)
                opp_state = d1.new_state
                opp_a = _action_to_jax(d1.action)
            state, info = transition(state, jnp.stack([_action_to_jax(agent_action), opp_a]))

            eng_next = get_obs(state, 0)
            _, og_next, _, _, _ = extract_numpy_boards(eng_next, h, w)
            next_enemy = count_visible_enemy_cells(og_next)
            assert_no_privileged_keys({"owner_grid_visible": True, "prev_enemy_cells": prev_enemy_cells})
            shaping, discovered = reward_cfg.contact_shaping.step_bonus(
                prev_enemy_cells=prev_enemy_cells,
                curr_enemy_cells=next_enemy,
                episode_cum=episode_shaping,
                discovered=discovered,
            )
            episode_shaping += shaping
            prev_enemy_cells = next_enemy

            reward = float(shaping)
            if bool(info.is_done):
                winner = int(info.winner)
                term = reward_cfg.terminal.terminal_reward(
                    winner=None if winner < 0 else winner, perspective=0
                )
                reward += float(term)
                state = make_board(env, seed + update + 1)
                h, w = (int(d) for d in state.armies.shape)
                hidden = model.initial_hidden(1, device=torch_device)
                if cell_mem is not None:
                    cell_mem = model.initial_cell_memory(1, device=torch_device)
                episode_shaping = 0.0
                prev_enemy_cells = 0
                discovered = False
                if opp_policy is not None:
                    from generals_bot.observation import GameContext

                    opp_state = opp_policy.initial_state(GameContext(1, h, w))
            rew_buf.append(reward)

        advantages, returns = _gae(rew_buf, val_buf)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=torch_device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=torch_device)
        old_logp = torch.tensor(logp_buf, dtype=torch.float32, device=torch_device)
        cells_t = torch.from_numpy(np.stack(cells_buf)).to(torch_device)
        globs_t = torch.from_numpy(np.stack(glob_buf)).to(torch_device)
        acts_t = torch.tensor(act_buf, dtype=torch.long, device=torch_device)

        b = cells_t.shape[0]
        hidden = model.initial_hidden(b, device=torch_device)
        if hasattr(model, "initial_cell_memory"):
            cell_mem = model.initial_cell_memory(b, device=torch_device)
            raw = model.forward_tensors(cells_t, globs_t, hidden, cell_mem, deterministic=False)
        else:
            flat = cells_t.reshape(b, -1) if isinstance(model, RecurrentMLPPolicy) else cells_t
            raw = model.forward_tensors(flat, globs_t, hidden, deterministic=False)
        fwd = adapt_forward_output(raw)
        logits = fwd.logits
        mask = torch.zeros(b, ACTION_DIM, dtype=torch.bool, device=torch_device)
        mask.scatter_(1, acts_t.unsqueeze(1), True)
        mask[:, 0] = True
        masked = apply_action_mask(logits, mask)
        dist = torch.distributions.Categorical(logits=masked)
        new_logp = dist.log_prob(acts_t)
        entropy = dist.entropy().mean()
        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(fwd.value, ret_t)
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        assert torch.isfinite(loss), "NaN/Inf PPO loss"
        with torch.no_grad():
            approx_kl = float((old_logp - new_logp).mean().item())
            clip_fraction = float(((ratio - 1.0).abs() > clip).float().mean().item())
            adv_mean = float(adv_t.mean().item())
            adv_std = float(adv_t.std(unbiased=False).item())
            var_y = float(ret_t.var(unbiased=False).item())
            explained_variance = (
                float(1.0 - (ret_t - fwd.value.detach()).var(unbiased=False).item() / (var_y + 1e-8))
                if var_y > 1e-12
                else 0.0
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 0.5))
        opt.step()
        with torch.no_grad():
            for p, q in zip(ema.parameters(), model.parameters(), strict=True):
                p.mul_(0.99).add_(q, alpha=0.01)

        history.append(
            {
                "update": float(update),
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "explained_variance": explained_variance,
                "grad_norm": grad_norm,
                "learning_rate": float(lr),
                "advantage_mean": adv_mean,
                "advantage_std": adv_std,
            }
        )

    custom_out = out_dir is not None
    out_dir = out_dir or Path("experiments/checkpoints/ppo") / architecture
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model"
    save_checkpoint(model, ckpt, architecture=architecture, config=model.config_dict())  # type: ignore[attr-defined]
    resumed = build_model(architecture).to(torch_device)
    apply_state_dict(resumed, ckpt.with_suffix(".json"), map_location=torch_device)
    resumed.eval()
    with torch.inference_mode():
        h0 = resumed.initial_hidden(1, device=torch_device)
        dummy_cells = torch.zeros(1, 10, 21, 21, device=torch_device)
        dummy_glob = torch.zeros(1, 9, device=torch_device)
        if hasattr(resumed, "initial_cell_memory"):
            raw = resumed.forward_tensors(
                dummy_cells, dummy_glob, h0, resumed.initial_cell_memory(1, device=torch_device)
            )
        else:
            flat = dummy_cells.reshape(1, -1) if isinstance(resumed, RecurrentMLPPolicy) else dummy_cells
            raw = resumed.forward_tensors(flat, dummy_glob, h0)
        assert torch.isfinite(adapt_forward_output(raw).logits).all()

    from generals_bot.training.telemetry_schema import annotate_history

    report = {
        "architecture": architecture,
        "device": device,
        "rollout_steps": rollout_steps,
        "updates": updates,
        "history": history,
        "telemetry": annotate_history(history, producer="generals_bot.training.ppo"),
        "legal_action_rate": legal_rate,
        "elapsed_s": time.perf_counter() - t0,
        "checkpoint": str(ckpt.with_suffix(".json")),
        "resume_ok": True,
        "nan_free": True,
        "bridge_decision": "PASS",
        "reward_config": reward_cfg.to_dict(),
        "note": "Bounded PPO; Phase 9D may pass reward_config for conversion curriculum.",
    }
    (out_dir / "ppo_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not custom_out:
        man = Path("experiments/manifests/ppo_smoke.json")
        man.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["manifest"] = str(man)
    return report


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Bounded PPO smoke (not INITIAL/OVERNIGHT/MARATHON)")
    p.add_argument("--architecture", default="recurrent_mlp_v1")
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--updates", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--init-checkpoint", default=None)
    args = p.parse_args()
    report = run_bounded_ppo(
        architecture=args.architecture,
        rollout_steps=args.rollout_steps,
        updates=args.updates,
        seed=args.seed,
        device=args.device,
        init_checkpoint=Path(args.init_checkpoint) if args.init_checkpoint else None,
    )
    man = Path("experiments/manifests") / f"ppo_smoke_{args.architecture}.json"
    man.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["manifest"] = str(man)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
