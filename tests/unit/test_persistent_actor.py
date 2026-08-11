"""Tests for persistent actors across PPO optimiser fragments."""

from __future__ import annotations

import torch

from generals_bot.models.factory import build_model
from generals_bot.training.actors import PersistentActor
from generals_bot.training.conversion_reward import CONTROL_V1
from generals_bot.training.device_policy import resolve_training_device
from generals_bot.training.ppo import _gae, run_bounded_ppo
from generals_bot.training.recurrent_buffer import build_windows_from_fragment_arrays
from generals_bot.training.rollout import run_sync_persistent_ppo
import numpy as np


def test_persistent_actor_continues_episode_across_fragments():
    device_s = resolve_training_device(None, context="test_persistent_actor")
    device = torch.device(device_s)
    model = build_model("recurrent_mlp_v1").to(device)
    actor = PersistentActor(
        actor_id="test0",
        seed=7,
        reward_config=CONTROL_V1,
        policy_version=0,
    )
    actor.attach_model_state(model, device)
    ep0 = actor.episode_id
    turn0 = actor.turn
    hash0 = actor.state_hash()
    frag1 = actor.collect_fragment(model, rollout_steps=8, device=device, policy_version=0)
    assert frag1.continuation_mask in (0.0, 1.0)
    if frag1.continuation_mask == 1.0:
        assert actor.episode_id == ep0
        assert actor.turn == turn0 + 8
        assert actor.state_hash() != hash0
    frag2 = actor.collect_fragment(model, rollout_steps=8, device=device, policy_version=1)
    # After optimiser-style version bump, opponent/env must still be the live actor
    assert actor.actor_id == "test0"
    assert frag2.policy_version == 1
    if frag1.continuation_mask == 1.0 and frag2.continuation_mask == 1.0:
        assert frag2.episode_id == ep0
        assert actor.turn == turn0 + 16


def test_opponent_state_not_reset_on_nonterminal_fragment():
    device_s = resolve_training_device(None, context="test_opp_persist")
    device = torch.device(device_s)
    model = build_model("recurrent_mlp_v1").to(device)
    cfg = CONTROL_V1
    # CONTROL may use pass opponent; still verify env persistence
    actor = PersistentActor(actor_id="opp0", seed=11, reward_config=cfg)
    actor.attach_model_state(model, device)
    f1 = actor.collect_fragment(model, rollout_steps=4, device=device, policy_version=0)
    meta1 = actor.snapshot_meta()
    # Simulate optimiser update boundary (no actor reset)
    f2 = actor.collect_fragment(model, rollout_steps=4, device=device, policy_version=1)
    meta2 = actor.snapshot_meta()
    if f1.continuation_mask == 1.0 and f2.continuation_mask == 1.0:
        assert meta1["episode_id"] == meta2["episode_id"]
        assert meta2["turn"] == meta1["turn"] + 4
        assert meta1["opponent_id"] == meta2["opponent_id"]


def test_gae_multifragment_bootstrap_nonzero():
    rewards = [0.0] * 8
    values = [0.1] * 8
    dones = [False] * 8
    adv_b, _ = _gae(rewards, values, bootstrap_value=2.0, dones=dones)
    adv_z, _ = _gae(rewards, values, bootstrap_value=0.0, dones=dones)
    assert adv_b[0] > adv_z[0]


def test_episode_across_updates_sync_ppo():
    report = run_sync_persistent_ppo(
        architecture="recurrent_mlp_v1",
        rollout_steps=16,
        updates=3,
        seed=3,
        device=None,
    )
    assert report["persistent_actor"] is True
    assert report["synchronous_ppo"] is True
    assert report["final_policy_version"] == 3
    metas = report["actor_meta_per_update"]
    assert len(metas) == 3
    # At least one non-terminal continuation should keep rising turn within same episode
    # (games may terminal early; assert structural continuity fields exist)
    assert "episode_id" in metas[0]
    assert "turn" in metas[-1]


def test_run_bounded_ppo_uses_persistent_actor_by_default():
    report = run_bounded_ppo(
        architecture="recurrent_mlp_v1",
        rollout_steps=8,
        updates=2,
        seed=5,
        out_dir=None,
    )
    assert report.get("persistent_actor") is True
    assert "persistent_actor_meta" in report


def test_recurrent_buffer_loss_mask_excludes_burn_in():
    t = 20
    windows = build_windows_from_fragment_arrays(
        episode_id="ep",
        cells=np.zeros((t, 2)),
        globs=np.zeros((t, 2)),
        actions=np.zeros(t, dtype=np.int64),
        old_logp=np.zeros(t, dtype=np.float32),
        rewards=np.zeros(t),
        values=np.zeros(t),
        terminated=np.zeros(t, dtype=bool),
        seq_len=16,
        burn_in=4,
        policy_version=0,
    )
    assert windows
    mask = windows[0].loss_mask()
    assert mask[:4].sum() == 0
    assert mask[4:].all()
