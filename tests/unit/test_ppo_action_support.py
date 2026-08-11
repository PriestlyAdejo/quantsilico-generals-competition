"""Unit tests for PPO action-support replay and step-zero ratio semantics."""

from __future__ import annotations

import numpy as np
import torch

from generals_bot.models.factory import build_model
from generals_bot.training.action_support import support_hash_from_mask
from generals_bot.training.actors import PersistentActor
from generals_bot.training.conversion_reward import CONTROL_V1
from generals_bot.training.device_policy import resolve_training_device
from generals_bot.training.rollout import fragment_to_buffers, ppo_update_from_fragment


def test_support_hash_stable():
    m = np.zeros(16, dtype=bool)
    m[0] = True
    m[3] = True
    assert support_hash_from_mask(m) == support_hash_from_mask(m.copy())
    m2 = m.copy()
    m2[1] = True
    assert support_hash_from_mask(m) != support_hash_from_mask(m2)


def test_fragment_persists_legal_mask_and_support_hash():
    device_s = resolve_training_device(None, context="test_support_persist")
    device = torch.device(device_s)
    model = build_model("recurrent_mlp_v1").to(device)
    actor = PersistentActor(actor_id="sup0", seed=5, reward_config=CONTROL_V1)
    actor.attach_model_state(model, device)
    frag = actor.collect_fragment(
        model, rollout_steps=8, device=device, policy_version=0, mixture_deterministic=True
    )
    assert len(frag.transitions) == 8
    for tr in frag.transitions:
        assert tr.legal_mask.dtype == np.bool_ or tr.legal_mask.dtype == bool
        assert tr.legal_mask.ndim == 1
        assert bool(tr.legal_mask[tr.action])
        assert tr.support_hash == support_hash_from_mask(tr.legal_mask)
        assert tr.support_kind == "FULL_ACTION_SPACE_LEGAL_MASK"


def test_step_zero_ratio_near_one_after_support_fix():
    """Collect then recompute logp with same weights before any optimiser step."""
    device_s = resolve_training_device(None, context="test_step_zero")
    device = torch.device(device_s)
    model = build_model("recurrent_mlp_v1").to(device)
    actor = PersistentActor(actor_id="z0", seed=17, reward_config=CONTROL_V1)
    actor.attach_model_state(model, device)
    frag = actor.collect_fragment(
        model, rollout_steps=16, device=device, policy_version=0, mixture_deterministic=True
    )
    # Zero-LR update path: compute metrics via ppo_update but restore weights
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam(model.parameters(), lr=0.0)
    metrics = ppo_update_from_fragment(
        model,
        opt,
        frag,
        device=device,
        expected_policy_version=0,
        mixture_deterministic=True,
    )
    model.load_state_dict(state)
    # CUDA f32 tolerance from plan; CPU may be similar
    assert metrics["support_mismatch"] == 0.0
    assert metrics["max_abs_delta_logp"] <= 1e-4
    assert metrics["max_abs_ratio_err"] <= 1e-4


def test_policy_version_replay_rejects_stale():
    device_s = resolve_training_device(None, context="test_pv")
    device = torch.device(device_s)
    model = build_model("recurrent_mlp_v1").to(device)
    actor = PersistentActor(actor_id="pv0", seed=3, reward_config=CONTROL_V1)
    actor.attach_model_state(model, device)
    frag = actor.collect_fragment(model, rollout_steps=4, device=device, policy_version=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    try:
        ppo_update_from_fragment(model, opt, frag, device=device, expected_policy_version=1)
        raised = False
    except RuntimeError as exc:
        raised = "stale fragment" in str(exc)
    assert raised


def test_fragment_buffers_include_masks():
    device_s = resolve_training_device(None, context="test_buf")
    device = torch.device(device_s)
    model = build_model("recurrent_mlp_v1").to(device)
    actor = PersistentActor(actor_id="buf0", seed=9, reward_config=CONTROL_V1)
    actor.attach_model_state(model, device)
    frag = actor.collect_fragment(model, rollout_steps=4, device=device, policy_version=0)
    buf = fragment_to_buffers(frag)
    assert buf["legal_masks"].shape[0] == 4
    assert len(buf["support_hashes"]) == 4
