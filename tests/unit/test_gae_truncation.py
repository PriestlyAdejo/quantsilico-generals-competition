"""Unit tests for GAE truncation bootstrap (Phase 9F foundation repair)."""

from __future__ import annotations

from generals_bot.training.ppo import _gae


def test_gae_zero_bootstrap_kills_early_credit_for_late_reward():
    rewards = [0.0] * 64
    rewards[-1] = 1.0
    values = [0.0] * 64
    adv0, _ = _gae(rewards, values, gamma=0.99, lam=1.0, bootstrap_value=0.0)
    # With lambda=1 and zero values, early advantage is gamma**(T-1)
    assert adv0[0] == 0.99**63


def test_gae_nonzero_bootstrap_on_truncation():
    rewards = [0.0] * 8
    values = [0.5] * 8
    dones = [False] * 8
    adv_b, ret_b = _gae(rewards, values, gamma=0.99, lam=0.95, bootstrap_value=10.0, dones=dones)
    adv_z, ret_z = _gae(rewards, values, gamma=0.99, lam=0.95, bootstrap_value=0.0, dones=dones)
    assert adv_b[0] > adv_z[0]
    assert ret_b[-1] > ret_z[-1]


def test_gae_resets_on_done():
    rewards = [0.0, 1.0, 0.0, 0.0]
    values = [0.0, 0.0, 0.0, 0.0]
    dones = [False, True, False, False]
    adv, _ = _gae(rewards, values, gamma=0.99, lam=1.0, bootstrap_value=5.0, dones=dones)
    # After done at t=1, bootstrap should not leak into pre-done advantage beyond the terminal reward
    assert adv[1] == 1.0
    # Post-done segment sees bootstrap
    assert adv[2] > 0.0
