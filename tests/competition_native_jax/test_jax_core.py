"""JAX PPO zero-update and forward smoke tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from generals_bot.competition_native_jax.transformer_jax import forward, init_params
from train.competition_native_jax.gae_jax import gae_advantages
from train.competition_native_jax.ppo_jax import assert_zero_update_ratio


def test_jax_forward_and_zero_update() -> None:
    key = jax.random.PRNGKey(0)
    params = init_params(key)
    spatial = jnp.zeros((8, 21, 21), dtype=jnp.float32)
    global_vec = jnp.zeros((8,), dtype=jnp.float32)
    out = forward(params, spatial, global_vec)
    assert out["flat_logits"].shape[0] == 3970
    mask = jnp.zeros_like(out["flat_logits"], dtype=bool).at[0].set(True)
    mask = mask.at[1:30].set(True)
    rho = assert_zero_update_ratio(out["flat_logits"], mask, jnp.array(0))
    assert abs(float(rho) - 1.0) < 1e-5


def test_gae_scan_shapes() -> None:
    rewards = jnp.ones(5)
    values = jnp.zeros(6)
    dones = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0])
    adv, ret = gae_advantages(rewards, values, dones)
    assert adv.shape == (5,)
    assert ret.shape == (5,)
