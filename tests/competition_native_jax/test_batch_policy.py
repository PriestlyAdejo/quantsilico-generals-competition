"""Tests for batched JAX policy."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from generals_bot.competition_native_jax.competition_env_jax import legal_mask_one_jax, reset_one_jax
from generals_bot.competition_native_jax.transformer_jax import forward, forward_batch, init_params


def test_forward_batch_matches_single():
    key = jax.random.PRNGKey(0)
    params = init_params(key)
    n = 3
    spatial = jax.random.normal(key, (n, 8, 21, 21))
    global_vec = jax.random.normal(jax.random.fold_in(key, 1), (n, 8))
    batched = forward_batch(params, spatial, global_vec)
    for i in range(n):
        single = forward(params, spatial[i], global_vec[i])
        max_logit = float(jnp.max(jnp.abs(batched["flat_logits"][i] - single["flat_logits"])))
        max_val = float(jnp.max(jnp.abs(batched["value_logits"][i] - single["value_logits"])))
        assert max_logit < 5e-2, max_logit
        assert max_val < 5e-2, max_val


def test_legal_mask_pass_always_legal():
    key = jax.random.PRNGKey(0)
    state = reset_one_jax(key, 21, 21)
    mask = legal_mask_one_jax(state, 0)
    assert bool(mask[0])
