"""JAX GAE with reverse lax.scan."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def gae_advantages(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    *,
    gamma: float = 1.0,
    lam: float = 0.9,
) -> tuple[jax.Array, jax.Array]:
    """rewards/dones [T], values [T+1]. Returns advantages [T], returns [T]."""

    def body(carry, inputs):
        last_gae = carry
        reward, value, next_value, done = inputs
        next_nonterminal = 1.0 - done
        delta = reward + gamma * next_value * next_nonterminal - value
        gae = delta + gamma * lam * next_nonterminal * last_gae
        return gae, gae

    rewards_r = rewards[::-1]
    values_r = values[:-1][::-1]
    next_values_r = values[1:][::-1]
    dones_r = dones[::-1]
    _, adv_r = jax.lax.scan(body, jnp.array(0.0, dtype=jnp.float32), (rewards_r, values_r, next_values_r, dones_r))
    adv = adv_r[::-1]
    returns = adv + values[:-1]
    return adv, returns


def gae_advantages_batch(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    *,
    gamma: float = 1.0,
    lam: float = 0.9,
) -> tuple[jax.Array, jax.Array]:
    """Batched device-resident GAE.

    rewards/dones: [T, B]
    values: [T+1, B]
    returns advantages/returns: [T, B]

    Single reverse lax.scan over time; environment batch is an array dimension.
    No Python per-environment loop.
    """

    def body(last_gae, inputs):
        reward, value, next_value, done = inputs
        next_nonterminal = 1.0 - done
        delta = reward + gamma * next_value * next_nonterminal - value
        gae = delta + gamma * lam * next_nonterminal * last_gae
        return gae, gae

    b = rewards.shape[1]
    rewards_r = rewards[::-1]
    values_r = values[:-1][::-1]
    next_values_r = values[1:][::-1]
    dones_r = dones[::-1]
    init = jnp.zeros((b,), dtype=rewards.dtype)
    _, adv_r = jax.lax.scan(body, init, (rewards_r, values_r, next_values_r, dones_r))
    adv = adv_r[::-1]
    returns = adv + values[:-1]
    return adv, returns


gae_advantages_jit = jax.jit(gae_advantages, static_argnames=("gamma", "lam"))
gae_advantages_batch_jit = jax.jit(gae_advantages_batch, static_argnames=("gamma", "lam"))
