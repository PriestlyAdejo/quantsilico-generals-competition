"""JAX masked log-softmax, sampling, and inference helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from generals_bot.competition_native_jax.transformer_jax import forward


def masked_log_softmax(logits: jax.Array, mask: jax.Array) -> jax.Array:
    neg_inf = jnp.finfo(jnp.float32).min
    x = jnp.where(mask, logits, neg_inf)
    x = x - jax.nn.logsumexp(jnp.where(mask, x, neg_inf))
    return jnp.where(mask, x, neg_inf)


def sample_action(
    key: jax.Array, logits: jax.Array, mask: jax.Array
) -> tuple[jax.Array, jax.Array]:
    logp = masked_log_softmax(logits, mask)
    # Sample from the masked log distribution directly.  Constructing a
    # float32 CDF via random.choice is unnecessary and, over very large
    # rollout counts, can select outside zero-probability support at a
    # rounding boundary.  Categorical preserves the exact masked support.
    idx = jax.random.categorical(key, logp).astype(jnp.int32)
    return idx, logp[idx]


def greedy_action(logits: jax.Array, mask: jax.Array) -> jax.Array:
    logp = masked_log_softmax(logits, mask)
    return jnp.argmax(jnp.where(mask, logp, jnp.finfo(jnp.float32).min))


def infer(
    params: dict,
    spatial: jax.Array,
    global_vec: jax.Array,
    mask: jax.Array,
    key: jax.Array | None = None,
):
    out = forward(params, spatial, global_vec)
    if key is None:
        idx = greedy_action(out["flat_logits"], mask)
        logp = masked_log_softmax(out["flat_logits"], mask)[idx]
    else:
        idx, logp = sample_action(key, out["flat_logits"], mask)
    return idx, logp, out
