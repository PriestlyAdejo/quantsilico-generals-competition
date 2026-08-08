from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from generals_bot.competition_native_jax.inference_jax import sample_action


def test_large_batched_sampling_never_escapes_legal_support() -> None:
    # Independent draws across the full 3,970-action support while keeping the
    # deadline-path CPU/JAX regression bounded.
    draws = 256
    action_count = 3_970
    keys = jax.random.split(jax.random.PRNGKey(20260808), draws)
    logits = jnp.linspace(-20.0, 20.0, action_count, dtype=jnp.float32)
    legal_indices = jnp.asarray([0, 17, 3969], dtype=jnp.int32)
    mask = jnp.zeros((action_count,), dtype=bool).at[legal_indices].set(True)

    actions, logp = jax.jit(jax.vmap(sample_action, in_axes=(0, None, None)))(
        keys, logits, mask
    )

    sampled = np.asarray(actions)
    assert np.isin(sampled, np.asarray([0, 17, 3969])).all()
    assert np.isfinite(np.asarray(logp)).all()
