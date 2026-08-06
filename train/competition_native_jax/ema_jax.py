"""JAX EMA parameter averaging."""

from __future__ import annotations

import jax

from generals_bot.competition_native_jax.constants import EMA_TAU


def ema_update(ema_params, params, tau: float = EMA_TAU):
    return jax.tree_util.tree_map(lambda e, p: tau * e + (1.0 - tau) * p, ema_params, params)
