"""Top-advantage fraction masking for the PPO policy-gradient signal.

TOPADV family knob (Stage 4A): restrict the policy-gradient signal to the
top ``fraction`` of transitions ranked by |advantage|; the remaining
transitions contribute zero policy gradient (value and entropy terms are
unchanged by construction, since advantages only enter the clipped PG term).

PPO_SEMANTICS: UNCHANGED for serving/action selection - this is a
training-objective ablation; inference and legal action semantics are
untouched. ``fraction == 1.0`` is the exact identity (default programme
behaviour). Ties at the magnitude threshold are included, so the number of
kept transitions is >= ceil(fraction * n).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def top_advantage_mask(advantages: jax.Array, fraction: float) -> jax.Array:
    """Zero all but the top-``fraction`` advantages by magnitude."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"top_advantage_fraction must be in (0, 1], got {fraction}")
    n = int(advantages.shape[0])
    k = max(1, math.ceil(fraction * n))
    if k >= n:
        return advantages
    mag = jnp.abs(advantages)
    top_vals, _ = jax.lax.top_k(mag, k)
    threshold = top_vals[-1]
    return jnp.where(mag >= threshold, advantages, 0.0)
