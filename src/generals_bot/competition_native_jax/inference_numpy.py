"""NumPy inference reference (deployment / parity). Re-exports existing transformer."""

from __future__ import annotations

from generals_bot.competition_native_jax.transformer import (  # noqa: F401
    TransformerWeights,
    forward,
    init_weights,
    weights_from_dict,
    weights_to_dict,
)
