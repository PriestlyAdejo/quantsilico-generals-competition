"""Compare private castle pricing to the official JAX modifier."""

from __future__ import annotations

import jax.numpy as jnp
from generals.core.game import GameState
from generals.modifiers import build_castles as bc

from generals_bot.castle_cost import castle_price_at


def test_prices_match_official() -> None:
    h, w = 12, 12
    ownership = jnp.zeros((2, h, w), dtype=bool).at[0, 0, 0].set(True)
    generals = jnp.zeros((h, w), dtype=bool).at[0, 0].set(True)
    castles = jnp.zeros((h, w), dtype=bool)
    mountains = jnp.zeros((h, w), dtype=bool)
    passable = ~mountains
    armies = jnp.zeros((h, w), dtype=jnp.int32).at[0, 0].set(50)
    ownership_neutral = jnp.ones((h, w), dtype=bool).at[0, 0].set(False)
    general_positions = jnp.array([[0, 0], [h - 1, w - 1]], dtype=jnp.int32)
    state = GameState(
        armies=armies,
        ownership=ownership,
        ownership_neutral=ownership_neutral,
        generals=generals,
        castles=castles,
        mountains=mountains,
        passable=passable,
        general_positions=general_positions,
        time=jnp.int32(0),
        winner=jnp.int32(-1),
        pool_idx=jnp.int32(0),
    )
    official = bc.build_cost_grid(state, 0)
    structures = [(0, 0)]
    for r in range(h):
        for c in range(w):
            assert int(official[r, c]) == castle_price_at(r, c, structures)
