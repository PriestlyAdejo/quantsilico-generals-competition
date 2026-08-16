"""BC-A pilot leak guard (predeclared fixture test, bc_a_pilot_round_1_plan.yaml).

Held-in-state features must never contain hidden values: perturbing enemy
army counts on tiles the trained seat CANNOT see must leave the canonical
observation bit-identical, while a perturbation on a VISIBLE tile must show
up (sensitivity control). The observation path under test is exactly the one
the pilot trainer consumes (state_from_tick -> observe_one_jax with fresh fog
memory).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "src", REPO / "third_party" / "generals-bots"):
    entry = str(candidate)
    if entry not in sys.path:
        sys.path.insert(0, entry)

import jax.numpy as jnp  # noqa: E402

from generals_bot.competition_native_jax.competition_env_jax import (  # noqa: E402
    empty_memory,
    observe_one_jax,
)
from scripts.data.replay_engine_oracle import state_from_tick  # noqa: E402

H = W = 11


def _board():
    """p0 owns cols 0-1 + (5,7); p1 owns cols 9-10 plus (5,2) adjacent to p0.

    p0 vision = owned tiles + their 4-neighbours -> reaches col 2 and col 8
    at most; p1 tiles in cols 9-10 are UNSEEN; p1 tile (5,2) is VISIBLE.
    """
    owners = [[-1] * W for _ in range(H)]
    armies = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            armies[r][c] = 1
    for r in range(H):
        for c in (0, 1):
            owners[r][c] = 0
        for c in (9, 10):
            owners[r][c] = 1
    owners[5][2] = 1  # visible enemy tile (adjacent to p0 col 1)
    owners[5][7] = 0  # p0 outpost so col 8 is visible but not col 9
    for r in range(H):
        for c in range(W):
            armies[r][c] = 5 if owners[r][c] >= 0 else 0
    tick = {"owners": owners, "armies": armies}
    return state_from_tick(
        tick,
        dims=(H, W),
        mountains=set(),
        castles=set(),
        generals={0: (5, 0), 1: (5, 10)},
        time=3,
    )


def test_hidden_enemy_army_perturbation_leaves_observation_unchanged():
    state = _board()
    eng = state.engine_state
    # (0,9) and (9,10): enemy-owned, outside p0 vision, not generals
    perturbed = eng._replace(
        armies=eng.armies.at[0, 9].set(eng.armies[0, 9] + 57).at[9, 10].set(3)
    )
    spatial_a, global_a, _ = observe_one_jax(eng, 0, empty_memory())
    spatial_b, global_b, _ = observe_one_jax(perturbed, 0, empty_memory())
    assert bool(jnp.array_equal(spatial_a, spatial_b))
    assert bool(jnp.array_equal(global_a, global_b))


def test_visible_enemy_army_change_DOES_change_observation():
    state = _board()
    eng = state.engine_state
    perturbed = eng._replace(armies=eng.armies.at[5, 2].set(eng.armies[5, 2] + 57))
    spatial_a, _, _ = observe_one_jax(eng, 0, empty_memory())
    spatial_b, _, _ = observe_one_jax(perturbed, 0, empty_memory())
    assert not bool(jnp.array_equal(spatial_a, spatial_b))
