"""Critical official-engine parity fixtures for Phase 9E Stage 2."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest
from generals import GeneralsEnv
from generals.core import game
from generals.core.game import create_initial_state
from generals.modifiers import build_castles as bc
from generals.modifiers import deathtouch as dt

from generals_bot.evaluation.match import make_board
from generals_bot.models.observation_encoder import MAX_HW, padding_mask
from generals_bot.rules import (
    CASTLE_BASE_COST,
    CASTLE_PROXIMITY_DECAY,
    CASTLE_PROXIMITY_PENALTY,
    DEATHTOUCH_TURN,
    DRAW_TURN,
    LAND_GROWTH_PERIOD,
    MAP_SIZE_MAX,
    MAP_SIZE_MIN,
)

PASS = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
RIGHT = 3


def _open_board(size: int = 8, time: int = 0):
    grid = jnp.zeros((size, size), dtype=jnp.int32).at[0, 0].set(1).at[0, size - 1].set(2)
    return create_initial_state(grid)._replace(time=jnp.int32(time))


def _give(state, player, ij, army):
    i, j = ij
    return state._replace(
        armies=state.armies.at[i, j].set(army),
        ownership=state.ownership.at[player, i, j].set(True),
        ownership_neutral=state.ownership_neutral.at[i, j].set(False),
    )


def test_local_constants_match_competition_preset():
    env = GeneralsEnv(mode="competition")
    assert MAP_SIZE_MIN == 18
    assert MAP_SIZE_MAX == 21
    assert env.min_grid_size == MAP_SIZE_MIN
    assert env.max_grid_size == MAP_SIZE_MAX
    assert env.pad_to == MAX_HW == 21
    assert env.build_castles is True
    assert env.deathtouch_turn == DEATHTOUCH_TURN == 800
    assert CASTLE_BASE_COST == bc.BASE_COST == 35
    assert CASTLE_PROXIMITY_PENALTY == bc.PROXIMITY_PENALTY == 14
    assert CASTLE_PROXIMITY_DECAY == bc.PROXIMITY_DECAY == 2
    assert DRAW_TURN == 1200
    assert LAND_GROWTH_PERIOD == 50


def test_competition_reset_pads_to_21():
    env = GeneralsEnv(mode="competition")
    _pool, state = env.reset(jrandom.PRNGKey(0))
    assert tuple(int(x) for x in state.armies.shape) == (21, 21)
    assert tuple(int(x) for x in state.passable.shape) == (21, 21)


@pytest.mark.parametrize("h,w", [(18, 18), (18, 21), (19, 20), (21, 21)])
def test_make_board_active_geometry_matches_padding_mask(h: int, w: int):
    env = GeneralsEnv(mode="competition")
    found = None
    for seed in range(800):
        state = make_board(env, seed)
        ah, aw = int(state.armies.shape[0]), int(state.armies.shape[1])
        if ah == h and aw == w:
            found = state
            break
    if found is None:
        pytest.skip(f"no make_board sample hit exact {h}x{w} in 800 seeds")
    mask = padding_mask(h, w)
    assert mask.shape == (21, 21)
    assert not mask[:h, :w].any()
    if h < 21:
        assert mask[h:, :].all()
    if w < 21:
        assert mask[:, w:].all()


def test_padding_mask_for_all_supported_active_sizes():
    for h in range(MAP_SIZE_MIN, MAP_SIZE_MAX + 1):
        for w in range(MAP_SIZE_MIN, MAP_SIZE_MAX + 1):
            mask = padding_mask(h, w)
            assert mask.dtype == bool
            assert mask.shape == (MAX_HW, MAX_HW)
            assert bool((~mask[:h, :w]).all())
            if h < MAX_HW:
                assert bool(mask[h:, :].all())
            if w < MAX_HW:
                assert bool(mask[:, w:].all())


def test_move_order_bigger_army_holds_contested_neutral():
    s = _open_board()
    s = _give(s, 0, (2, 1), 25)
    s = _give(s, 1, (2, 3), 40)
    s = s._replace(armies=s.armies.at[2, 2].set(20))
    a0 = jnp.array([0, 2, 1, RIGHT, 0], dtype=jnp.int32)
    a1 = jnp.array([0, 2, 3, 2, 0], dtype=jnp.int32)  # LEFT=2
    s2, _ = game.step(s, jnp.stack([a0, a1]))
    assert bool(s2.ownership[1, 2, 2])
    assert int(s2.armies[2, 2]) == 35


def test_invalid_build_is_consumed_as_pass():
    s = _open_board()
    bad_build = jnp.array([2, 0, 0, 0, 0], dtype=jnp.int32)  # build on general cell
    before = s
    s2, actions = bc.apply_build_actions(s, jnp.stack([bad_build, PASS]))
    assert int(actions[0, 0]) == 1
    assert (s2.armies == before.armies).all()
    assert (s2.castles == before.castles).all()


def test_valid_build_resolves_before_moves_and_deducts_cost():
    s = _open_board()
    s = _give(s, 0, (3, 3), 40)
    price = int(bc.build_cost_grid(s, 0)[3, 3])
    assert price >= CASTLE_BASE_COST
    assert 40 >= price
    build = jnp.array([2, 3, 3, 0, 0], dtype=jnp.int32)
    s2, actions = bc.apply_build_actions(s, jnp.stack([build, PASS]))
    assert bool(s2.castles[3, 3])
    assert int(s2.armies[3, 3]) == 40 - price
    assert int(actions[0, 0]) == 1  # rewritten to pass for subsequent move phase


def test_visibility_is_local_moore_neighbourhood():
    env = GeneralsEnv(mode="competition")
    state = make_board(env, 0)
    vis = np.array(game.get_visibility(state.ownership[0]))
    assert vis.shape == state.armies.shape
    gen = tuple(int(x) for x in np.array(state.general_positions[0]))
    assert bool(vis[gen])
    r0, c0 = gen
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < vis.shape[0] and 0 <= c < vis.shape[1] and bool(state.passable[r, c]):
                assert bool(vis[r, c])


def test_land_growth_every_fifty_turns():
    s = _open_board(size=6, time=49)
    s = _give(s, 0, (1, 1), 3)
    before = int(s.armies[1, 1])
    s2, _ = game.step(s, jnp.stack([PASS, PASS]))
    assert int(s2.time) == 50
    assert int(s2.armies[1, 1]) == before + 1


def test_deathtouch_threshold_aligned():
    assert DEATHTOUCH_TURN == 800
    s = _open_board(size=6, time=800)
    s = _give(s, 0, (0, 4), 5)
    s = s._replace(armies=s.armies.at[0, 5].set(50))
    move = jnp.array([0, 0, 4, RIGHT, 0], dtype=jnp.int32)
    _, info = dt.step(s, jnp.stack([move, PASS]), DEATHTOUCH_TURN)
    assert bool(info.is_done)
    assert int(info.winner) == 0
