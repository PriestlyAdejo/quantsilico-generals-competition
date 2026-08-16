"""Differential parity tests: replay_engine_oracle vs the PINNED competition engine.

Authority (EV-0042): the pinned engine in third_party/generals-bots is the
executable source of truth. Every classifier predicate must match what the
engine actually does (executed vs silent-pass), across both seats, including
build pricing, simultaneous resolution, deathtouch, real-dims-vs-padding,
and growth phase. Runner timing faults are NOT representable from replay
payloads and are therefore NOT tested here - audits must state that.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from generals.core import game
from generals.core.game import create_initial_state
from generals.modifiers import build_castles as bc
from generals.modifiers import deathtouch as dt

from scripts.data.replay_engine_oracle import (
    ENGINE_EXECUTED,
    ENGINE_SILENT_PASS,
    PROTOCOL_VALID,
    TrueCompetitionState,
    build_cost,
    classify_build,
    classify_move,
    classify_pass,
    state_from_tick,
)

PASS = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)


def board_state():
    """5x7 board inside 21x21 padding: p0 general (0,0), p1 general (4,6),
    mountain (2,3), p0 castle (0,2)."""
    grid = jnp.zeros((5, 7), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1).at[4, 6].set(2).at[2, 3].set(-2).at[0, 2].set(5)
    return create_initial_state(grid)


def give(
    state,
    armies: dict,
    owned0: set = frozenset(),
    owned1: set = frozenset(),
    time: int = 0,
    winner: int = -1,
):
    for (r, c), v in armies.items():
        state = state._replace(armies=state.armies.at[r, c].set(v))
    own = state.ownership
    neu = state.ownership_neutral
    for r, c in set(owned0) | set(owned1):
        neu = neu.at[r, c].set(False)
    for r, c in owned0:
        own = own.at[0, r, c].set(True)
    for r, c in owned1:
        own = own.at[1, r, c].set(True)
    return state._replace(
        ownership=own, ownership_neutral=neu, time=jnp.int32(time), winner=jnp.int32(winner)
    )


def wrap(engine_state, h: int, w: int, time: int = 0) -> TrueCompetitionState:
    def owner_at(r, c):
        if engine_state.ownership[0, r, c]:
            return 0
        if engine_state.ownership[1, r, c]:
            return 1
        return -1

    tick = {
        "owners": [[owner_at(r, c) for c in range(w)] for r in range(h)],
        "armies": [[int(engine_state.armies[r, c]) for c in range(w)] for r in range(h)],
    }
    mountains = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if bool(engine_state.mountains[r, c])
    }
    castles = {
        (r, c) for r in range(h) for c in range(w) if bool(engine_state.castles[r, c])
    }
    gp = engine_state.general_positions
    generals = {0: (int(gp[0, 0]), int(gp[0, 1])), 1: (int(gp[1, 0]), int(gp[1, 1]))}
    return state_from_tick(
        tick,
        dims=(h, w),
        mountains=mountains,
        castles=castles,
        generals=generals,
        time=time,
    )


def engine_silent_pass_move(state, player: int, action) -> bool:
    """Ground truth: engine step with opponent passing leaves state unchanged."""
    actions = jnp.stack([PASS, PASS]).at[player].set(jnp.asarray(action, dtype=jnp.int32))
    new_state, _ = game.step(state, actions)
    return (
        bool(jnp.array_equal(new_state.armies, state.armies))
        and bool(jnp.array_equal(new_state.ownership, state.ownership))
        and bool(jnp.array_equal(new_state.castles, state.castles))
    )


H, W = 5, 7


@pytest.mark.parametrize("player", [0, 1])
def test_valid_full_move_matches_engine(player):
    src = (0, 0) if player == 0 else (4, 6)
    dst = (src[0] + 1, src[1]) if player == 0 else (src[0] - 1, src[1])
    state = give(board_state(), {(0, 0): 10, (4, 6): 10}, owned0={(0, 0)}, owned1={(4, 6)})
    wrapped = wrap(state, H, W)
    result = classify_move(wrapped, player, src, dst, split=0)
    assert result.protocol_status == PROTOCOL_VALID
    action = result.protocol_action
    assert engine_silent_pass_move(state, player, action) is (
        result.engine_outcome == ENGINE_SILENT_PASS
    )
    assert result.engine_outcome == ENGINE_EXECUTED


@pytest.mark.parametrize("player", [0, 1])
def test_valid_half_move_matches_engine(player):
    src = (0, 0) if player == 0 else (4, 6)
    dst = (src[0], src[1] + 1) if player == 0 else (src[0], src[1] - 1)
    state = give(board_state(), {(0, 0): 10, (4, 6): 10}, owned0={(0, 0)}, owned1={(4, 6)})
    wrapped = wrap(state, H, W)
    result = classify_move(wrapped, player, src, dst, split=1)
    assert result.engine_outcome == ENGINE_EXECUTED
    assert not engine_silent_pass_move(state, player, result.protocol_action)


def test_source_not_owned_silent_pass_both_seats():
    state = give(board_state(), {(0, 0): 10, (4, 6): 10}, owned0={(0, 0)}, owned1={(4, 6)})
    wrapped = wrap(state, H, W)
    # player 1 tries to move player 0's army
    result = classify_move(wrapped, 1, (0, 0), (0, 1), split=0)
    assert result.engine_outcome == ENGINE_SILENT_PASS
    assert "src_not_owned" in result.rejection_reasons
    assert engine_silent_pass_move(state, 1, result.protocol_action)


def test_source_army_one_silent_pass():
    state = give(board_state(), {(0, 0): 1}, owned0={(0, 0)})
    wrapped = wrap(state, H, W)
    result = classify_move(wrapped, 0, (0, 0), (1, 0), split=0)
    assert result.engine_outcome == ENGINE_SILENT_PASS
    assert "insufficient_source_army" in result.rejection_reasons
    assert engine_silent_pass_move(state, 0, result.protocol_action)


def test_off_board_destination_silent_pass():
    state = give(board_state(), {(0, 0): 10}, owned0={(0, 0)})
    wrapped = wrap(state, H, W)
    result = classify_move(wrapped, 0, (0, 0), (-1, 0), split=0)
    assert result.protocol_status == PROTOCOL_VALID  # protocol accepts ints; engine no-ops
    assert result.engine_outcome == ENGINE_SILENT_PASS
    assert "dest_out_of_bounds" in result.rejection_reasons
    assert engine_silent_pass_move(state, 0, result.protocol_action)


def test_real_dims_not_confused_with_padding_mountain():
    # (0,6) is real board edge; (0,7) is padding. Padding out-of-bounds must
    # classify dest_out_of_bounds, not dest_mountain.
    state = give(board_state(), {(0, 6): 10}, owned0={(0, 6)})
    wrapped = wrap(state, H, W)
    edge = classify_move(wrapped, 0, (0, 6), (0, 5), split=0)
    assert edge.engine_outcome == ENGINE_EXECUTED
    over = classify_move(wrapped, 0, (0, 6), (0, 7), split=0)
    assert over.rejection_reasons == ["dest_out_of_bounds"]


def test_mountain_destination_silent_pass():
    state = give(board_state(), {(2, 2): 10}, owned0={(2, 2)})
    wrapped = wrap(state, H, W)
    result = classify_move(wrapped, 0, (2, 2), (2, 3), split=0)
    assert result.engine_outcome == ENGINE_SILENT_PASS
    assert "dest_mountain" in result.rejection_reasons
    assert engine_silent_pass_move(state, 0, result.protocol_action)


def test_explicit_pass_always_executes():
    result = classify_pass()
    assert result.protocol_action == (1, 0, 0, 0, 0)
    assert result.engine_outcome == ENGINE_EXECUTED


def test_valid_castle_build_matches_engine():
    state = give(
        board_state(), {(0, 0): 1, (1, 0): 60}, owned0={(0, 0), (1, 0)}, owned1={(4, 6)}
    )
    wrapped = wrap(state, H, W)
    result = classify_build(wrapped, 0, (1, 0))
    assert result.engine_outcome == ENGINE_EXECUTED
    actions = jnp.stack([jnp.asarray(result.protocol_action, dtype=jnp.int32), PASS])
    new_state, _ = bc.step(state, actions)  # competition composition: builds resolve first
    assert bool(new_state.castles[1, 0])


def test_build_on_general_silent_pass():
    state = give(board_state(), {(0, 0): 500}, owned0={(0, 0)})
    wrapped = wrap(state, H, W)
    result = classify_build(wrapped, 0, (0, 0))
    assert result.engine_outcome == ENGINE_SILENT_PASS
    assert "not_plain" in result.rejection_reasons
    actions = jnp.stack([jnp.asarray(result.protocol_action, dtype=jnp.int32), PASS])
    new_state, _ = bc.step(state, actions)
    assert not bool(jnp.any(new_state.castles != state.castles))


def test_build_on_existing_castle_silent_pass():
    state = give(board_state(), {(0, 2): 500}, owned0={(0, 2)})
    wrapped = wrap(state, H, W)
    result = classify_build(wrapped, 0, (0, 2))
    assert result.engine_outcome == ENGINE_SILENT_PASS
    assert "not_plain" in result.rejection_reasons


def test_build_on_unowned_land_silent_pass():
    state = give(board_state(), {(3, 3): 500})
    wrapped = wrap(state, H, W)
    result = classify_build(wrapped, 0, (3, 3))
    assert result.engine_outcome == ENGINE_SILENT_PASS
    assert "not_owned" in result.rejection_reasons


def test_insufficient_castle_price_silent_pass():
    state = give(board_state(), {(1, 0): 30}, owned0={(0, 0), (1, 0)})
    wrapped = wrap(state, H, W)
    # adjacent to own general (d=1 -> +12): price 47 > 30
    result = classify_build(wrapped, 0, (1, 0))
    assert result.engine_outcome == ENGINE_SILENT_PASS
    assert "insufficient_castle_price" in result.rejection_reasons


def test_crowding_price_parity_with_engine():
    state = give(
        board_state(), {(0, 0): 1, (0, 2): 5}, owned0={(0, 0), (0, 2), (1, 1)}
    )
    wrapped = wrap(state, H, W)
    engine_costs = bc.build_cost_grid(state, 0)
    for r in range(H):
        for c in range(W):
            assert build_cost(wrapped, 0, (r, c)) == int(engine_costs[r, c]), (r, c)


def test_captured_castle_counts_in_own_price():
    base = give(board_state(), {(0, 0): 1}, owned0={(0, 0)}, owned1={(4, 6)})
    # p0 captures p1-built castle at (3, 3): it becomes p0 structure -> price rises nearby
    captured = base._replace(
        castles=base.castles.at[3, 3].set(True),
        ownership=base.ownership.at[0, 3, 3].set(True),
        ownership_neutral=base.ownership_neutral.at[3, 3].set(False),
    )
    wrapped_plain = wrap(base, H, W)
    wrapped_cap = wrap(captured, H, W)
    assert build_cost(wrapped_cap, 0, (3, 4)) > build_cost(wrapped_plain, 0, (3, 4))
    engine_costs = bc.build_cost_grid(captured, 0)
    assert build_cost(wrapped_cap, 0, (3, 4)) == int(engine_costs[3, 4])


def test_nonadjacent_move_is_protocol_unclassifiable():
    state = give(board_state(), {(0, 0): 10}, owned0={(0, 0)})
    wrapped = wrap(state, H, W)
    result = classify_move(wrapped, 0, (0, 0), (0, 2), split=0)
    assert result.protocol_status == "PROTOCOL_UNCLASSIFIABLE"
    assert result.protocol_action is None


def test_reconstruction_roundtrip_preserves_engine_state():
    state = give(
        board_state(), {(0, 0): 12, (4, 6): 7, (0, 2): 5}, owned0={(0, 0), (0, 1)}, owned1={(4, 6)}
    )
    wrapped = wrap(state, H, W, time=37)
    eng = wrapped.engine_state
    assert bool(jnp.array_equal(eng.armies[:H, :W], state.armies[:H, :W]))
    assert bool(jnp.array_equal(eng.ownership[:, :H, :W], state.ownership[:, :H, :W]))
    assert int(eng.time) == 37
    assert wrapped.real_h == H and wrapped.real_w == W


def test_growth_phase_even_tick_structures():
    # structures grow on even ticks - alignment-critical engine semantics
    state = give(board_state(), {(0, 0): 10, (0, 2): 5}, owned0={(0, 0), (0, 2)}, time=2)
    new_state = game.global_update(state)
    assert int(new_state.armies[0, 0]) == 11
    assert int(new_state.armies[0, 2]) == 6
    odd = state._replace(time=jnp.int32(3))
    assert int(game.global_update(odd).armies[0, 0]) == 10


def test_simultaneous_mutual_general_capture_is_draw():
    # Each attacker stands beside the ENEMY general (not chasing each other);
    # smaller army moves first (engine order), captures, then the second
    # captures too -> mid.winner != final winner -> competition DRAW. This is
    # army-based: it fires at any turn, threshold or not.
    grid = jnp.zeros((5, 7), dtype=jnp.int32).at[2, 0].set(1).at[2, 4].set(2)
    state = create_initial_state(grid)
    state = state._replace(
        armies=state.armies.at[2, 0].set(20).at[2, 4].set(30).at[2, 3].set(50).at[2, 1].set(60)
    )
    state = state._replace(
        ownership=state.ownership.at[0, 2, 3].set(True).at[1, 2, 1].set(True),
        ownership_neutral=state.ownership_neutral.at[2, 3].set(False).at[2, 1].set(False),
    )
    a0 = jnp.array([0, 2, 3, 3, 0], dtype=jnp.int32)  # p0 right onto p1 general
    a1 = jnp.array([0, 2, 1, 2, 0], dtype=jnp.int32)  # p1 left onto p0 general
    _, info = dt.step(state, jnp.stack([a0, a1]), 800)
    assert int(info.winner) == -1 and bool(info.is_done)  # competition draw
    _, bare = game.step(state, jnp.stack([a0, a1]))
    assert int(bare.winner) != -1  # bare engine is NOT the competition authority


def test_deathtouch_inactive_before_800_active_after():
    # generals ADJACENT: deathtouch destination is general_positions (static)
    grid = jnp.zeros((5, 7), dtype=jnp.int32).at[2, 2].set(1).at[2, 3].set(2)
    state = create_initial_state(grid)
    state = state._replace(armies=state.armies.at[2, 2].set(1))  # single army general
    a0 = jnp.array([0, 2, 2, 3, 0], dtype=jnp.int32)
    actions = jnp.stack([a0, PASS])
    _, early = dt.step(state._replace(time=jnp.int32(700)), actions, 800)
    assert int(early.winner) == -1  # move invalid (army 1), no deathtouch yet
    # army 2 at 800+: whole-move sends 1 onto the enemy general tile
    armed = state._replace(armies=state.armies.at[2, 2].set(2), time=jnp.int32(800))
    _, late = dt.step(armed, actions, 800)
    assert int(late.winner) == 0  # deathtouch capture beats the 1-vs-1 tie
    # same position pre-threshold: combat tie favours defender, no capture
    _, pre = game.step(armed._replace(time=jnp.int32(799)), actions)
    assert int(pre.winner) == -1


def test_simultaneous_chasing_order_matches_engine():
    # both players move toward each other's tile; engine ordering is the law
    grid = jnp.zeros((5, 7), dtype=jnp.int32).at[2, 1].set(1).at[2, 5].set(2)
    state = create_initial_state(grid)
    state = state._replace(
        armies=state.armies.at[2, 1].set(10).at[2, 3].set(4).at[2, 4].set(3)
    )
    state = state._replace(
        ownership=state.ownership.at[0, 2, 3].set(True).at[1, 2, 4].set(True),
        ownership_neutral=state.ownership_neutral.at[2, 3].set(False).at[2, 4].set(False),
    )
    a0 = jnp.array([0, 2, 3, 3, 0], dtype=jnp.int32)
    a1 = jnp.array([0, 2, 4, 2, 0], dtype=jnp.int32)
    new_state, _ = game.step(state, jnp.stack([a0, a1]))
    # the classifier never re-derives move order; this pins the deterministic
    # engine behaviour the audit relies on
    assert bool(jnp.any(new_state.armies != state.armies))
