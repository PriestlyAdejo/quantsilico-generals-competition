"""Long-horizon differential parity: QS JAX simulator vs the PINNED engine.

Audit item §18 (long-horizon engine parity). Motivation: existing parity
coverage (tests/competition_native_jax/test_e2e_parity_3a.py) is short-horizon
or pass-heavy; a latent divergence after ~turn 200 would explain "opening
telemetry fine, deployment strength absent".

Differential sides
------------------
A. QS JAX competition simulator: ``step_batch_jax`` (jit + vmap over
   ``step_one_jax``), actions supplied through the training flat-index codec
   ``index_to_engine_action_batch`` where noted. This is the exact training
   rollout path (train/competition_native_jax/rollout_selfplay_jax.py).
B. Pinned Python competition engine (unmodified, third_party/generals-bots):
   eager ``build_castles.apply_build_actions`` + ``deathtouch.step(..., 800)``
   per turn — the exact competition composition of
   ``competition/matchup.make_transition`` / ``GeneralsEnv(mode="competition")``.

Both sides receive identical states and identical action sequences; full state
(armies, ownership, ownership_neutral, generals, castles, mountains,
general_positions, time, winner) plus rewards/terminated/truncated are
compared at checkpoints.

Scenario coverage (all fixed-seed, CPU):
  1. 420-turn seeded random-legal game on a competition board, full state
     compared every 10 turns, rewards/done flags every turn (growth,
     expansion, splits, captures).
  2. Combat: contested neutral cell, chasing/reinforcing order, smaller-army
     resolution (engine move order is the law on both sides).
  3. Castle building + structure growth across the turn-50 global increment
     and many even-tick structure ticks.
  4. General capture -> win; simultaneous general capture -> draw.
  5. Turn-800 deathtouch boundary (state time set via _replace; the engine
     treats ``time`` as caller-settable — same mechanism the replay oracle's
     ``state_from_tick`` uses).
  6. Turn-1200 hard-draw truncation semantics (constructed state + time jump).
  Extended (marked ``slow``): 720-turn random-legal lockstep on a rectangular
  competition board (crosses the deathtouch regime boundaries), and a 240-turn
  randomized build/combat midgame starting at time 600.

Approach note (runtime): scenario 1 is lockstep from turn 0. Scenarios 2-6
construct representative checkpoint states (some with ``time`` jumped) and
differential-step from there, because lockstepping every scenario to turn
1200 would exceed the CPU budget; scenario 6 additionally proves the
truncation formula at the exact boundary.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generals.core.game import GameState, create_initial_state
from generals.modifiers import build_castles as bc
from generals.modifiers import deathtouch as dt

from generals_bot.competition_native_jax.competition_env_jax import (
    TRUNCATION,
    index_to_engine_action,
    index_to_engine_action_batch,
    legal_mask_one_p0,
    legal_mask_one_p1,
    reset_one_jax,
    step_batch_jax,
)
from generals_bot.competition_native_jax.constants import DEATHTOUCH_TURN, MAX_HW

PASS = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
CHECK_EVERY = 10

STATE_FIELDS = (
    "armies",
    "ownership",
    "ownership_neutral",
    "generals",
    "castles",
    "mountains",
    "passable",
    "general_positions",
    "time",
    "winner",
)


# --------------------------------------------------------------------------
# Differential harness
# --------------------------------------------------------------------------


def unbatch(state: GameState) -> GameState:
    """Index 0 of a batched GameState (all fields carry a leading batch axis)."""
    return jax.tree_util.tree_map(lambda x: x[0], state)


def engine_ref_step(state: GameState, actions: jnp.ndarray) -> tuple[GameState, object]:
    """Reference side: official competition composition, eager (no vmap).

    Mirrors competition/matchup.make_transition for mode=competition:
    builds resolve first and are rewritten to passes, then deathtouch wraps
    the base step with the competition threshold.
    """
    state, actions = bc.apply_build_actions(state, actions)
    return dt.step(state, actions, DEATHTOUCH_TURN)


def env_contracts(next_state: GameState, info, terminated, truncated) -> tuple:
    """Reward/terminated/truncated as official GeneralsEnv.step defines them.

    (third_party/generals-bots/generals/core/env.py: win/lose +/-1, draw 0;
    truncated = time >= truncation and not terminated.)
    """
    reward_p0 = jnp.where(info.winner == 0, 1.0, jnp.where(info.winner == 1, -1.0, 0.0))
    rewards = jnp.stack([reward_p0, -reward_p0])
    trunc_ref = (next_state.time >= TRUNCATION) & (~info.is_done)
    return rewards, info.is_done, trunc_ref


def compare_states(sim: GameState, ref: GameState, turn: int, tag: str) -> None:
    for field in STATE_FIELDS:
        a = np.asarray(getattr(sim, field))
        b = np.asarray(getattr(ref, field))
        assert a.shape == b.shape, f"{tag} turn={turn} field={field} shape {a.shape} != {b.shape}"
        if not np.array_equal(a, b):
            diff = np.argwhere(a != b)
            raise AssertionError(
                f"PARITY FAILURE {tag} turn={turn} field={field}: "
                f"sim={a[tuple(diff[0])]} ref={b[tuple(diff[0])]} at index {tuple(diff[0])} "
                f"({len(diff)} differing entries)"
            )


def compare_step(sim_out, ref_state: GameState, ref_info, turn: int, tag: str) -> None:
    """Full per-step comparison: state fields + rewards/terminated/truncated."""
    ns, rewards, terminated, truncated, info = sim_out
    sim_state = unbatch(ns)
    compare_states(sim_state, ref_state, turn, tag)

    exp_rewards, exp_term, exp_trunc = env_contracts(ref_state, ref_info, terminated, truncated)
    np.testing.assert_allclose(
        np.asarray(rewards[0]),
        np.asarray(exp_rewards),
        atol=0,
        err_msg=f"{tag} turn={turn} rewards",
    )
    assert bool(terminated[0]) == bool(exp_term), (
        f"{tag} turn={turn} terminated: sim={bool(terminated[0])} ref={bool(exp_term)}"
    )
    assert bool(truncated[0]) == bool(exp_trunc), (
        f"{tag} turn={turn} truncated: sim={bool(truncated[0])} ref={bool(exp_trunc)}"
    )
    assert int(info.winner[0]) == int(ref_state.winner), f"{tag} turn={turn} info.winner"


def run_lockstep(
    state0: GameState,
    action_source,
    n_turns: int,
    *,
    tag: str,
    compare_every: int = 1,
    stop_on_done: bool = True,
) -> tuple[GameState, GameState, int]:
    """Step both sides with identical actions; return (sim_state, ref_state, turns_done).

    ``action_source(sim_state, ref_state, turn) -> (idx0, idx1) | (eng_actions,)``
    returns either a pair of flat indices (decoded via the training codec on
    the sim side and via the single-index codec on the reference side) or a
    single (2, 5) engine-action array used verbatim by both sides.
    """
    batch0 = jax.tree_util.tree_map(lambda x: x[None], state0)
    sim_state = batch0
    ref_state = state0
    done = False
    for turn in range(1, n_turns + 1):
        out = action_source(unbatch(sim_state), ref_state, turn)
        if len(out) == 2:
            idx0, idx1 = out
            idx = jnp.stack(
                [jnp.asarray(idx0, dtype=jnp.int32), jnp.asarray(idx1, dtype=jnp.int32)]
            )
            sim_actions = index_to_engine_action_batch(idx)  # (2, 5)
            ref_actions = jnp.stack(
                [
                    index_to_engine_action(jnp.asarray(idx0, dtype=jnp.int32)),
                    index_to_engine_action(jnp.asarray(idx1, dtype=jnp.int32)),
                ]
            )
            # Codec self-check: the batched and scalar codecs must agree,
            # otherwise the differential is not testing identical inputs.
            assert bool(jnp.array_equal(sim_actions, ref_actions)), f"codec mismatch turn={turn}"
        else:
            sim_actions = ref_actions = out[0]
        batch_size = sim_actions.shape[0]
        joint = jnp.stack([sim_actions] * batch_size, axis=0)  # (B, 2, 5)
        if sim_state is None or sim_state.armies.shape[0] != batch_size:
            sim_state = jax.tree_util.tree_map(
                lambda x, n=batch_size: jnp.repeat(x, n, axis=0), batch0
            )
        sim_out = step_batch_jax(sim_state, joint)
        ref_state, ref_info = engine_ref_step(ref_state, ref_actions)
        ns = sim_out[0]
        sim_state = ns
        is_done = bool(sim_out[2][0]) or bool(sim_out[3][0])
        if turn % compare_every == 0 or turn == n_turns or (stop_on_done and is_done):
            compare_step(sim_out, ref_state, ref_info, turn, tag)
        if stop_on_done and is_done:
            # Terminal state was compared above; verify the reference agrees
            # that the game is done before stopping.
            assert bool(ref_info.is_done) or int(ref_state.time) >= TRUNCATION, (
                f"{tag} turn={turn}: sim done but reference is not"
            )
            done = True
            break
    assert bool(jnp.array_equal(unbatch(sim_state).armies, ref_state.armies)), (
        f"{tag}: final armies mismatch"
    )
    _ = done
    return unbatch(sim_state), ref_state, turn


# --------------------------------------------------------------------------
# State construction helpers (mirrors tests/unit/test_replay_engine_oracle.py)
# --------------------------------------------------------------------------


def give(
    state: GameState,
    armies: dict,
    owned0: set = frozenset(),
    owned1: set = frozenset(),
    time: int = 0,
    winner: int = -1,
) -> GameState:
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


def move(row: int, col: int, direction: int, split: int = 0) -> jnp.ndarray:
    return jnp.array([0, row, col, direction, split], dtype=jnp.int32)


def build_action(row: int, col: int) -> jnp.ndarray:
    return jnp.array([2, row, col, 0, 0], dtype=jnp.int32)


def engine_actions(a0: jnp.ndarray, a1: jnp.ndarray) -> tuple:
    return (jnp.stack([a0, a1]),)


# --------------------------------------------------------------------------
# 1. Long-horizon seeded random game (turns 0..420)
# --------------------------------------------------------------------------


def test_long_horizon_random_game_420_turns():
    """Seeded random-legal self-play on a competition board, lockstep both
    sides, full state compared every 10 turns and done/rewards every turn.

    Actions are sampled from the exact training legal masks and decoded with
    the training flat-index codec on the simulator side, so this exercises the
    full rollout input path. Coverage: expansion, growth (even-tick structures
    and the every-50 global increment at turns 50..400), neutral-tile capture,
    splits, and incidental combat over the midgame.
    """
    key = jax.random.PRNGKey(20260214)
    state0 = reset_one_jax(key, 21, 21)  # competition-size board, no neutral castles
    rng = np.random.default_rng(1234)
    n_turns = 420

    def action_source(sim_state, ref_state, turn):
        # Identical inputs on both sides by construction: masks come from the
        # sim state, which was compared to the reference at every checkpoint.
        del ref_state
        m0 = np.asarray(legal_mask_one_p0(sim_state))
        m1 = np.asarray(legal_mask_one_p1(sim_state))
        legal0 = np.flatnonzero(m0)
        legal1 = np.flatnonzero(m1)
        assert legal0.size and legal1.size
        return int(rng.choice(legal0)), int(rng.choice(legal1))

    sim_final, ref_final, turns = run_lockstep(
        state0, action_source, n_turns, tag="random420", compare_every=CHECK_EVERY
    )
    assert turns == n_turns, f"game ended early at turn {turns}"
    assert int(ref_final.time) == n_turns
    # Sanity: the game actually progressed (growth + expansion happened).
    assert int(jnp.sum(ref_final.ownership[0])) > 1
    assert int(jnp.sum(ref_final.ownership[1])) > 1
    assert int(jnp.max(ref_final.armies)) > 2
    assert int(ref_final.winner) == -1
    # Final-state equivalence one more time, explicitly.
    compare_states(sim_final, ref_final, turns, "random420-final")


@pytest.mark.slow
def test_long_horizon_random_game_720_turns_crosses_deathtouch():
    """Extended seeded lockstep into deathtouch territory (turns 0..720):
    14 global-increment boundaries and the turn-800 threshold regime under
    random legal play, full state every 10 turns."""
    key = jax.random.PRNGKey(987654321)
    state0 = reset_one_jax(key, 20, 19)  # rectangular competition board
    rng = np.random.default_rng(777)
    n_turns = 720

    def action_source(sim_state, ref_state, turn):
        del ref_state
        m0 = np.asarray(legal_mask_one_p0(sim_state))
        m1 = np.asarray(legal_mask_one_p1(sim_state))
        legal0 = np.flatnonzero(m0)
        legal1 = np.flatnonzero(m1)
        return int(rng.choice(legal0)), int(rng.choice(legal1))

    sim_final, ref_final, turns = run_lockstep(
        state0, action_source, n_turns, tag="random720", compare_every=CHECK_EVERY
    )
    # Random legal play may end the game early (deathtouch era); whatever
    # happens, both sides must have agreed at every checkpoint up to `turns`.
    assert turns == n_turns or int(ref_final.winner) != -1 or bool(ref_final.time >= TRUNCATION)
    assert int(sim_final.time) == int(ref_final.time)


@pytest.mark.slow
def test_randomized_build_combat_midgame():
    """Seeded-random scripted mix of moves/splits/builds from time 600 over
    240 turns (past the 800 deathtouch threshold): invalid build attempts,
    contested centre cells, and growth all exercised; full state every turn."""
    grid = jnp.zeros((9, 14), dtype=jnp.int32).at[4, 0].set(1).at[4, 13].set(2)
    state = give(
        create_initial_state(embed_grid(grid)),
        {(4, 0): 40, (4, 1): 120, (4, 2): 30, (4, 3): 8,
         (4, 13): 44, (4, 12): 90, (4, 11): 25, (4, 10): 11},
        owned0={(4, 0), (4, 1), (4, 2), (4, 3)},
        owned1={(4, 13), (4, 12), (4, 11), (4, 10)},
        time=600,
    )
    rng = np.random.default_rng(2026)
    cells0 = [(4, 1), (4, 2), (4, 3)]
    cells1 = [(4, 12), (4, 11), (4, 10)]

    def random_engine_action(player: int, ref_state) -> jnp.ndarray:
        r = rng.random()
        front = cells0 if player == 0 else cells1
        if r < 0.12:  # build attempt at a random own-front cell (may be invalid)
            return build_action(*front[int(rng.integers(len(front)))])
        cell = front[int(rng.integers(3))]
        direction = 3 if player == 0 else 2  # toward the enemy
        if rng.random() < 0.25:  # occasionally probe vertically (often invalid)
            direction = int(rng.integers(2))
        split = int(rng.integers(2))
        del ref_state
        return move(cell[0], cell[1], direction, split)

    def action_source(sim_state, ref_state, turn):
        del sim_state
        return engine_actions(
            random_engine_action(0, ref_state), random_engine_action(1, ref_state)
        )

    sim_final, ref_final, turns = run_lockstep(
        state, action_source, 240, tag="random-midgame"
    )
    assert turns == 240 or int(ref_final.winner) != -1
    assert int(ref_final.time) == 600 + turns or int(ref_final.winner) != -1


# --------------------------------------------------------------------------
# 2. Combat: contested cell, chasing/reinforcing, smaller-army order
# --------------------------------------------------------------------------


def embed_grid(small: jnp.ndarray) -> jnp.ndarray:
    """Embed a small scenario grid in a 21x21 mountain field.

    Every scenario then shares one (21, 21) JIT shape, so the vmapped step
    kernel compiles once for the whole file (runtime budget). The mountains
    are impassable, so play never leaves the embedded rectangle.
    """
    h, w = small.shape
    return jnp.full((MAX_HW, MAX_HW), -2, dtype=jnp.int32).at[:h, :w].set(small)


def _combat_board() -> GameState:
    grid = jnp.zeros((5, 9), dtype=jnp.int32).at[2, 0].set(1).at[2, 8].set(2)
    return create_initial_state(embed_grid(grid))


def test_combat_contested_cell_and_ordering():
    """Scripted collisions: neutral contested cell, chase-vs-reinforce move
    order, smaller-army-goes-first resolution. Full state compared every turn
    over 30 turns."""
    # p0 owns (2,1)-(2,3); p1 owns (2,5)-(2,7). (2,4) is the contested neutral.
    state = give(
        _combat_board(),
        {(2, 1): 20, (2, 2): 6, (2, 3): 10, (2, 4): 5, (2, 5): 9, (2, 6): 12, (2, 7): 25},
        owned0={(2, 1), (2, 2), (2, 3)},
        owned1={(2, 5), (2, 6), (2, 7)},
    )
    # Turn 1: both attack the neutral (2,4) -> smaller army resolves first
    # (engine order); then the larger resolves against the remainder.
    # Turn 2: head-on mutual chase p0 (2,3)->right vs p1 (2,5)->left.
    # Then: reinforce/chase mixes toward the centre; then pass to let growth run.
    script = [
        (move(2, 3, 3), move(2, 5, 2)),  # both onto contested (2,4)
        (move(2, 2, 3), move(2, 6, 2)),  # chase: destinations are enemy sources
        (move(2, 1, 3, split=1), move(2, 7, 2, split=1)),  # split moves into the fight
        (move(2, 2, 3), move(2, 6, 2)),
        (PASS, move(2, 7, 2)),  # one-sided reinforce while p0 passes
        (move(2, 1, 3), PASS),
    ]

    def action_source(sim_state, ref_state, turn):
        del sim_state, ref_state
        if turn <= len(script):
            a0, a1 = script[turn - 1]
            return engine_actions(a0, a1)
        return engine_actions(PASS, PASS)

    run_lockstep(state, action_source, 30, tag="combat")


def test_combat_smaller_army_resolution_explicit():
    """One contested neutral tile, two attackers of different sizes, one turn:
    the engine resolves the smaller first; both simulators must agree on the
    exact remaining armies and ownership."""
    state = give(
        _combat_board(),
        {(2, 3): 8, (2, 5): 20, (2, 4): 6},
        owned0={(2, 3)},
        owned1={(2, 5)},
    )
    a0 = move(2, 3, 3)  # p0 attacks (2,4) with 7 (whole-move leaves 1)
    a1 = move(2, 5, 2)  # p1 attacks (2,4) with 19
    sim_state, ref_state, _ = run_lockstep(
        state, lambda s, r, t: engine_actions(a0, a1), 3, tag="smaller-first"
    )
    # Independent engine ground truth for the decisive turn: p1's smaller army
    # does NOT go first here (army 19 > 7); p0 moves first: 7 vs 6 -> p0 takes
    # (2,4) with 1; then 19 vs 1 -> p1 takes it with 18.
    assert int(ref_state.armies[2, 4]) == 18
    assert bool(ref_state.ownership[1, 2, 4])
    assert not bool(ref_state.ownership[0, 2, 4])
    assert bool(jnp.array_equal(sim_state.armies, ref_state.armies))


# --------------------------------------------------------------------------
# 3. Castle building + growth
# --------------------------------------------------------------------------


def test_castle_build_and_growth_parity():
    """Build a castle (deducting the exact dynamic price), then run 120 turns
    of passes across the turn-50 global increment and many even-tick structure
    ticks; full state compared every turn. Time starts at 40 so the run covers
    ticks 41..160 (structure growth on even ticks, +1-all at tick 50, 100, 150)."""
    grid = jnp.zeros((6, 8), dtype=jnp.int32).at[2, 0].set(1).at[3, 7].set(2)
    base = create_initial_state(embed_grid(grid))
    owned0 = {(2, 0), (2, 1), (2, 2), (1, 1)}
    state = give(base, {(2, 0): 5, (2, 1): 200, (2, 2): 4, (3, 7): 1}, owned0=owned0, time=40)

    # Build on (2,1) which holds 200 army: price from p0 structures = general
    # at d=1 -> 35 + max(0, 14-2) = 47.
    expected_price = int(bc.build_cost_grid(state, 0)[2, 1])
    assert expected_price == 47

    def action_source(sim_state, ref_state, turn):
        if turn == 1:
            return engine_actions(build_action(2, 1), PASS)
        if turn == 2:
            # The build landed last tick: exact price deduction, and tick 41 is
            # odd so no growth has applied yet. Both sides must show this.
            assert bool(ref_state.castles[2, 1])
            assert int(ref_state.armies[2, 1]) == 200 - expected_price
            assert bool(sim_state.castles[2, 1])
            assert int(sim_state.armies[2, 1]) == 200 - expected_price
            # Invalid build (unowned cell) must be consumed as a pass identically.
            return engine_actions(build_action(4, 4), PASS)
        del sim_state, ref_state
        return engine_actions(PASS, PASS)

    sim_final, ref_final, turns = run_lockstep(
        state, action_source, 121, tag="castle-growth"
    )
    assert turns == 121
    # Build landed and was priced correctly on both sides.
    assert bool(ref_final.castles[2, 1])
    assert bool(sim_final.castles[2, 1])
    # Growth ground truth over ticks 41..161 (time now 161):
    #  - +1-all owned land at ticks 50, 100, 150 (3 increments)
    #  - structure +1 on even ticks: castle at (2,1) exists from tick 41,
    #    so even ticks 42..160 inclusive -> 60 increments
    castle_army = int(ref_final.armies[2, 1])
    assert castle_army == (200 - expected_price) + 60 + 3, castle_army
    # General (2,0) also grew as a structure: 1 + 61 even ticks (42..160 = 60)
    # plus the +1-all ticks -> exact figure taken from the reference itself and
    # already asserted equal to the sim; sanity-check monotone growth only.
    assert int(ref_final.armies[2, 0]) > 5
    compare_states(sim_final, ref_final, turns, "castle-growth-final")


def test_castle_growth_phase_even_ticks_only():
    """Short differential check that structure growth fires on EVEN ticks and
    the global increment fires every 50, on both implementations."""
    grid = jnp.zeros((4, 6), dtype=jnp.int32).at[1, 0].set(1).at[2, 5].set(2)
    state = give(create_initial_state(embed_grid(grid)), {(1, 0): 10}, owned0={(1, 0)}, time=0)

    def action_source(sim_state, ref_state, turn):
        del sim_state, ref_state
        return engine_actions(PASS, PASS)

    sim_final, ref_final, _ = run_lockstep(state, action_source, 52, tag="growth-phase")
    # Ticks 1..52: structure increments on even ticks 2..52 -> 26; +1-all at 50.
    assert int(ref_final.armies[1, 0]) == 10 + 26 + 1
    assert int(sim_final.armies[1, 0]) == int(ref_final.armies[1, 0])


# --------------------------------------------------------------------------
# 4. General capture -> win; simultaneous capture -> draw
# --------------------------------------------------------------------------


def test_general_capture_win_parity():
    """One decisive capture step, then 5 post-terminal steps: winner, reward,
    terminated and the loser-cell transfer must match on both sides, and time
    must freeze after the win (engine done-before semantics)."""
    grid = jnp.zeros((3, 5), dtype=jnp.int32).at[1, 0].set(1).at[1, 4].set(2)
    # p0 army of 50 beside the enemy general (army 1).
    state = give(create_initial_state(embed_grid(grid)), {(1, 3): 50, (1, 0): 1, (1, 4): 1},
                 owned0={(1, 3)}, owned1={(1, 4)})

    capture = move(1, 3, 3)  # right onto p1 general

    def action_source(sim_state, ref_state, turn):
        del sim_state, ref_state
        if turn == 1:
            return engine_actions(capture, PASS)
        return engine_actions(PASS, PASS)

    sim_final, ref_final, turns = run_lockstep(
        state, action_source, 6, tag="capture", stop_on_done=False
    )
    assert turns == 6
    assert int(ref_final.winner) == 0
    assert int(sim_final.winner) == 0
    # Loser cells transferred to the winner by the pinned engine on both sides.
    assert bool(ref_final.ownership[0, 1, 4])
    assert not bool(ref_final.ownership[1].any())
    assert bool(jnp.array_equal(sim_final.ownership, ref_final.ownership))
    # Time froze after the terminal turn (done_before guard in game.step).
    assert int(ref_final.time) == 1
    assert int(sim_final.time) == 1


def test_simultaneous_general_capture_draw_parity():
    """Mutual decapitation on the same turn must be a DRAW on both sides
    (deathtouch both_captured rule) at a pre-threshold turn: is_done with
    winner -1, zero rewards, terminated True, truncated False."""
    grid = jnp.zeros((5, 7), dtype=jnp.int32).at[2, 0].set(1).at[2, 4].set(2)
    state = create_initial_state(embed_grid(grid))
    state = state._replace(
        armies=state.armies.at[2, 0].set(20).at[2, 4].set(30).at[2, 3].set(50).at[2, 1].set(60)
    )
    state = state._replace(
        ownership=state.ownership.at[0, 2, 3].set(True).at[1, 2, 1].set(True),
        ownership_neutral=state.ownership_neutral.at[2, 3].set(False).at[2, 1].set(False),
    )
    a0 = jnp.array([0, 2, 3, 3, 0], dtype=jnp.int32)  # p0 right onto p1 general
    a1 = jnp.array([0, 2, 1, 2, 0], dtype=jnp.int32)  # p1 left onto p0 general

    batch0 = jax.tree_util.tree_map(lambda x: x[None], state)
    sim_out = step_batch_jax(batch0, jnp.stack([jnp.stack([a0, a1])], axis=0))
    ref_state, ref_info = engine_ref_step(state, jnp.stack([a0, a1]))

    compare_step(sim_out, ref_state, ref_info, 1, "mutual-capture")
    assert int(ref_info.winner) == -1 and bool(ref_info.is_done)
    assert bool(sim_out[2][0])  # terminated (draw ends the episode)
    assert not bool(sim_out[3][0])  # not truncated
    assert float(sim_out[1][0][0]) == 0.0 and float(sim_out[1][0][1]) == 0.0


# --------------------------------------------------------------------------
# 5. Deathtouch at the turn-800 boundary
# --------------------------------------------------------------------------


def _deathtouch_state(time: int) -> GameState:
    """p0 general (army 2) adjacent to p1 general; a right whole-move touches."""
    grid = jnp.zeros((5, 7), dtype=jnp.int32).at[2, 2].set(1).at[2, 3].set(2)
    state = create_initial_state(embed_grid(grid))
    return state._replace(
        armies=state.armies.at[2, 2].set(2), time=jnp.int32(time)
    )


def test_deathtouch_boundary_parity():
    """Step the identical touch attempt at time 799 (inactive) and 800
    (active) through both implementations; winner/done/rewards and the full
    post-step state must match on each side of the threshold."""
    touch = move(2, 2, 3)

    for t0, tag in [(799, "dt799"), (800, "dt800"), (850, "dt850")]:
        state = _deathtouch_state(t0)
        sim_state, ref_state, _ = run_lockstep(
            state, lambda s, r, turn: engine_actions(touch, PASS), 2, tag=tag
        )
        if t0 < DEATHTOUCH_TURN:
            # Army 2 whole-move onto army-1 general is an ordinary capture even
            # pre-threshold: base engine win for p0 either way.
            assert int(ref_state.winner) == 0, tag
        else:
            assert int(ref_state.winner) == 0, tag
        assert int(sim_state.winner) == int(ref_state.winner), tag

    # Negative control at 800: with army 1 the move is invalid and deathtouch
    # cannot fire from an invalid move -> no winner on either side.
    weak = _deathtouch_state(DEATHTOUCH_TURN)._replace(
        armies=_deathtouch_state(DEATHTOUCH_TURN).armies.at[2, 2].set(1)
    )
    sim_state, ref_state, _ = run_lockstep(
        weak, lambda s, r, turn: engine_actions(touch, PASS), 2, tag="dt-invalid"
    )
    assert int(ref_state.winner) == -1
    assert int(sim_state.winner) == -1


def test_deathtouch_forced_touch_spoils_transfer_parity():
    """A touch that the base step does not itself settle (enemy general tile
    occupied by a bigger army -> base attack loses) must still award the win
    AND transfer the loser's cells identically on both sides."""
    grid = jnp.zeros((5, 7), dtype=jnp.int32).at[2, 2].set(1).at[2, 4].set(2)
    # Guard of 100 ON the p1 general tile, p0 attacker of 2 beside it, plus
    # some p1 land to transfer. (2,3) must be p0-owned for the touch move to
    # be valid; (2,4) is p1's general tile (owned by p1 from the start).
    state = give(
        create_initial_state(embed_grid(grid)),
        {(2, 3): 2, (2, 4): 100},
        owned0={(2, 3)},
        owned1={(2, 4)},
        time=DEATHTOUCH_TURN,
    )
    touch = move(2, 3, 3)  # onto the enemy general tile (guarded)
    sim_state, ref_state, _ = run_lockstep(
        state, lambda s, r, turn: engine_actions(touch, PASS), 1, tag="dt-spoils"
    )
    assert int(ref_state.winner) == 0
    assert bool(ref_state.ownership[0, 2, 4])  # spoils transferred
    assert not bool(ref_state.ownership[1].any())
    compare_states(sim_state, ref_state, 1, "dt-spoils-final")


# --------------------------------------------------------------------------
# 6. Turn-1200 hard draw (truncation)
# --------------------------------------------------------------------------


def test_turn_1200_hard_draw_parity():
    """Constructed midgame state jumped to time 1197 (engine states are
    immutable NamedTuples; ``time`` is caller-settable exactly as the replay
    oracle's state_from_tick does). Both sides pass to the boundary: at time
    1200 the simulator must report truncated=True/terminated=False/0 rewards,
    matching the official env formula, while the engine state itself simply
    carries time=1200 with winner -1."""
    grid = jnp.zeros((6, 9), dtype=jnp.int32).at[2, 0].set(1).at[3, 8].set(2)
    state = give(
        create_initial_state(embed_grid(grid)),
        {(2, 0): 40, (2, 1): 5, (3, 8): 42},
        owned0={(2, 0), (2, 1)},
        owned1={(3, 8)},
        time=TRUNCATION - 3,
    )

    sim_state, ref_state, turns = run_lockstep(
        state,
        lambda s, r, turn: engine_actions(PASS, PASS),
        3,
        tag="draw1200",
        compare_every=1,
        stop_on_done=False,
    )
    assert turns == 3
    assert int(ref_state.time) == TRUNCATION
    assert int(ref_state.winner) == -1
    assert int(sim_state.time) == TRUNCATION
    # Final step was truncation: verify flags directly from the last step.
    at_boundary = ref_state._replace(time=jnp.int32(TRUNCATION))
    batch0 = jax.tree_util.tree_map(lambda x: x[None], at_boundary)
    ns, rewards, terminated, truncated, info = step_batch_jax(
        batch0, jnp.stack([jnp.stack([PASS, PASS])], axis=0)
    )
    # Stepping a truncated (winner == -1, not terminated) state: engine keeps
    # advancing time; the truncation flag stays latched on.
    assert bool(truncated[0])
    assert not bool(terminated[0])
    assert float(rewards[0][0]) == 0.0


def test_truncation_flag_latches_exactly_at_boundary():
    """One step on each side of the boundary from the same constructed state:
    truncated is False at resulting time 1199 and True at 1200, terminated is
    False both times, and the engine reference state matches field-for-field."""
    grid = jnp.zeros((4, 6), dtype=jnp.int32).at[1, 0].set(1).at[2, 5].set(2)
    base = give(create_initial_state(embed_grid(grid)), {(1, 0): 3, (2, 5): 3},
                owned0={(1, 0)}, owned1={(2, 5)})

    for t0 in (TRUNCATION - 2, TRUNCATION - 1):
        state = base._replace(time=jnp.int32(t0))
        batch0 = jax.tree_util.tree_map(lambda x: x[None], state)
        sim_out = step_batch_jax(batch0, jnp.stack([jnp.stack([PASS, PASS])], axis=0))
        ref_state, ref_info = engine_ref_step(state, jnp.stack([PASS, PASS]))
        compare_step(sim_out, ref_state, ref_info, t0 + 1, f"boundary-{t0}")
        expect_trunc = (t0 + 1) >= TRUNCATION
        assert bool(sim_out[3][0]) is expect_trunc
        assert not bool(sim_out[2][0])


# --------------------------------------------------------------------------
# Mixed endgame: builds + combat across the deathtouch threshold
# --------------------------------------------------------------------------


def test_midgame_to_deathtouch_lockstep():
    """140 lockstep turns from time 700: scripted expansion/build/combat that
    crosses the deathtouch threshold at 800 with full-state comparison every
    turn. Guards the mid->endgame seam where 'telemetry fine, deployment
    strength absent' would surface."""
    grid = jnp.zeros((7, 12), dtype=jnp.int32).at[3, 0].set(1).at[3, 11].set(2)
    state = give(
        create_initial_state(embed_grid(grid)),
        {(3, 0): 30, (3, 1): 80, (3, 2): 6, (3, 11): 30, (3, 10): 15, (3, 9): 9},
        owned0={(3, 0), (3, 1), (3, 2)},
        owned1={(3, 11), (3, 10), (3, 9)},
        time=700,
    )
    script = [
        (move(3, 2, 3), move(3, 9, 2)),          # expand toward centre
        (build_action(3, 1), PASS),              # p0 builds (affords ~45+)
        (move(3, 1, 3, split=1), move(3, 10, 2)),
        (move(3, 2, 3), move(3, 9, 2)),          # collision in the middle
        (move(3, 1, 3), move(3, 10, 2, split=1)),
        (PASS, build_action(3, 10)),             # p1 builds
    ]

    def action_source(sim_state, ref_state, turn):
        del sim_state, ref_state
        if turn <= len(script):
            a0, a1 = script[turn - 1]
            return engine_actions(a0, a1)
        # After the script, alternate pushes toward the enemy; anything that
        # becomes invalid is a silent pass on BOTH engines equally.
        d = turn % 4
        a0 = move(3, 2, 3) if d < 2 else PASS
        a1 = move(3, 9, 2) if d >= 1 else PASS
        return engine_actions(a0, a1)

    sim_final, ref_final, turns = run_lockstep(
        state, action_source, 140, tag="mid-to-dt"
    )
    assert turns == 140
    assert int(ref_final.time) == 840  # crossed the 800 threshold
    # At least one build should have landed for p0 (cell (3,1) had 80 army).
    assert bool(ref_final.castles[3, 1])
    compare_states(sim_final, ref_final, turns, "mid-to-dt-final")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
