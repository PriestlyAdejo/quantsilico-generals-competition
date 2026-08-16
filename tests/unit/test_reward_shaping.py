"""Reward shaping knob tests (REWARD-SHAPING-R1, EV-0044)."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from train.competition_native_jax.reward_shaping_jax import (
    active_shaping,
    set_active_shaping,
    shape_step_rewards,
)


def s(own_army, opp_army):
    """Batched fake GameState-shaped object (B envs, 1x2 boards).

    Cell (0,0) belongs to the trained seat with `own_army`; cell (0,1) to the
    opponent with `opp_army` - list values are per-env.
    """
    armies = jnp.stack(
        [jnp.stack([jnp.asarray(own_army), jnp.asarray(opp_army)], axis=-1)], axis=1
    ).astype(jnp.float32)  # (B, 1, 2)
    batch = armies.shape[0]
    own = jnp.zeros((batch, 2, 1, 2), dtype=bool)  # real batched layout (B, 2, H, W)
    own = own.at[:, 0, 0, 0].set(True)
    own = own.at[:, 1, 0, 1].set(True)
    return SimpleNamespace(armies=armies, ownership=own)


B = 2
ZEROS = jnp.zeros(B)
ONES = jnp.ones(B)


def test_none_mode_is_identity():
    states = s([5, 6], [3, 4])
    out = shape_step_rewards(states, states, ONES, ZEROS, ZEROS, "none", 0.5)
    assert bool(jnp.array_equal(out, ONES))


def test_zero_beta_is_identity():
    states = s([5, 6], [3, 4])
    out = shape_step_rewards(states, states, ONES, ZEROS, ZEROS, "kill_delta", 0.0)
    assert bool(jnp.array_equal(out, ONES))


def test_kill_delta_rewards_enemy_loss_and_penalises_own_loss():
    states = s([10, 10], [8, 8])
    next_enemy_loss = s([10, 10], [5, 5])  # opponent lost 3 -> +3*beta
    out = shape_step_rewards(states, next_enemy_loss, ZEROS, ZEROS, ZEROS, "kill_delta", 0.1)
    assert bool(jnp.allclose(out, jnp.full(B, 0.3), atol=1e-5))
    next_own_loss = s([7, 7], [8, 8])  # own lost 3 -> -3*beta
    out2 = shape_step_rewards(states, next_own_loss, ZEROS, ZEROS, ZEROS, "kill_delta", 0.1)
    assert bool(jnp.allclose(out2, jnp.full(B, -0.3), atol=1e-5))


def test_alive_mask_keeps_terminal_engine_reward():
    states = s([10, 10], [8, 8])
    next_enemy_loss = s([10, 10], [0, 0])
    terminal_reward = jnp.array([1.0, 1.0])
    out = shape_step_rewards(
        states, next_enemy_loss, terminal_reward, ONES, ZEROS, "kill_delta", 0.5
    )
    # terminated ticks keep the engine reward exactly, no shaped increment
    assert bool(jnp.array_equal(out, terminal_reward))
    out_trunc = shape_step_rewards(
        states, next_enemy_loss, terminal_reward, ZEROS, ONES, "kill_delta", 0.5
    )
    assert bool(jnp.array_equal(out_trunc, terminal_reward))


def test_potential_mode_increment_sign():
    states = s([10, 10], [10, 10])  # Phi = 0
    stronger = s([20, 20], [10, 10])  # Phi > 0 -> positive increment
    out = shape_step_rewards(states, stronger, ZEROS, ZEROS, ZEROS, "potential", 1.0)
    assert bool(jnp.all(out > 0))
    weaker = s([5, 5], [10, 10])  # Phi < 0 -> negative increment
    out2 = shape_step_rewards(states, weaker, ZEROS, ZEROS, ZEROS, "potential", 1.0)
    assert bool(jnp.all(out2 < 0))


def test_potential_terminal_correction_and_return_invariance():
    # gamma=1 potential-based shaping: shaped episode return == unshaped return.
    states = s([10, 10], [10, 10])  # Phi(s0)=0
    mid = s([15, 15], [10, 10])  # Phi(s1)>0
    # two-step alive then terminal win +1 from mid
    r_alive = shape_step_rewards(states, mid, ZEROS, ZEROS, ZEROS, "potential", 1.0)
    terminal = jnp.array([1.0, 1.0])
    r_end = shape_step_rewards(mid, mid, terminal, ONES, ZEROS, "potential", 1.0)
    # unshaped return = 0 + 0 + 1 = 1
    shaped_return = float(r_alive[0]) + 0.0 + float(r_end[0])
    assert abs(shaped_return - 1.0) < 1e-4


def test_potential_zero_when_totals_unchanged():
    states = s([10, 10], [10, 10])
    same = s([10, 10], [10, 10])
    out = shape_step_rewards(states, same, ZEROS, ZEROS, ZEROS, "potential", 1.0)
    assert bool(jnp.allclose(out, ZEROS, atol=1e-6))


def test_kill_delta_bounded_on_one_step():
    states = s([1, 1], [100, 100])
    wiped = s([1, 1], [0, 0])
    out = shape_step_rewards(states, wiped, ZEROS, ZEROS, ZEROS, "kill_delta", 0.01)
    assert bool(jnp.all(jnp.abs(out) <= 1.0))


def test_land_potential_signal_on_capture_and_return_invariance():
    # land changes break the army-total symmetry: owner swap of one tile
    base = s([10, 10], [10, 10])
    own_more = SimpleNamespace(
        armies=base.armies,
        ownership=jnp.concatenate(
            [
                jnp.ones((base.armies.shape[0], 1, 1, 2), dtype=bool),
                base.ownership[:, 1:, :, :],
            ],
            axis=1,
        ),
    )
    out = shape_step_rewards(base, own_more, ZEROS, ZEROS, ZEROS, "land_potential", 1.0)
    assert bool(jnp.all(out > 0))  # own land grew -> positive increment
    # return invariance at gamma=1: alive increment + terminal correction cancel
    r_alive = shape_step_rewards(base, own_more, ZEROS, ZEROS, ZEROS, "land_potential", 1.0)
    terminal = jnp.array([1.0, 1.0])
    r_end = shape_step_rewards(own_more, own_more, terminal, ONES, ZEROS, "land_potential", 1.0)
    assert abs(float(r_alive[0]) + float(r_end[0]) - 1.0) < 1e-4


def test_land_potential_zero_when_land_unchanged():
    base = s([10, 10], [10, 10])
    out = shape_step_rewards(base, base, ZEROS, ZEROS, ZEROS, "land_potential", 1.0)
    assert bool(jnp.allclose(out, ZEROS, atol=1e-6))


def test_potential_truncation_correction_return_invariance():
    # The 1200-turn hard draw is a genuine episode end (engine reward 0):
    # the shaped return must still telescope to the unshaped return.
    base = s([10, 10], [10, 10])  # Phi(s0)=0
    own_more = SimpleNamespace(
        armies=base.armies,
        ownership=jnp.concatenate(
            [
                jnp.ones((base.armies.shape[0], 1, 1, 2), dtype=bool),
                base.ownership[:, 1:, :, :],
            ],
            axis=1,
        ),
    )  # Phi(s1) > 0
    r_alive = shape_step_rewards(base, own_more, ZEROS, ZEROS, ZEROS, "land_potential", 1.0)
    draw_reward = jnp.zeros(B)
    r_end = shape_step_rewards(
        own_more, own_more, draw_reward, ZEROS, ONES, "land_potential", 1.0
    )
    # unshaped return = 0 (draw); shaped return must equal it at gamma = 1
    assert abs(float(r_alive[0]) + float(r_end[0])) < 1e-4
    # correction at truncation is -Phi(s) (negative here), engine reward kept
    assert float(r_end[0]) < 0.0
    assert bool(jnp.all(jnp.isfinite(r_end)))


def test_invalid_mode_and_negative_beta_rejected():
    with pytest.raises(ValueError):
        set_active_shaping("bogus", 0.1)
    with pytest.raises(ValueError):
        set_active_shaping("none", -0.1)


def test_active_shaping_round_trip_and_reset():
    set_active_shaping("potential", 0.05)
    assert active_shaping() == ("potential", 0.05)
    set_active_shaping("none", 0.0)  # restore control default for other tests
    assert active_shaping() == ("none", 0.0)
