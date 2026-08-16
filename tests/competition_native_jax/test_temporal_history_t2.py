"""STAGE5 T2 (legal temporal history) regression guards.

Predeclared: experiments/marathon/stage5_capacity_value_r1_plan.yaml. The
canonical 8-plane path must stay EXACT when the mode is off; k1 must present
16 planes built from LEGAL observations only, with history zeroed at
initialisation and across carried updates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from generals_bot.competition_native_jax.transformer_jax import init_params
from train.competition_native_jax.rollout_selfplay_jax import (
    collect_selfplay_batch,
    initialise_rollout_carry,
)
from train.competition_native_jax.temporal_history_jax import (
    active_temporal_history,
    set_temporal_history_mode,
)


def jax_key(seed: int):
    return jax.random.PRNGKey(seed)


@pytest.fixture(autouse=True)
def _reset_mode():
    set_temporal_history_mode("off")
    yield
    set_temporal_history_mode("off")


def test_default_mode_is_off():
    assert active_temporal_history() == "off"
    with pytest.raises(ValueError):
        set_temporal_history_mode("k2")


def test_off_mode_keeps_canonical_8_planes_and_deterministic():
    params = init_params(jax_key(1))
    batch = collect_selfplay_batch(params, num_envs=2, rollout_len=4, seed=7)
    assert batch["spatial"].shape[2:] == (8, 21, 21)
    carry = initialise_rollout_carry(params, num_envs=2, seed=7)
    assert carry.prev_sp0.shape == (2, 8, 21, 21)
    assert bool(jnp.all(carry.prev_sp0 == 0.0))
    batch2 = collect_selfplay_batch(params, num_envs=2, rollout_len=4, seed=7)
    assert bool(jnp.all(batch["rewards"] == batch2["rewards"]))
    assert bool(jnp.all(batch["actions"] == batch2["actions"]))


def test_k1_mode_presents_16_planes_with_zero_initial_history():
    set_temporal_history_mode("k1")
    params = init_params(jax_key(2), spatial_planes=16)
    assert params["patch_proj"].shape[0] == 16 * 9
    batch, carry = collect_selfplay_batch(
        params, num_envs=2, rollout_len=4, seed=11, return_carry=True
    )
    spatial = batch["spatial"]
    assert spatial.shape[2:] == (16, 21, 21)
    # first tick of a fresh carry: history planes are exactly zero
    assert bool(jnp.all(spatial[0, :, 8:, :, :] == 0.0))
    assert bool(jnp.all(jnp.isfinite(spatial)))
    assert bool(jnp.all(jnp.isfinite(batch["values"])))
    assert bool(jnp.all(jnp.isfinite(batch["bootstrap_values"])))
    # carry persists across updates with the same plane geometry
    batch2, _ = collect_selfplay_batch(
        params, num_envs=2, rollout_len=4, seed=11, carry=carry, return_carry=True
    )
    assert batch2["spatial"].shape[2:] == (16, 21, 21)
    assert bool(jnp.all(jnp.isfinite(batch2["rewards"])))


def test_k1_history_advances_after_first_tick():
    set_temporal_history_mode("k1")
    params = init_params(jax_key(3), spatial_planes=16)
    batch, _ = collect_selfplay_batch(
        params, num_envs=2, rollout_len=4, seed=13, return_carry=True
    )
    # from tick 1 onward the history channel carries the previous obs (the
    # board is non-trivial, so some history plane must be nonzero somewhere)
    assert bool(jnp.any(batch["spatial"][1:, :, 8:, :, :] != 0.0))
