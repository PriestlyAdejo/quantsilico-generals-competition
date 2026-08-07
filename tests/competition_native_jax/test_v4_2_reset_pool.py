"""Tests for V4.2 reset pool and static geometry."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from generals_bot.competition_native_jax.competition_env_jax import (
    auto_reset_from_pool,
    build_competition_reset_pool,
    index_to_engine_action,
    reset_one_jax,
)
from generals_bot.competition_native_jax.static_geometry_jax import (
    get_static_geometry,
    index_to_engine_action_static,
)
from generals_bot.competition_native_jax.transformer_jax import init_params
from train.competition_native_jax.rollout_selfplay_jax import collect_selfplay_batch


def test_reset_pool_entry_matches_reset_one() -> None:
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    a = reset_one_jax(k1, height=21, width=21)
    b = reset_one_jax(k1, height=21, width=21)
    assert jnp.array_equal(a.armies, b.armies)
    assert jnp.array_equal(a.ownership, b.ownership)
    assert int(a.time) == int(b.time)
    pool = build_competition_reset_pool(k2, pool_size=32, min_grid=21, max_grid=21)
    assert jax.tree_util.tree_leaves(pool)[0].shape[0] == 32


def test_auto_reset_from_pool_advances_cursor() -> None:
    key = jax.random.PRNGKey(3)
    pool = build_competition_reset_pool(key, pool_size=32, min_grid=21, max_grid=21)
    n = 4
    states = jax.tree_util.tree_map(lambda x: x[:n], pool)
    cursor = jnp.arange(n, dtype=jnp.int32) + n
    terminated = jnp.array([True, False, True, False])
    truncated = jnp.zeros((n,), dtype=bool)
    new_states, new_cursor = auto_reset_from_pool(states, terminated, truncated, pool, cursor)
    assert int(new_cursor[0]) == int(cursor[0]) + 1
    assert int(new_cursor[1]) == int(cursor[1])
    assert int(new_cursor[2]) == int(cursor[2]) + 1
    # Done envs should equal pool[cursor % size]
    ps = int(jax.tree_util.tree_leaves(pool)[0].shape[0])
    idx0 = int(cursor[0]) % ps
    assert jnp.array_equal(new_states.armies[0], pool.armies[idx0])


def test_static_geometry_action_codec_parity() -> None:
    # Legacy inline formula vs static geometry for a dense sample of indices
    def legacy(idx):
        is_pass = idx == 0
        flat = idx - 1
        cell = flat // 9
        local = flat % 9
        row = cell // 21
        col = cell % 21
        is_build = local == 8
        direction = local // 2
        split = local % 2
        kind = jnp.where(is_pass, 1, jnp.where(is_build, 2, 0))
        return jnp.stack(
            [
                kind.astype(jnp.int32),
                jnp.where(is_pass, 0, row).astype(jnp.int32),
                jnp.where(is_pass, 0, col).astype(jnp.int32),
                jnp.where(is_pass | is_build, 0, direction).astype(jnp.int32),
                jnp.where(is_pass | is_build, 0, split).astype(jnp.int32),
            ]
        )

    for idx in [0, 1, 9, 10, 17, 100, 500, 1000, 2000, 3969]:
        a = index_to_engine_action(jnp.array(idx, dtype=jnp.int32))
        b = index_to_engine_action_static(jnp.array(idx, dtype=jnp.int32))
        c = legacy(jnp.array(idx, dtype=jnp.int32))
        assert jnp.array_equal(a, b)
        assert jnp.array_equal(a, c)


def test_static_geometry_shapes() -> None:
    g = get_static_geometry()
    assert g.manhattan.shape == (441, 441)
    assert g.action_source_cell.shape == (3970,)
    assert g.playable.shape == (4, 21, 21)


def test_rollout_carry_continues_episode_across_ppo_updates() -> None:
    """A second rollout must not silently restart every game at turn zero."""
    pool = build_competition_reset_pool(
        jax.random.PRNGKey(41), pool_size=2, min_grid=18, max_grid=18
    )
    params = init_params(jax.random.PRNGKey(42))
    _batch1, carry1 = collect_selfplay_batch(
        params,
        num_envs=1,
        rollout_len=2,
        pool=pool,
        seed=43,
        return_carry=True,
    )
    _batch2, carry2 = collect_selfplay_batch(
        params,
        num_envs=1,
        rollout_len=2,
        pool=pool,
        seed=44,
        carry=carry1,
        return_carry=True,
    )
    assert int(carry1.states.time[0]) == 2
    assert int(carry2.states.time[0]) == 4
