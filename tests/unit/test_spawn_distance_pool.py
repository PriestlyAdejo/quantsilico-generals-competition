"""Spawn-distance curriculum knob guards (PPO_SEMANTICS UNCHANGED).

The min_generals_distance parameter is additive and defaults to the
competition value (17). These tests guard:
1. default-call determinism: same seed -> identical pool leaves;
2. knob effect: a different distance actually changes generated boards;
3. default equivalence: passing 17 explicitly matches the implicit default.
"""

from __future__ import annotations

import jax
import numpy as np

from generals_bot.competition_native_jax.competition_env_jax import (
    build_competition_reset_pool,
)

POOL_SIZE = 16  # one board per size combo -> fast on CPU


def leaves(pool):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(pool)]


def pools_equal(a, b) -> bool:
    la, lb = leaves(a), leaves(b)
    return len(la) == len(lb) and all(np.array_equal(x, y) for x, y in zip(la, lb, strict=True))


def test_default_pool_deterministic_and_explicit_default_equivalent():
    pool_a = build_competition_reset_pool(jax.random.PRNGKey(7), POOL_SIZE)
    pool_b = build_competition_reset_pool(jax.random.PRNGKey(7), POOL_SIZE)
    assert pools_equal(pool_a, pool_b)
    pool_explicit = build_competition_reset_pool(
        jax.random.PRNGKey(7), POOL_SIZE, min_generals_distance=17
    )
    assert pools_equal(pool_a, pool_explicit)


def test_distance_knob_changes_boards():
    default_pool = build_competition_reset_pool(jax.random.PRNGKey(7), POOL_SIZE)
    close_pool = build_competition_reset_pool(
        jax.random.PRNGKey(7), POOL_SIZE, min_generals_distance=8
    )
    assert not pools_equal(default_pool, close_pool)
