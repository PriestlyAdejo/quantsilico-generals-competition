from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from generals.agents.expander_agent import ExpanderAgent
from generals.agents.hunter_agent import HunterAgent
from generals.agents.random_agent import RandomAgent
from generals.core import game

from generals_bot.competition_native_jax.competition_env_jax import (
    build_competition_reset_pool,
    legal_mask_batch_p0,
    legal_mask_batch_p1,
)
from generals_bot.competition_native_jax.transformer_jax import init_params
from train.competition_native_jax.checkpoint_recovery import (
    persisted_carry,
    save_checkpoint,
    verify_checkpoint,
)
from train.competition_native_jax.gae_jax import gae_advantages_batch
from train.competition_native_jax.opponents_jax import (
    OpponentKind,
    batched_opponent_actions,
    build_static_schedule,
    official_to_engine_action,
)
from train.competition_native_jax.ppo_jax import ppo_loss_on_batch
from train.competition_native_jax.rollout_curriculum_jax import (
    collect_curriculum_batch,
    initialise_curriculum_carry,
)


def _pool(size: int = 8):
    return build_competition_reset_pool(jax.random.PRNGKey(17), size)


def _action_index(action: np.ndarray) -> int:
    kind, row, col, direction, split = (int(item) for item in action)
    if kind == 1:
        return 0
    assert kind == 0
    return 1 + (row * 21 + col) * 9 + direction * 2 + split


def test_all_four_opponents_match_canonical_batched_actions_and_are_legal() -> None:
    pool = _pool(16)
    states = jax.tree_util.tree_map(lambda value: value[:8], pool)
    seats = jnp.asarray([0, 1, 0, 1, 1, 0, 1, 0], dtype=jnp.int32)
    observations = jax.vmap(game.get_observation)(states, seats)
    observation = lambda i: jax.tree_util.tree_map(lambda value: value[i], observations)  # noqa: E731
    mask0 = np.asarray(legal_mask_batch_p0(states))
    mask1 = np.asarray(legal_mask_batch_p1(states))
    cases = (
        (OpponentKind.PASS, None),
        (OpponentKind.RANDOM, RandomAgent()),
        (OpponentKind.EXPANDER, ExpanderAgent()),
        (OpponentKind.HUNTER, HunterAgent()),
    )
    for offset, (kind, canonical_agent) in enumerate(cases):
        keys = jax.random.split(jax.random.PRNGKey(29 + offset), 8)
        schedule = tuple([int(kind)] * 8)
        actual = np.asarray(batched_opponent_actions(states, seats, keys, schedule))
        expected = np.stack(
            [
                np.asarray([1, 0, 0, 0, 0], dtype=np.int32)
                if canonical_agent is None
                else np.asarray(
                    official_to_engine_action(canonical_agent.act(observation(i), keys[i]))
                )
                for i in range(8)
            ]
        )
        np.testing.assert_array_equal(actual, expected)
        for i, action in enumerate(actual):
            mask = mask1[i] if int(seats[i]) else mask0[i]
            assert mask[_action_index(action)]


def test_static_curriculum_schedule_uses_exact_largest_remainder_counts() -> None:
    schedule = build_static_schedule(512, (0.4, 0.4, 0.2, 0.0))
    assert len(schedule) == 512
    assert tuple(schedule.count(kind) for kind in range(4)) == (205, 205, 102, 0)


def test_carry_continues_across_fragments_and_uses_distinct_reset_cursors() -> None:
    params = init_params(jax.random.PRNGKey(3))
    pool = _pool(8)
    schedule = (int(OpponentKind.PASS), int(OpponentKind.PASS))
    carry = initialise_curriculum_carry(
        params, num_envs=2, seed=5, reset_pool_size=8, pool=pool
    )
    np.testing.assert_array_equal(np.asarray(carry.pool_cursor), [2, 3])
    first, carry = collect_curriculum_batch(
        params, opponent_schedule=schedule, rollout_len=2, pool=pool, carry=carry
    )
    second, carry = collect_curriculum_batch(
        params, opponent_schedule=schedule, rollout_len=2, pool=pool, carry=carry
    )
    np.testing.assert_array_equal(np.asarray(first["turn"][:, 0]), [0, 1])
    np.testing.assert_array_equal(np.asarray(second["turn"][:, 0]), [2, 3])
    assert int(np.asarray(carry.states.time)[0]) == 4
    assert np.asarray(first["learner_controlled_mask"]).all()
    assert np.asarray(second["learner_controlled_mask"]).all()


def test_terminal_reward_after_fragment_reaches_gae_return_and_ppo_gradient() -> None:
    params = init_params(jax.random.PRNGKey(7))
    pool = _pool(4)
    schedule = (int(OpponentKind.PASS),)
    carry = initialise_curriculum_carry(
        params, num_envs=1, seed=11, reset_pool_size=4, pool=pool
    )
    _, carry = collect_curriculum_batch(
        params, opponent_schedule=schedule, rollout_len=2, pool=pool, carry=carry
    )
    assert int(np.asarray(carry.states.time)[0]) == 2

    # Deterministically place a terminal event after the first collector call.
    # step_one_jax observes winner=0, emits +1 for a seat-0 learner, then resets.
    terminal_states = carry.states._replace(
        time=jnp.asarray([33], dtype=jnp.int32),
        winner=jnp.asarray([0], dtype=jnp.int32),
    )
    carry = carry._replace(
        states=terminal_states,
        learner_seat=jnp.asarray([0], dtype=jnp.int32),
    )
    batch, carry = collect_curriculum_batch(
        params,
        opponent_schedule=schedule,
        rollout_len=1,
        pool=pool,
        carry=carry,
        gamma=1.0,
        shaping_lambda=0.0,
    )
    np.testing.assert_array_equal(np.asarray(batch["terminal_rewards"]), [[1.0]])
    np.testing.assert_array_equal(np.asarray(batch["dones"]), [[1.0]])
    assert int(np.asarray(carry.states.time)[0]) == 0

    values = jnp.concatenate([batch["values"], batch["bootstrap_values"][None, :]], axis=0)
    advantages, returns = gae_advantages_batch(
        batch["rewards"], values, batch["dones"], gamma=1.0, lam=0.9
    )
    np.testing.assert_allclose(np.asarray(returns), [[1.0]], atol=1e-5)
    assert abs(float(np.asarray(advantages)[0, 0])) > 1e-4

    ppo_batch = {
        "spatial": batch["spatial"].reshape((1,) + batch["spatial"].shape[2:]),
        "global": batch["global"].reshape((1,) + batch["global"].shape[2:]),
        "mask": batch["mask"].reshape(1, -1),
        "actions": batch["actions"].reshape(1),
        "old_logp": batch["old_logp"].reshape(1),
        "advantages": advantages.reshape(1),
        "returns": returns.reshape(1),
    }

    def loss_fn(candidate):
        return ppo_loss_on_batch(
            candidate,
            ppo_batch["spatial"],
            ppo_batch["global"],
            ppo_batch["mask"],
            ppo_batch["actions"],
            ppo_batch["old_logp"],
            ppo_batch["advantages"],
            ppo_batch["returns"],
        )[0]

    grads = jax.grad(loss_fn)(params)
    grad_norm = sum(float(jnp.sum(jnp.square(leaf))) for leaf in jax.tree_util.tree_leaves(grads))
    assert np.isfinite(grad_norm)
    assert grad_norm > 1e-12


def test_checkpoint_roundtrip_includes_full_rollout_carry(tmp_path) -> None:
    params = init_params(jax.random.PRNGKey(19))
    pool = _pool(4)
    carry = initialise_curriculum_carry(
        params, num_envs=1, seed=23, reset_pool_size=4, pool=pool
    )
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)
    path = save_checkpoint(
        tmp_path,
        tag="test",
        params=params,
        ema=params,
        opt_state=opt_state,
        carry=carry,
        meta={"update": 1, "transitions": 32, "programme_transitions": 32},
    )
    result = verify_checkpoint(
        path,
        params_like=params,
        opt_state_like=opt_state,
        carry_like=persisted_carry(carry),
    )
    assert result["status"] == "PASS"
    assert (path / "COMPLETE").is_file()
