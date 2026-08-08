"""Persistent learner-vs-scripted curriculum rollout on the official JAX engine."""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from generals_bot.competition_native_jax.competition_env_jax import (
    ObsMemoryJax,
    auto_reset_from_pool,
    build_competition_reset_pool,
    empty_memory,
    index_to_engine_action_batch,
    legal_mask_batch_p0,
    legal_mask_batch_p1,
    observe_batch_p0,
    observe_batch_p1,
    step_batch_jax,
)
from generals_bot.competition_native_jax.inference_jax import masked_log_softmax, sample_action
from generals_bot.competition_native_jax.transformer_jax import forward_batch
from train.competition_native_jax.opponents_jax import OpponentKind, batched_opponent_actions

sample_action_batch = jax.jit(jax.vmap(sample_action, in_axes=(0, 0, 0)))


class CurriculumCarry(NamedTuple):
    states: Any
    mem0: ObsMemoryJax
    mem1: ObsMemoryJax
    key: jax.Array
    params: dict
    pool: Any
    pool_cursor: jax.Array
    learner_seat: jax.Array
    episode_id: jax.Array
    frozen_opponent_params: dict


def _value_from_logits(value_logits: jax.Array) -> jax.Array:
    centers = jnp.linspace(-1.0, 1.0, value_logits.shape[-1])
    return jnp.sum(jax.nn.softmax(value_logits, axis=-1) * centers, axis=-1)


def initialise_curriculum_carry(
    params: dict,
    *,
    num_envs: int,
    seed: int,
    reset_pool_size: int = 4096,
    pool: Any | None = None,
    frozen_opponent_params: dict | None = None,
) -> CurriculumCarry:
    key = jax.random.PRNGKey(seed)
    key, pool_key, seat_key, rollout_key = jax.random.split(key, 4)
    if pool is None:
        pool = build_competition_reset_pool(pool_key, reset_pool_size)
    pool_size = int(jax.tree_util.tree_leaves(pool)[0].shape[0])
    if pool_size < 2 * num_envs:
        raise ValueError("reset pool must contain at least two distinct entries per environment")
    initial_indices = jnp.arange(num_envs, dtype=jnp.int32)
    states = jax.tree_util.tree_map(lambda value: value[initial_indices], pool)
    memory = lambda: jax.tree_util.tree_map(  # noqa: E731
        lambda value: jnp.broadcast_to(value, (num_envs,) + value.shape), empty_memory()
    )
    learner_seat = jax.random.randint(seat_key, (num_envs,), 0, 2, dtype=jnp.int32)
    return CurriculumCarry(
        states=states,
        mem0=memory(),
        mem1=memory(),
        key=rollout_key,
        params=params,
        pool=pool,
        pool_cursor=jnp.arange(num_envs, 2 * num_envs, dtype=jnp.int32),
        learner_seat=learner_seat,
        episode_id=jnp.arange(num_envs, dtype=jnp.int32),
        frozen_opponent_params=(
            params if frozen_opponent_params is None else frozen_opponent_params
        ),
    )


def _potential(states: Any, learner_seat: jax.Array) -> jax.Array:
    """Bounded potential using only each learner's official fogged observation."""
    from generals.core import game

    obs = jax.vmap(game.get_observation)(states, learner_seat)
    land = obs.owned_land_count.astype(jnp.float32)
    visible_enemy = jnp.sum(obs.opponent_cells, axis=(-2, -1)).astype(jnp.float32)
    return 0.6 * jnp.tanh(land / 30.0) + 0.4 * jnp.tanh(visible_enemy / 5.0)


def rollout_step(
    carry: CurriculumCarry,
    _,
    *,
    opponent_schedule: tuple[int, ...],
    gamma: float,
    shaping_lambda: float,
    deterministic_learner: bool,
):
    (
        states,
        mem0,
        mem1,
        key,
        params,
        pool,
        pool_cursor,
        learner_seat,
        episode_id,
        frozen_opponent_params,
    ) = carry
    num_envs = len(opponent_schedule)
    key, learner_key, opponent_key, next_seat_key = jax.random.split(key, 4)

    spatial0, global0, mem0 = observe_batch_p0(states, mem0)
    spatial1, global1, mem1 = observe_batch_p1(states, mem1)
    mask0, mask1 = legal_mask_batch_p0(states), legal_mask_batch_p1(states)
    seat_spatial = learner_seat.reshape((-1, 1, 1, 1)).astype(bool)
    seat_global = learner_seat.reshape((-1, 1)).astype(bool)
    learner_spatial = jnp.where(seat_spatial, spatial1, spatial0)
    learner_global = jnp.where(seat_global, global1, global0)
    learner_mask = jnp.where(seat_global, mask1, mask0)

    output = forward_batch(params, learner_spatial, learner_global)
    learner_keys = jax.random.split(learner_key, num_envs)
    if deterministic_learner:
        learner_logp_all = jax.vmap(masked_log_softmax)(
            output["flat_logits"], learner_mask
        )
        learner_action = jnp.argmax(learner_logp_all, axis=-1).astype(jnp.int32)
        learner_logp = jnp.take_along_axis(
            learner_logp_all, learner_action[:, None], axis=1
        )[:, 0]
    else:
        learner_action, learner_logp = sample_action_batch(
            learner_keys, output["flat_logits"], learner_mask
        )
    learner_value = _value_from_logits(output["value_logits"])
    learner_engine = index_to_engine_action_batch(learner_action)

    opponent_seat = 1 - learner_seat
    opponent_keys = jax.random.split(opponent_key, num_envs)
    opponent_engine = batched_opponent_actions(
        states, opponent_seat, opponent_keys, opponent_schedule
    )
    frozen_indices = tuple(
        index
        for index, kind in enumerate(opponent_schedule)
        if kind == int(OpponentKind.FROZEN_BEST)
    )
    if frozen_indices:
        idx = jnp.asarray(frozen_indices, dtype=jnp.int32)
        opponent_is_p0 = opponent_seat.reshape((-1, 1)).astype(bool) == 0
        opponent_spatial = jnp.where(
            opponent_is_p0.reshape((-1, 1, 1, 1)), spatial0, spatial1
        )
        opponent_global = jnp.where(opponent_is_p0, global0, global1)
        opponent_mask = jnp.where(opponent_is_p0, mask0, mask1)
        frozen_output = forward_batch(
            frozen_opponent_params, opponent_spatial[idx], opponent_global[idx]
        )
        frozen_actions, _ = sample_action_batch(
            opponent_keys[idx], frozen_output["flat_logits"], opponent_mask[idx]
        )
        opponent_engine = opponent_engine.at[idx].set(
            index_to_engine_action_batch(frozen_actions)
        )
    learner_is_p1 = learner_seat.reshape((-1, 1)).astype(bool)
    action0 = jnp.where(learner_is_p1, opponent_engine, learner_engine)
    action1 = jnp.where(learner_is_p1, learner_engine, opponent_engine)
    joint_actions = jnp.stack([action0, action1], axis=1)

    prior_potential = _potential(states, learner_seat)
    next_states_unreset, rewards, terminated, truncated, info = step_batch_jax(
        states, joint_actions
    )
    done = terminated | truncated
    terminal_reward = jnp.take_along_axis(rewards, learner_seat[:, None], axis=1)[:, 0]
    next_potential = _potential(next_states_unreset, learner_seat)
    next_potential = jnp.where(done, 0.0, next_potential)
    shaped_reward = shaping_lambda * (gamma * next_potential - prior_potential)
    shaped_reward = jnp.clip(shaped_reward, -0.02, 0.02)
    combined_reward = terminal_reward + shaped_reward

    next_states, pool_cursor = auto_reset_from_pool(
        next_states_unreset, terminated, truncated, pool, pool_cursor
    )
    reset_shape = done.reshape((-1,) + (1,) * (mem0.seen_own.ndim - 1))
    zero0, zero1 = jnp.zeros_like(mem0.seen_own), jnp.zeros_like(mem1.seen_own)
    mem0 = ObsMemoryJax(
        seen_own=jnp.where(reset_shape, zero0, mem0.seen_own),
        last_army=jnp.where(reset_shape, zero0, mem0.last_army),
    )
    mem1 = ObsMemoryJax(
        seen_own=jnp.where(reset_shape, zero1, mem1.seen_own),
        last_army=jnp.where(reset_shape, zero1, mem1.last_army),
    )
    sampled_seat = jax.random.randint(next_seat_key, (num_envs,), 0, 2, dtype=jnp.int32)
    next_learner_seat = jnp.where(done, sampled_seat, learner_seat)
    next_episode_id = episode_id + done.astype(jnp.int32) * jnp.int32(num_envs)

    learner_won = terminated & (info.winner == learner_seat)
    learner_lost = terminated & (info.winner == opponent_seat)
    learner_land = jnp.take_along_axis(info.land, learner_seat[:, None], axis=1)[:, 0]
    opponent_land = jnp.take_along_axis(info.land, opponent_seat[:, None], axis=1)[:, 0]
    learner_army = jnp.take_along_axis(info.army, learner_seat[:, None], axis=1)[:, 0]
    opponent_army = jnp.take_along_axis(info.army, opponent_seat[:, None], axis=1)[:, 0]
    trajectory = {
        "spatial": learner_spatial,
        "global": learner_global,
        "mask": learner_mask,
        "actions": learner_action,
        "old_logp": learner_logp,
        "values": learner_value,
        "terminal_rewards": terminal_reward,
        "shaped_rewards": shaped_reward,
        "rewards": combined_reward,
        "dones": done.astype(jnp.float32),
        "terminated": terminated,
        "truncated": truncated,
        "learner_controlled_mask": jnp.ones((num_envs,), dtype=bool),
        "learner_seat": learner_seat,
        "opponent_kind": jnp.asarray(opponent_schedule, dtype=jnp.int32),
        "episode_id": episode_id,
        "turn": states.time,
        "learner_won": learner_won,
        "learner_lost": learner_lost,
        "learner_pass": learner_action == 0,
        "learner_land": learner_land,
        "opponent_land": opponent_land,
        "learner_army": learner_army,
        "opponent_army": opponent_army,
    }
    next_carry = CurriculumCarry(
        next_states,
        mem0,
        mem1,
        key,
        params,
        pool,
        pool_cursor,
        next_learner_seat,
        next_episode_id,
        frozen_opponent_params,
    )
    return next_carry, trajectory


@partial(
    jax.jit,
    static_argnames=(
        "rollout_len",
        "opponent_schedule",
        "gamma",
        "shaping_lambda",
        "deterministic_learner",
    ),
)
def _run_rollout_scan(
    carry: CurriculumCarry,
    *,
    rollout_len: int,
    opponent_schedule: tuple[int, ...],
    gamma: float,
    shaping_lambda: float,
    deterministic_learner: bool,
):
    def step(inner_carry, item):
        return rollout_step(
            inner_carry,
            item,
            opponent_schedule=opponent_schedule,
            gamma=gamma,
            shaping_lambda=shaping_lambda,
            deterministic_learner=deterministic_learner,
        )

    return jax.lax.scan(step, carry, xs=None, length=rollout_len)


def collect_curriculum_batch(
    params: dict,
    *,
    opponent_schedule: tuple[int, ...],
    rollout_len: int = 32,
    seed: int = 0,
    reset_pool_size: int = 4096,
    pool: Any | None = None,
    carry: CurriculumCarry | None = None,
    frozen_opponent_params: dict | None = None,
    gamma: float = 1.0,
    shaping_lambda: float = 0.0,
    deterministic_learner: bool = False,
) -> tuple[dict[str, Any], CurriculumCarry]:
    num_envs = len(opponent_schedule)
    if carry is None:
        carry = initialise_curriculum_carry(
            params,
            num_envs=num_envs,
            seed=seed,
            reset_pool_size=reset_pool_size,
            pool=pool,
            frozen_opponent_params=frozen_opponent_params,
        )
    else:
        carry = carry._replace(params=params)
    final_carry, trajectory = _run_rollout_scan(
        carry,
        rollout_len=rollout_len,
        opponent_schedule=opponent_schedule,
        gamma=gamma,
        shaping_lambda=shaping_lambda,
        deterministic_learner=deterministic_learner,
    )
    jax.block_until_ready(trajectory["rewards"])

    spatial0, global0, _ = observe_batch_p0(final_carry.states, final_carry.mem0)
    spatial1, global1, _ = observe_batch_p1(final_carry.states, final_carry.mem1)
    seat_spatial = final_carry.learner_seat.reshape((-1, 1, 1, 1)).astype(bool)
    seat_global = final_carry.learner_seat.reshape((-1, 1)).astype(bool)
    bootstrap_spatial = jnp.where(seat_spatial, spatial1, spatial0)
    bootstrap_global = jnp.where(seat_global, global1, global0)
    bootstrap_output = forward_batch(params, bootstrap_spatial, bootstrap_global)
    bootstrap = _value_from_logits(bootstrap_output["value_logits"])
    jax.block_until_ready(bootstrap)
    return {
        **trajectory,
        "bootstrap_values": bootstrap,
        "rollout_architecture": "PERSISTENT_LEARNER_VS_JAX_CURRICULUM_SCAN",
        "opponent_schedule": opponent_schedule,
        "gamma": gamma,
        "shaping_lambda": shaping_lambda,
    }, final_carry
