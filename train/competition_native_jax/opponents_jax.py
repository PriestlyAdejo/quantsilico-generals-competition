"""Batched adapters for the four frozen curriculum opponents.

The official agents consume fogged ``generals.core.observation.Observation``
objects and return ``[pass,row,col,direction,split]``.  The competition engine
consumes ``[kind,row,col,direction,split]`` where kind 0 is MOVE and kind 1 is
PASS.  These adapters perform only that public action-format conversion; they
never expose the underlying simulator state to an opponent.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import jax
import jax.numpy as jnp
from generals.agents.expander_agent import ExpanderAgent
from generals.agents.hunter_agent import HunterAgent
from generals.agents.random_agent import RandomAgent
from generals.core import game


class OpponentKind(IntEnum):
    PASS = 0
    RANDOM = 1
    EXPANDER = 2
    HUNTER = 3
    FROZEN_BEST = 4


PASS_OPPONENT_ID = "official_engine_pass_action"
RANDOM_OPPONENT_ID = "generals.agents.random_agent.RandomAgent"
EXPANDER_OPPONENT_ID = "generals.agents.expander_agent.ExpanderAgent"
HUNTER_OPPONENT_ID = "generals.agents.hunter_agent.HunterAgent"

_RANDOM = RandomAgent()
_EXPANDER = ExpanderAgent()
_HUNTER = HunterAgent()


def official_to_engine_action(action: jax.Array) -> jax.Array:
    """Convert canonical agent action format into competition engine format."""
    is_pass = action[0].astype(bool)
    return jnp.array(
        [
            jnp.where(is_pass, 1, 0),
            jnp.where(is_pass, 0, action[1]),
            jnp.where(is_pass, 0, action[2]),
            jnp.where(is_pass, 0, action[3]),
            jnp.where(is_pass, 0, action[4]),
        ],
        dtype=jnp.int32,
    )


official_to_engine_action_batch = jax.jit(jax.vmap(official_to_engine_action))


def _select_observations(states: Any, seats: jax.Array):
    return jax.vmap(game.get_observation)(states, seats)


def _act_random(observations: Any, keys: jax.Array) -> jax.Array:
    return jax.vmap(lambda obs, key: _RANDOM.act(obs, key))(observations, keys)


def _act_expander(observations: Any, keys: jax.Array) -> jax.Array:
    return jax.vmap(lambda obs, key: _EXPANDER.act(obs, key))(observations, keys)


def _act_hunter(observations: Any, keys: jax.Array) -> jax.Array:
    return jax.vmap(lambda obs, key: _HUNTER.act(obs, key))(observations, keys)


def batched_opponent_actions(
    states: Any,
    opponent_seats: jax.Array,
    keys: jax.Array,
    opponent_schedule: tuple[int, ...],
) -> jax.Array:
    """Return engine actions for a static, batched opponent schedule.

    Static grouping avoids Python calls per environment and avoids evaluating
    Hunter/Expander for environments assigned to cheaper opponents.
    """
    n = len(opponent_schedule)
    if n == 0:
        raise ValueError("opponent schedule must not be empty")
    observations = _select_observations(states, opponent_seats)
    actions = jnp.zeros((n, 5), dtype=jnp.int32)
    schedule = tuple(int(item) for item in opponent_schedule)

    for kind, actor in (
        (OpponentKind.RANDOM, _act_random),
        (OpponentKind.EXPANDER, _act_expander),
        (OpponentKind.HUNTER, _act_hunter),
    ):
        indices = tuple(i for i, item in enumerate(schedule) if item == int(kind))
        if not indices:
            continue
        idx = jnp.asarray(indices, dtype=jnp.int32)
        obs_subset = jax.tree_util.tree_map(
            lambda value, selected=idx: value[selected], observations
        )
        canonical = actor(obs_subset, keys[idx])
        actions = actions.at[idx].set(official_to_engine_action_batch(canonical))

    pass_indices = tuple(i for i, item in enumerate(schedule) if item == int(OpponentKind.PASS))
    if pass_indices:
        idx = jnp.asarray(pass_indices, dtype=jnp.int32)
        actions = actions.at[idx, 0].set(1)
    return actions


def build_static_schedule(
    num_envs: int, weights: tuple[float, ...]
) -> tuple[int, ...]:
    """Largest-remainder allocation for an exact static opponent mixture."""
    if num_envs <= 0 or len(weights) not in (4, 5) or any(weight < 0 for weight in weights):
        raise ValueError("invalid environment count/opponent weights")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("at least one opponent weight must be positive")
    quotas = [num_envs * float(weight) / total for weight in weights]
    counts = [int(quota) for quota in quotas]
    remainder = num_envs - sum(counts)
    order = sorted(
        range(len(weights)), key=lambda i: (quotas[i] - counts[i], -i), reverse=True
    )
    for i in order[:remainder]:
        counts[i] += 1
    schedule: list[int] = []
    for kind, count in enumerate(counts):
        schedule.extend([kind] * count)
    return tuple(schedule)
