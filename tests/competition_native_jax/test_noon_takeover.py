from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generals_bot.competition_native_jax.competition_env_jax import (
    empty_memory,
    reset_one_jax,
)
from generals_bot.observation import GameContext
from generals_bot.selector import create_policy
from scripts.collect_noon_closed_loop_dagger import (
    Snapshot,
    clone_gate,
    opponent_action_from_rollout_key,
)
from train.competition_native_jax.opponents_jax import (
    OpponentKind,
    batched_opponent_actions,
)


@pytest.mark.parametrize("opponent_kind", [OpponentKind.PASS, OpponentKind.RANDOM])
def test_teacher_shadow_clone_includes_policy_state_and_transition(
    opponent_kind: OpponentKind,
) -> None:
    teacher = create_policy("heuristic_v2f_plus_planner_terminal_fix")
    snapshot = Snapshot(
        turn=0,
        state=reset_one_jax(jax.random.PRNGKey(701), 21, 21),
        memory=empty_memory(),
        teacher=teacher,
        teacher_state=teacher.initial_state(GameContext(0, 21, 21)),
        opponent_kind=int(opponent_kind),
        rollout_key=jax.random.PRNGKey(702),
        opponent_env_index=0,
        opponent_num_envs=1,
        learner_seat=0,
        game_id="clone-fixture",
    )
    result = clone_gate(snapshot)
    assert result["status"] == "TEACHER_SHADOW_CLONE_GATE_PASS"
    assert result["next_transition_equal"] is True


def test_snapshot_opponent_rng_matches_batched_rollout_key_schedule() -> None:
    key = jax.random.PRNGKey(903)
    states = jax.vmap(lambda seed: reset_one_jax(jax.random.PRNGKey(seed), 21, 21))(
        jnp.arange(3, dtype=jnp.int32) + 910
    )
    action, next_key = opponent_action_from_rollout_key(
        jax.tree_util.tree_map(lambda value: value[1], states),
        learner_seat=0,
        opponent_kind=int(OpponentKind.RANDOM),
        rollout_key=key,
        environment_index=1,
        num_environments=3,
    )
    expected_next, _learner, opponent_root, _seat = jax.random.split(key, 4)
    expected = batched_opponent_actions(
        states,
        jnp.ones((3,), dtype=jnp.int32),
        jax.random.split(opponent_root, 3),
        (int(OpponentKind.RANDOM),) * 3,
    )[1]
    assert np.array_equal(np.asarray(action), np.asarray(expected))
    assert np.array_equal(np.asarray(next_key), np.asarray(expected_next))
