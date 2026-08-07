"""Frozen-fixture equivalence gate for the reusable rollout-scan candidate."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from generals_bot.competition_native_jax.competition_env_jax import (
    build_competition_reset_pool,
    empty_memory,
)
from generals_bot.competition_native_jax.transformer_jax import init_params
from train.competition_native_jax.gae_jax import gae_advantages_batch_jit
from train.competition_native_jax.ppo_jax import (
    make_optimizer,
    ppo_loss_on_batch,
    ppo_update,
)
from train.competition_native_jax.rollout_selfplay_jax import (
    RolloutCarry,
    _run_rollout_scan,
    rollout_step,
)
from train.competition_native_jax.train_jax import (
    load_training_checkpoint,
    save_training_checkpoint,
)


def _frozen_reference_run(carry: RolloutCarry, rollout_len: int):
    """Exact pre-candidate collector wrapper, retained only as test oracle."""

    def scan_fn(c, _):
        return rollout_step(c, None)

    @jax.jit
    def run(c):
        return jax.lax.scan(scan_fn, c, xs=None, length=rollout_len)

    return run(carry)


def _fixture() -> tuple[RolloutCarry, int]:
    num_envs = 1
    key = jax.random.PRNGKey(20260807)
    key, params_key, pool_key, rollout_key = jax.random.split(key, 4)
    del key
    params = init_params(params_key)
    pool = build_competition_reset_pool(pool_key, 2, min_grid=21, max_grid=21)
    states = jax.tree_util.tree_map(lambda x: x[:num_envs], pool)
    memory = empty_memory()
    mem0 = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_envs), memory)
    mem1 = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_envs), memory)
    carry = RolloutCarry(
        states,
        mem0,
        mem1,
        rollout_key,
        params,
        pool,
        jnp.full((num_envs,), num_envs, dtype=jnp.int32),
    )
    return carry, 2


def _assert_tree_close(left, right, *, rtol: float = 1e-6, atol: float = 1e-6) -> None:
    left_leaves, left_def = jax.tree_util.tree_flatten(left)
    right_leaves, right_def = jax.tree_util.tree_flatten(right)
    assert left_def == right_def
    assert len(left_leaves) == len(right_leaves)
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=rtol, atol=atol)


def _ppo_inputs(trajectory: dict[str, jax.Array], bootstrap: jax.Array) -> dict[str, jax.Array]:
    time_steps, num_envs = trajectory["rewards"].shape
    values = jnp.concatenate([trajectory["values"], bootstrap[None, :]], axis=0)
    advantages, returns = gae_advantages_batch_jit(
        trajectory["rewards"], values, trajectory["dones"]
    )
    return {
        "spatial": trajectory["spatial"].reshape(
            time_steps * num_envs, *trajectory["spatial"].shape[2:]
        ),
        "global": trajectory["global"].reshape(
            time_steps * num_envs, *trajectory["global"].shape[2:]
        ),
        "mask": trajectory["mask"].reshape(time_steps * num_envs, -1),
        "actions": trajectory["actions"].reshape(time_steps * num_envs),
        "old_logp": trajectory["old_logp"].reshape(time_steps * num_envs),
        "advantages": advantages.reshape(time_steps * num_envs),
        "returns": returns.reshape(time_steps * num_envs),
    }


def test_performance_equivalence_gate(tmp_path: Path) -> None:
    carry, rollout_len = _fixture()
    reference_carry, reference_trajectory = _frozen_reference_run(carry, rollout_len)
    candidate_carry, candidate_trajectory = _run_rollout_scan(carry, rollout_len=rollout_len)

    # Final state/memory/RNG/pool cursor covers transition and reset semantics.
    _assert_tree_close(reference_carry, candidate_carry)
    # Trajectory covers observations, legal masks, actions, log-probabilities,
    # values, rewards, and done semantics.
    _assert_tree_close(reference_trajectory, candidate_trajectory)

    cache_entries = _run_rollout_scan._cache_size()
    _run_rollout_scan(carry, rollout_len=rollout_len)[1]["rewards"].block_until_ready()
    assert cache_entries == 1
    assert _run_rollout_scan._cache_size() == cache_entries

    # The production bootstrap is a policy value on the post-scan state. The
    # optimizer gate only needs the same deterministic bootstrap fixture to
    # prove that rollout-wrapper reuse does not change downstream learning.
    bootstrap = jnp.zeros((reference_trajectory["rewards"].shape[1],), dtype=jnp.float32)
    reference_batch = _ppo_inputs(reference_trajectory, bootstrap)
    candidate_batch = _ppo_inputs(candidate_trajectory, bootstrap)
    _assert_tree_close(reference_batch, candidate_batch)

    optimizer = make_optimizer(3e-4)
    reference_opt = optimizer.init(reference_carry.params)
    candidate_opt = optimizer.init(candidate_carry.params)

    def loss_and_grad(params, batch):
        def loss_fn(p):
            return ppo_loss_on_batch(
                p,
                batch["spatial"],
                batch["global"],
                batch["mask"],
                batch["actions"],
                batch["old_logp"],
                batch["advantages"],
                batch["returns"],
            )

        return jax.value_and_grad(loss_fn, has_aux=True)(params)

    (reference_loss, reference_metrics_pre), reference_grads = loss_and_grad(
        reference_carry.params, reference_batch
    )
    (candidate_loss, candidate_metrics_pre), candidate_grads = loss_and_grad(
        candidate_carry.params, candidate_batch
    )
    _assert_tree_close(reference_loss, candidate_loss)
    _assert_tree_close(reference_metrics_pre, candidate_metrics_pre)
    _assert_tree_close(reference_grads, candidate_grads)

    reference_params, reference_opt, reference_metrics = ppo_update(
        reference_carry.params, reference_opt, optimizer, reference_batch
    )
    candidate_params, candidate_opt, candidate_metrics = ppo_update(
        candidate_carry.params, candidate_opt, optimizer, candidate_batch
    )
    _assert_tree_close(reference_params, candidate_params)
    _assert_tree_close(reference_opt, candidate_opt)
    _assert_tree_close(reference_metrics, candidate_metrics)

    checkpoint = tmp_path / "candidate-checkpoint"
    meta = {"runtime_implementation": "PERF_V1_REUSABLE_ROLLOUT_SCAN"}
    save_training_checkpoint(
        checkpoint,
        params=candidate_params,
        ema=candidate_params,
        opt_state=candidate_opt,
        meta=meta,
    )
    loaded = load_training_checkpoint(
        checkpoint, params_like=candidate_params, opt_state_like=candidate_opt
    )
    _assert_tree_close(loaded["params"], candidate_params)
    _assert_tree_close(loaded["ema"], candidate_params)
    _assert_tree_close(loaded["opt_state"], candidate_opt)
    assert loaded["meta"] == meta
