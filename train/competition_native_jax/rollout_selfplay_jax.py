"""Canonical fused self-play rollout: official MIT JAX transitions + lax.scan.

Architecture: END_TO_END_COMPETITION_JAX_ROLLOUT

The Python timestep collector previously in this module is superseded and must
not be used as the training entrypoint. Host-bound optimiser lineage must not
resume into this path.
"""

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
from generals_bot.competition_native_jax.constants import MAX_HW
from generals_bot.competition_native_jax.inference_jax import sample_action
from generals_bot.competition_native_jax.obs_memory import N_SPATIAL
from generals_bot.competition_native_jax.transformer_jax import forward_batch
from train.competition_native_jax.reward_shaping_jax import active_shaping, shape_step_rewards
from train.competition_native_jax.temporal_history_jax import active_temporal_history

sample_action_batch = jax.jit(jax.vmap(sample_action, in_axes=(0, 0, 0)))

ROLLOUT_ARCHITECTURE = "END_TO_END_COMPETITION_JAX_ROLLOUT"
# Explicit demotion of prior host/Python-timestep path
SUPERSEDED_COLLECTOR = "HOST_BOUND_PYTHON_TIMESTEP_COLLECTOR"

# STAGE6_OPPDIST_R1: opponent-distribution knob. "self" = canonical mirror
# self-play (bit-identical control path). "teacher_frozen" = seat 1 acts from
# params carried in RolloutCarry.opp_params, loaded once and never updated
# (environment dynamics; PPO_SEMANTICS UNCHANGED for the trained seat).
_OPPONENT_STATE = {"mode": "self"}


def set_opponent_mode(mode: str) -> None:
    if mode not in ("self", "teacher_frozen"):
        raise ValueError(f"unknown opponent mode: {mode}")
    _OPPONENT_STATE["mode"] = mode


def active_opponent_mode() -> str:
    return _OPPONENT_STATE["mode"]


class RolloutCarry(NamedTuple):
    states: Any
    mem0: ObsMemoryJax
    mem1: ObsMemoryJax
    key: jax.Array
    params: dict
    pool: Any
    pool_cursor: jax.Array
    # STAGE5 T2 (k1): previous tick's LEGAL spatial obs per seat; zeros when
    # temporal history mode is off (bit-inert) and at episode boundaries.
    prev_sp0: jax.Array
    prev_sp1: jax.Array
    # STAGE6_OPPDIST_R1: frozen seat-1 params; None in mode "self" (unused by
    # the trace, keeps the control path bit-identical).
    opp_params: dict | None = None


def initialise_rollout_carry(
    params: dict,
    *,
    num_envs: int,
    seed: int,
    reset_pool_size: int = 4096,
    pool: Any | None = None,
    opp_params: dict | None = None,
) -> RolloutCarry:
    """Create episode state once for a continuous sequence of PPO updates."""
    key = jax.random.PRNGKey(seed)
    key, pk, rk = jax.random.split(key, 3)
    if pool is None:
        pool = build_competition_reset_pool(pk, reset_pool_size)
    pool_size = int(jax.tree_util.tree_leaves(pool)[0].shape[0])
    if pool_size < num_envs:
        raise ValueError(f"reset_pool_size/actual {pool_size} < num_envs {num_envs}")
    init_idx = jnp.arange(num_envs, dtype=jnp.int32)
    states = jax.tree_util.tree_map(lambda x: x[init_idx], pool)
    pool_cursor = jnp.full((num_envs,), num_envs, dtype=jnp.int32)
    mem0 = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_envs), empty_memory())
    mem1 = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_envs), empty_memory())
    prev = jnp.zeros((num_envs, N_SPATIAL, MAX_HW, MAX_HW), dtype=jnp.float32)
    if active_opponent_mode() == "teacher_frozen" and opp_params is None:
        raise ValueError("opponent mode teacher_frozen requires opp_params at carry init")
    return RolloutCarry(states, mem0, mem1, rk, params, pool, pool_cursor, prev, prev, opp_params)


@partial(jax.jit, static_argnames=("rollout_len",))
def _run_rollout_scan(carry: RolloutCarry, *, rollout_len: int):
    return jax.lax.scan(rollout_step, carry, xs=None, length=rollout_len)


def _value_from_logits(value_logits: jax.Array) -> jax.Array:
    centers = jnp.linspace(-1.0, 1.0, value_logits.shape[-1])
    return jnp.sum(jax.nn.softmax(value_logits, axis=-1) * centers, axis=-1)


def rollout_step(carry: RolloutCarry, _):
    """One fused self-play step (both seats); scanned over time."""
    (states, mem0, mem1, key, params, pool, pool_cursor, prev_sp0, prev_sp1, opp_params) = carry
    key, k0, _k1 = jax.random.split(key, 3)

    sp0, gv0, mem0 = observe_batch_p0(states, mem0)
    sp1, gv1, mem1 = observe_batch_p1(states, mem1)
    m0 = legal_mask_batch_p0(states)
    m1 = legal_mask_batch_p1(states)

    # STAGE5 T2: k1 appends the previous tick's LEGAL spatial obs (zeroed at
    # episode boundaries). Mode "off" keeps the canonical 8-plane path exact.
    if active_temporal_history() == "k1":
        in0 = jnp.concatenate([sp0, prev_sp0], axis=1)
        in1 = jnp.concatenate([sp1, prev_sp1], axis=1)
    else:
        in0, in1 = sp0, sp1

    n = sp0.shape[0]
    if active_opponent_mode() == "teacher_frozen":
        # STAGE6_OPPDIST_R1: seat 0 = trained policy (canonical fused forward);
        # seat 1 = frozen opponent acting from opp_params, sampled
        # stochastically (temperature 1.0). Trained-seat action selection is
        # untouched; the opponent is environment dynamics.
        out0 = forward_batch(params, in0, gv0)
        out1 = forward_batch(opp_params, in1, gv1)
        keys = jax.random.split(k0, 2 * n)
        a0, lp0 = sample_action_batch(keys[:n], out0["flat_logits"], m0)
        a1, _lp1 = sample_action_batch(keys[n:], out1["flat_logits"], m1)
        v0 = _value_from_logits(out0["value_logits"])
    else:
        # Concatenate both seats into one native batch forward [2N, ...]
        spatial = jnp.concatenate([in0, in1], axis=0)
        global_vec = jnp.concatenate([gv0, gv1], axis=0)
        masks = jnp.concatenate([m0, m1], axis=0)
        out = forward_batch(params, spatial, global_vec)
        keys = jax.random.split(k0, 2 * n)
        actions, logps = sample_action_batch(keys, out["flat_logits"], masks)
        a0, a1 = actions[:n], actions[n:]
        lp0 = logps[:n]
        v0 = _value_from_logits(out["value_logits"])[:n]
    eng0 = index_to_engine_action_batch(a0)
    eng1 = index_to_engine_action_batch(a1)
    joint = jnp.stack([eng0, eng1], axis=1)

    next_states, rewards, terminated, truncated, _info = step_batch_jax(states, joint)
    done = terminated | truncated
    rewards0 = rewards[:, 0]
    rewards1 = rewards[:, 1]
    # REWARD-SHAPING-R1 (EV-0044): training-reward shaping for the trained seat
    # only. PPO_SEMANTICS UNCHANGED - identity at mode "none" (control path).
    _shape_mode, _shape_beta = active_shaping()
    if _shape_mode != "none" and _shape_beta > 0.0:
        rewards0 = shape_step_rewards(
            states, next_states, rewards0, terminated, truncated, _shape_mode, _shape_beta
        )

    next_states, pool_cursor = auto_reset_from_pool(
        next_states, terminated, truncated, pool, pool_cursor
    )

    # Clear memory on episode boundary
    z = jnp.zeros_like(mem0.seen_own)
    done_m = done.reshape((-1,) + (1,) * (mem0.seen_own.ndim - 1))
    mem0 = ObsMemoryJax(
        seen_own=jnp.where(done_m, z, mem0.seen_own),
        last_army=jnp.where(done_m, z, mem0.last_army),
    )
    mem1 = ObsMemoryJax(
        seen_own=jnp.where(done_m, z, mem1.seen_own),
        last_army=jnp.where(done_m, z, mem1.last_army),
    )

    # STAGE5 T2: advance history; zero the carried frame for envs that ended.
    done_sp = done.reshape((-1,) + (1,) * (sp0.ndim - 1))
    prev_sp0 = jnp.where(done_sp, jnp.zeros_like(sp0), sp0)
    prev_sp1 = jnp.where(done_sp, jnp.zeros_like(sp1), sp1)

    traj = {
        "spatial": in0,
        "global": gv0,
        "mask": m0,
        "actions": a0,
        "old_logp": lp0,
        "values": v0,
        "rewards": rewards0,
        "rewards1": rewards1,  # STAGE6_OPPDIST_R1: teacher-opponent outcome diagnostic
        "dones": done.astype(jnp.float32),
        "terminals": terminated.astype(jnp.float32),  # decisive-game diagnostic (EV-0044)
    }
    return (
        RolloutCarry(
            next_states, mem0, mem1, key, params, pool, pool_cursor, prev_sp0, prev_sp1, opp_params
        ),
        traj,
    )


def collect_selfplay_batch(
    params: dict,
    *,
    num_envs: int = 4,
    rollout_len: int = 32,
    seed: int = 0,
    reset_pool_size: int = 4096,
    pool: Any | None = None,
    carry: RolloutCarry | None = None,
    return_carry: bool = False,
    opp_params: dict | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], RolloutCarry]:
    """Collect one rollout, optionally continuing episode state from the prior update.

    Callers performing multiple PPO updates must retain and pass ``carry``.  The
    default remains a one-shot collector for benchmarks and compatibility.
    """
    if carry is None:
        carry = initialise_rollout_carry(
            params,
            num_envs=num_envs,
            seed=seed,
            reset_pool_size=reset_pool_size,
            pool=pool,
            opp_params=opp_params,
        )
    else:
        carry = carry._replace(params=params)
    pool_size = int(jax.tree_util.tree_leaves(carry.pool)[0].shape[0])
    final_carry, traj = _run_rollout_scan(carry, rollout_len=rollout_len)
    jax.block_until_ready(traj["rewards"])
    # Bootstrap value from post-scan p0 observation (device-side; no host traj rebuild)
    sp_b, gv_b, _ = observe_batch_p0(final_carry.states, final_carry.mem0)
    if active_temporal_history() == "k1":
        sp_b = jnp.concatenate([sp_b, final_carry.prev_sp0], axis=1)
    out_b = forward_batch(params, sp_b, gv_b)
    bootstrap = _value_from_logits(out_b["value_logits"])
    jax.block_until_ready(bootstrap)
    batch = {
        **traj,
        "bootstrap_values": bootstrap,
        "backend": "official_mit_jax_primitives_plus_qs_scan",
        "rollout_architecture": ROLLOUT_ARCHITECTURE,
        "superseded_collector": SUPERSEDED_COLLECTOR,
        "reset_pool_size": pool_size,
        "reset_path": "device_competition_reset_pool",
    }
    return (batch, final_carry) if return_carry else batch


# Compatibility re-exports used by older tests (wrappers, not training entrypoint)
from generals_bot.competition_native_jax.competition_env_jax import (  # noqa: E402, F401, I001
    legal_mask_one_jax as legal_mask_jax,
    observe_one_jax,
)

forward_batch_export = forward_batch
