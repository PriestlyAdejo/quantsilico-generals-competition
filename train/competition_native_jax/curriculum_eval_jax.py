"""RC_R1_BRIDGE D2: competence-based spawn-distance curriculum diagnostic eval.

Greedy-vs-legal_random evaluation entirely in the canonical JAX environment.
This is a TRAINING-SCHEDULE DIAGNOSTIC (curriculum advancement rule), not
promotion evidence: gameplay arbiter remains the promotion authority.
PPO_SEMANTICS: UNCHANGED (never touches the trained seat's action selection).
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from generals_bot.competition_native_jax.competition_env_jax import (
    ObsMemoryJax,
    auto_reset_from_pool,
    empty_memory,
    index_to_engine_action_batch,
    legal_mask_batch_p0,
    legal_mask_batch_p1,
    observe_batch_p0,
    observe_batch_p1,
    step_batch_jax,
)
from generals_bot.competition_native_jax.inference_jax import sample_action
from generals_bot.competition_native_jax.transformer_jax import forward_batch

sample_action_batch = jax.jit(jax.vmap(sample_action, in_axes=(0, 0, 0)))


class EvalCarry(NamedTuple):
    states: Any
    mem0: ObsMemoryJax
    mem1: ObsMemoryJax
    key: jax.Array
    params: dict
    pool: Any
    pool_cursor: jax.Array
    wins: jax.Array
    losses: jax.Array
    decided: jax.Array


def _eval_step(carry: EvalCarry, _):
    (states, mem0, mem1, key, params, pool, pool_cursor, wins, losses, decided) = carry
    key, k1 = jax.random.split(key)

    sp0, gv0, mem0 = observe_batch_p0(states, mem0)
    sp1, gv1, mem1 = observe_batch_p1(states, mem1)
    m0 = legal_mask_batch_p0(states)
    m1 = legal_mask_batch_p1(states)

    n = sp0.shape[0]
    # Seat 0: greedy argmax over the legal mask (serving semantics).
    neg_inf = jnp.finfo(jnp.float32).min
    logits0 = forward_batch(params, sp0, gv0)["flat_logits"]
    a0 = jnp.argmax(jnp.where(m0, logits0, neg_inf), axis=-1)
    # Seat 1: uniform over legal actions (legal_random opponent): flat logits.
    keys = jax.random.split(k1, n)
    a1, _ = sample_action_batch(keys, jnp.zeros((n, m1.shape[-1])), m1)

    eng0 = index_to_engine_action_batch(a0)
    eng1 = index_to_engine_action_batch(a1)
    joint = jnp.stack([eng0, eng1], axis=1)
    next_states, rewards, terminated, truncated, _info = step_batch_jax(states, joint)
    done = terminated | truncated
    new_decided = done & ~decided
    wins = wins + jnp.sum((rewards[:, 0] > 0.5) & new_decided)
    losses = losses + jnp.sum((rewards[:, 0] < -0.5) & new_decided)
    decided = decided | done

    next_states, pool_cursor = auto_reset_from_pool(
        next_states, terminated, truncated, pool, pool_cursor
    )
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
    return (
        EvalCarry(
            next_states, mem0, mem1, key, params, pool, pool_cursor,
            wins, losses, decided,
        ),
        None,
    )


@partial(jax.jit, static_argnames=("horizon",))
def _run_eval_scan(carry: EvalCarry, *, horizon: int):
    final, _ = jax.lax.scan(_eval_step, carry, xs=None, length=horizon)
    return final


def greedy_win_rate_vs_random(
    params: dict,
    pool: Any,
    *,
    num_envs: int = 64,
    horizon: int = 1200,
    seed: int = 0,
) -> dict[str, float]:
    """Run num_envs greedy-vs-legal_random games to the horizon cap.

    Returns win/loss counts against games decided before the cap plus the
    undecided-at-cap share. Diagnostic only.
    """
    key = jax.random.PRNGKey(seed)
    init_idx = jnp.arange(num_envs, dtype=jnp.int32)
    states = jax.tree_util.tree_map(lambda x: x[init_idx], pool)
    pool_cursor = jnp.full((num_envs,), num_envs, dtype=jnp.int32)
    mem0 = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_envs), empty_memory())
    mem1 = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_envs), empty_memory())
    carry = EvalCarry(
        states, mem0, mem1, key, params, pool, pool_cursor,
        jnp.int32(0), jnp.int32(0), jnp.zeros((num_envs,), dtype=jnp.bool_),
    )
    final = _run_eval_scan(carry, horizon=horizon)
    wins = int(final.wins)
    losses = int(final.losses)
    decided = wins + losses
    return {
        "games": num_envs,
        "wins": wins,
        "losses": losses,
        "decided": decided,
        "win_rate_vs_decided": wins / decided if decided else 0.0,
        "draw_share_at_cap": (num_envs - decided) / num_envs,
    }
