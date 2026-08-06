"""JAX-native self-play rollout helpers using GeneralsEnv + scanned policy."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from generals import GeneralsEnv
from generals.core import game
from generals.core.action import compute_valid_move_mask

from generals_bot.competition_native_jax.constants import ACTION_DIM, MAX_HW, PASS_INDEX
from generals_bot.competition_native_jax.inference_jax import masked_log_softmax, sample_action
from generals_bot.competition_native_jax.transformer_jax import forward


def engine_obs_to_spatial_global(obs, player: int = 0) -> tuple[jax.Array, jax.Array]:
    """Convert engine observation to padded spatial/global tensors (JAX)."""
    # obs fields: armies, owned_cells, mountains, etc. as JAX arrays [H,W] padded
    armies = obs.armies.astype(jnp.float32)
    owned = obs.owned_cells.astype(jnp.float32)
    mountains = obs.mountains.astype(jnp.float32)
    h, w = armies.shape
    spatial = jnp.zeros((8, MAX_HW, MAX_HW), dtype=jnp.float32)
    playable = jnp.zeros((MAX_HW, MAX_HW), dtype=jnp.float32)
    playable = playable.at[:h, :w].set(1.0)
    spatial = spatial.at[0].set(playable)
    spatial = spatial.at[1, :h, :w].set((owned == 1).astype(jnp.float32))
    spatial = spatial.at[2, :h, :w].set((owned == 2).astype(jnp.float32))
    spatial = spatial.at[3, :h, :w].set(armies / 100.0)
    spatial = spatial.at[4, :h, :w].set(mountains)
    turn = jnp.array(getattr(obs, "turn", 0), dtype=jnp.float32)
    if hasattr(obs, "timestep"):
        turn = obs.timestep.astype(jnp.float32)
    global_vec = jnp.array(
        [
            turn / 1200.0,
            (turn % 50) / 50.0,
            jnp.clip((800.0 - turn) / 800.0, -1.0, 1.0),
            (turn >= 800).astype(jnp.float32),
            jnp.maximum(0.0, turn - 800.0) / 1200.0,
            jnp.clip((1200.0 - turn) / 1200.0, 0.0, 1.0),
            h / float(MAX_HW),
            w / float(MAX_HW),
        ],
        dtype=jnp.float32,
    )
    return spatial, global_vec


def legal_mask_from_engine_obs(obs) -> jax.Array:
    """Approximate full-support mask: PASS + engine move mask + build on owned plain."""
    mask = jnp.zeros((ACTION_DIM,), dtype=bool)
    mask = mask.at[PASS_INDEX].set(True)
    move_mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    # move_mask shape typically [H,W,4] or similar — adapt to flat layout
    # Official mask: (H, W, 4) directions; we also have splits → duplicate
    if move_mask.ndim == 3:
        h, w, nd = move_mask.shape
        for r in range(h):
            for c in range(w):
                cell = r * MAX_HW + c
                base = 1 + cell * 9
                for d in range(min(nd, 4)):
                    for s in range(2):
                        mask = mask.at[base + d * 2 + s].set(move_mask[r, c, d])
                # build: owned and army >= 35 and not mountain
                can_build = (obs.owned_cells[r, c] == 1) & (obs.armies[r, c] >= 35) & (~obs.mountains[r, c].astype(bool))
                mask = mask.at[base + 8].set(can_build)
    return mask


def index_to_engine_action(idx: jax.Array) -> jax.Array:
    """Map flat policy index to [kind,row,col,dir,split]."""
    is_pass = idx == 0
    flat = idx - 1
    cell = flat // 9
    local = flat % 9
    row = cell // MAX_HW
    col = cell % MAX_HW
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


def collect_selfplay_batch(
    params: dict,
    *,
    num_envs: int = 4,
    rollout_len: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Collect symmetric self-play transitions (JAX policy; env via GeneralsEnv).

    Uses device-resident arrays for stored trajectories. Env reset/step use the
    official competition simulator. Policy forward and sampling are JAX.
    """
    env = GeneralsEnv(mode="competition")
    key = jax.random.PRNGKey(seed)
    spatials = []
    globals_ = []
    masks = []
    actions = []
    old_logps = []
    values = []
    rewards = []
    dones = []

    # Initialize envs sequentially then stack — first version; later vmap pool
    pools_states = []
    for i in range(num_envs):
        key, k = jax.random.split(key)
        pool, state = env.reset(k)
        pools_states.append((pool, state))

    for t in range(rollout_len):
        batch_sp0 = []
        batch_gv0 = []
        batch_m0 = []
        batch_a0 = []
        batch_lp0 = []
        batch_v0 = []
        batch_sp1 = []
        batch_a1 = []
        step_rewards = []
        step_dones = []
        new_pools_states = []
        for i, (pool, state) in enumerate(pools_states):
            key, k0, k1 = jax.random.split(key, 3)
            obs0 = game.get_observation(state, 0)
            obs1 = game.get_observation(state, 1)
            sp0, gv0 = engine_obs_to_spatial_global(obs0)
            sp1, gv1 = engine_obs_to_spatial_global(obs1)
            m0 = legal_mask_from_engine_obs(obs0)
            m1 = legal_mask_from_engine_obs(obs1)
            out0 = forward(params, sp0, gv0)
            out1 = forward(params, sp1, gv1)
            a0, lp0 = sample_action(k0, out0["flat_logits"], m0)
            a1, lp1 = sample_action(k1, out1["flat_logits"], m1)
            v0 = jnp.sum(jax.nn.softmax(out0["value_logits"]) * jnp.linspace(-1, 1, out0["value_logits"].shape[0]))
            eng0 = index_to_engine_action(a0)
            eng1 = index_to_engine_action(a1)
            actions_pair = jnp.stack([eng0, eng1])
            ts, new_state = env.step(state, actions_pair, pool)
            # reward from perspective of both; store player0 trajectory primarily
            r = ts.reward[0]
            done = jnp.logical_or(ts.terminated, ts.truncated).astype(jnp.float32)
            batch_sp0.append(sp0)
            batch_gv0.append(gv0)
            batch_m0.append(m0)
            batch_a0.append(a0)
            batch_lp0.append(lp0)
            batch_v0.append(v0)
            step_rewards.append(r)
            step_dones.append(done)
            new_pools_states.append((pool, new_state))
        pools_states = new_pools_states
        spatials.append(jnp.stack(batch_sp0))
        globals_.append(jnp.stack(batch_gv0))
        masks.append(jnp.stack(batch_m0))
        actions.append(jnp.stack(batch_a0))
        old_logps.append(jnp.stack(batch_lp0))
        values.append(jnp.stack(batch_v0))
        rewards.append(jnp.stack(step_rewards))
        dones.append(jnp.stack(step_dones))

    # shapes [T, N, ...]
    return {
        "spatial": jnp.stack(spatials),
        "global": jnp.stack(globals_),
        "mask": jnp.stack(masks),
        "actions": jnp.stack(actions),
        "old_logp": jnp.stack(old_logps),
        "values": jnp.stack(values),
        "rewards": jnp.stack(rewards),
        "dones": jnp.stack(dones),
        "backend": "jax_policy_generalsenv_selfplay",
    }
