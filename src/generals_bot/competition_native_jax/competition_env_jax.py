"""Competition-mode functional JAX wrappers over official MIT primitives.

Hot-path reuse of pinned `third_party/generals-bots` JAX kernels is intentional:
game.step, build_castles, deathtouch, get_observation, compute_valid_move_mask.

QuantSilico owns: 3970 legal-mask packing, obs/memory channel layout, scan collect.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from generals.core import game
from generals.core.action import compute_valid_move_mask
from generals.core.grid import generate_grid
from generals.modifiers import build_castles as _bc
from generals.modifiers import deathtouch as _dt

from generals_bot.competition_native_jax.constants import (
    ACTION_DIM,
    DEATHTOUCH_TURN,
    DRAW_TURN,
    MAX_HW,
    PASS_INDEX,
)

# Competition truncation (official mode preset)
TRUNCATION = DRAW_TURN  # 1200


class ObsMemoryJax(NamedTuple):
    """Device-resident deterministic observation memory (per player)."""

    seen_own: jnp.ndarray  # [H,W] float32
    last_army: jnp.ndarray  # [H,W] float32


def empty_memory() -> ObsMemoryJax:
    z = jnp.zeros((MAX_HW, MAX_HW), dtype=jnp.float32)
    return ObsMemoryJax(seen_own=z, last_army=z)


@jax.jit
def competition_transition(state: game.GameState, actions: jnp.ndarray):
    """Official competition composition: builds then deathtouch(base step).

    Mirrors competition/matchup.make_transition for mode=competition.
    """
    state, actions = _bc.apply_build_actions(state, actions)
    return _dt.step(state, actions, DEATHTOUCH_TURN)


@jax.jit
def step_one_jax(state: game.GameState, actions: jnp.ndarray):
    """One env step -> (next_state, reward[2], terminated, truncated, info)."""
    new_state, info = competition_transition(state, actions)
    reward_p0 = jnp.where(info.winner == 0, 1.0, jnp.where(info.winner == 1, -1.0, 0.0))
    rewards = jnp.stack([reward_p0, -reward_p0])
    terminated = info.is_done
    truncated = (new_state.time >= TRUNCATION) & (~terminated)
    return new_state, rewards, terminated, truncated, info


def reset_one_jax(
    key: jax.Array,
    height: int = MAX_HW,
    width: int = MAX_HW,
    *,
    min_generals_distance: int = 17,
) -> game.GameState:
    """Generate one competition board (no neutral castles), padded to pad_to via generate_grid."""
    key, kg = jax.random.split(key)
    # Official competition mountain/castle ranges from env preset
    grid = generate_grid(
        kg,
        grid_dims=(height, width),
        mountain_density_range=(0.24, 0.26),
        num_castles_range=(9, 11),
        min_generals_distance=min_generals_distance,
        castle_val_range=(20, 26),
        pad_to=MAX_HW,
    )
    grid = _bc.strip_neutral_castles(grid)
    return game.create_initial_state(grid.astype(jnp.int32))


def reset_batch_jax(keys: jax.Array, height: int = MAX_HW, width: int = MAX_HW) -> game.GameState:
    return jax.vmap(lambda k: reset_one_jax(k, height, width))(keys)


def update_memory(memory: ObsMemoryJax, obs) -> ObsMemoryJax:
    owned = obs.owned_cells.astype(jnp.float32)
    armies = obs.armies.astype(jnp.float32)
    h, w = armies.shape
    seen = memory.seen_own.at[:h, :w].max(owned)
    last = jnp.where(owned > 0, armies, memory.last_army[:h, :w])
    last_full = memory.last_army.at[:h, :w].set(last)
    return ObsMemoryJax(seen_own=seen, last_army=last_full)


def observe_one_jax(state: game.GameState, player: int, memory: ObsMemoryJax):
    """Official fog observation -> (spatial[8,21,21], global[8], new_memory)."""
    obs = game.get_observation(state, player)
    memory = update_memory(memory, obs)
    armies = obs.armies.astype(jnp.float32)
    owned = obs.owned_cells.astype(jnp.float32)
    opponent = obs.opponent_cells.astype(jnp.float32)
    mountains = obs.mountains.astype(jnp.float32)
    h, w = armies.shape
    spatial = jnp.zeros((8, MAX_HW, MAX_HW), dtype=jnp.float32)
    spatial = spatial.at[0, :h, :w].set(1.0)
    spatial = spatial.at[1, :h, :w].set(owned)
    spatial = spatial.at[2, :h, :w].set(opponent)
    spatial = spatial.at[3, :h, :w].set(armies / 100.0)
    spatial = spatial.at[4, :h, :w].set(mountains)
    spatial = spatial.at[5].set(memory.seen_own)
    spatial = spatial.at[6].set(memory.last_army / 100.0)
    # Channel 7: normalised castle cost floor map (own structures via official grid)
    cost = _bc.build_cost_grid(state, player).astype(jnp.float32)
    spatial = spatial.at[7, :h, :w].set(jnp.clip(cost / 100.0, 0.0, 2.0))

    turn = state.time.astype(jnp.float32)
    global_vec = jnp.array(
        [
            turn / float(TRUNCATION),
            (turn % 50.0) / 50.0,
            jnp.clip((float(DEATHTOUCH_TURN) - turn) / float(DEATHTOUCH_TURN), -1.0, 1.0),
            jnp.where(turn >= float(DEATHTOUCH_TURN), 1.0, 0.0),
            jnp.maximum(0.0, turn - float(DEATHTOUCH_TURN)) / float(TRUNCATION),
            jnp.clip((float(TRUNCATION) - turn) / float(TRUNCATION), 0.0, 1.0),
            h / float(MAX_HW),
            w / float(MAX_HW),
        ],
        dtype=jnp.float32,
    )
    return spatial, global_vec, memory


def legal_mask_one_jax(state: game.GameState, player: int) -> jnp.ndarray:
    """Exact 3970 legal mask: PASS + moves + BUILD with official castle cost."""
    # player is static for JIT via dedicated seat helpers below
    return _legal_mask_player(state, player)


def _legal_mask_player(state: game.GameState, player: int) -> jnp.ndarray:
    obs = game.get_observation(state, player)
    mask = jnp.zeros((ACTION_DIM,), dtype=bool).at[PASS_INDEX].set(True)
    move_mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    h, w = obs.armies.shape
    cells = jnp.arange(MAX_HW * MAX_HW)
    rows = cells // MAX_HW
    cols = cells % MAX_HW
    in_board = (rows < h) & (cols < w)
    base = 1 + cells * 9

    for d in range(4):
        legal = jnp.zeros((MAX_HW, MAX_HW), dtype=bool)
        legal = legal.at[:h, :w].set(move_mask[:, :, d])
        flat = legal.reshape(-1) & in_board
        mask = mask.at[base + d * 2].set(flat)
        mask = mask.at[base + d * 2 + 1].set(flat)

    cost = _bc.build_cost_grid(state, player)
    can_build = jnp.zeros((MAX_HW, MAX_HW), dtype=bool)
    can_build = can_build.at[:h, :w].set(
        obs.owned_cells
        & (obs.armies >= cost)
        & (~obs.mountains)
        & (~obs.generals)
        & (~obs.castles)
        & (~obs.fog_cells)
    )
    mask = mask.at[base + 8].set(can_build.reshape(-1) & in_board)
    return mask


legal_mask_one_p0 = jax.jit(lambda s: _legal_mask_player(s, 0))
legal_mask_one_p1 = jax.jit(lambda s: _legal_mask_player(s, 1))


def index_to_engine_action(idx: jnp.ndarray) -> jnp.ndarray:
    """Scalar action index -> [5] engine action [kind,row,col,dir,split]."""
    # Inline arithmetic (not module-level JAX arrays) so jit/vmap cannot leak tracers.
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


# Batched APIs
step_batch_jax = jax.jit(jax.vmap(step_one_jax, in_axes=(0, 0)))
observe_batch_jax = jax.jit(jax.vmap(observe_one_jax, in_axes=(0, None, 0)))
# player is scalar broadcast — use two dedicated helpers for seats
observe_batch_p0 = jax.jit(jax.vmap(lambda s, m: observe_one_jax(s, 0, m)))
observe_batch_p1 = jax.jit(jax.vmap(lambda s, m: observe_one_jax(s, 1, m)))
legal_mask_batch_p0 = jax.jit(jax.vmap(legal_mask_one_p0))
legal_mask_batch_p1 = jax.jit(jax.vmap(legal_mask_one_p1))
index_to_engine_action_batch = jax.jit(jax.vmap(index_to_engine_action))


def _make_pool_batch(keys: jax.Array, h: int, w: int, *, min_generals_distance: int = 17) -> game.GameState:
    """JIT-friendly batch of same-sized boards via canonical reset_one_jax."""
    return _make_pool_batch_cached(h, w, min_generals_distance)(keys)


def _make_pool_batch_cached(h: int, w: int, min_generals_distance: int = 17):
    @jax.jit
    def _fn(keys: jax.Array) -> game.GameState:
        return jax.vmap(
            lambda k: reset_one_jax(k, height=h, width=w, min_generals_distance=min_generals_distance)
        )(keys)

    return _fn


def build_competition_reset_pool(
    key: jax.Array,
    pool_size: int,
    *,
    min_grid: int = 18,
    max_grid: int = 21,
    min_generals_distance: int = 17,
) -> game.GameState:
    """Pregenerate competition initial states using exact canonical reset semantics.

    Mirrors official GeneralsEnv.reset pool construction (size combos, concat, shuffle).
    Each entry is produced by ``reset_one_jax`` — no extra post-process beyond that path.
    ``min_generals_distance`` defaults to the competition value (17); curriculum
    research arms may override it (PPO_SEMANTICS UNCHANGED: map generation only).
    """
    k_pool, k_shuffle = jax.random.split(key)
    sizes = [(h, w) for h in range(min_grid, max_grid + 1) for w in range(min_grid, max_grid + 1)]
    num_combos = len(sizes)
    per_combo = max(1, int(pool_size) // num_combos)
    actual = num_combos * per_combo
    pool_keys = jax.random.split(k_pool, actual)
    pools = []
    for i, (h, w) in enumerate(sizes):
        combo_keys = pool_keys[i * per_combo : (i + 1) * per_combo]
        # static h,w per compiled specialty via python loop over combos (outside scan)
        pools.append(
            _make_pool_batch(combo_keys, h, w, min_generals_distance=min_generals_distance)
        )
    pool = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *pools)
    perm = jax.random.permutation(k_shuffle, actual)
    return jax.tree_util.tree_map(lambda x: x[perm], pool)


def auto_reset_from_pool(
    states: game.GameState,
    terminated: jnp.ndarray,
    truncated: jnp.ndarray,
    pool: game.GameState,
    cursor: jnp.ndarray,
) -> tuple[game.GameState, jnp.ndarray]:
    """Indexed competition reset from a pregenerated pool (no map generation).

    cursor: int32 [N] — advanced by one on each done environment.
    """
    done = terminated | truncated
    pool_size = jax.tree_util.tree_leaves(pool)[0].shape[0]
    idx = cursor % pool_size
    fresh = jax.tree_util.tree_map(lambda x: x[idx], pool)
    new_cursor = cursor + done.astype(cursor.dtype)

    def pick(old, new):
        mask = done.reshape((-1,) + (1,) * (old.ndim - 1))
        return jnp.where(mask, new, old)

    return jax.tree_util.tree_map(pick, states, fresh), new_cursor


def auto_reset_batch(
    states: game.GameState,
    terminated: jnp.ndarray,
    truncated: jnp.ndarray,
    keys: jax.Array,
) -> game.GameState:
    """Legacy in-scan generate_grid reset (compat / tests). Prefer auto_reset_from_pool."""
    done = terminated | truncated
    fresh = reset_batch_jax(keys)

    def pick(old, new):
        mask = done.reshape((-1,) + (1,) * (old.ndim - 1))
        return jnp.where(mask, new, old)

    return jax.tree_util.tree_map(pick, states, fresh)
