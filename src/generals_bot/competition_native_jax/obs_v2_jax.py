"""OBS-V2 objective-aware observation (EV-0071; obs_v2_r1_plan.yaml).

Additive parallel observation path - the canonical 8-plane path (v1) is the
bit-identical control and is untouched here. All features are LEGAL
observation content (fog-applied official observation / protocol frame); no
privileged information.

Added spatial planes over v1 (8 -> 14):
  P8  fog mask
  P9  visible enemy general (current tick)
  P10 visible castles (all visible castle cells)
  P11 structures-in-fog
  P12 revealed-enemy-general memory (persistent until episode reset)
  P13 enemy-seen memory (cells ever observed enemy-owned)
Added globals (8 -> 12): own land, own army, opponent land, opponent army
(all /441; legal scoreboard values; opponent totals include fogged cells as
the protocol lawfully reports them).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from generals_bot.competition_native_jax.constants import MAX_HW
from generals_bot.competition_native_jax.deathtouch import DEATHTOUCH_TURN
from generals_bot.competition_native_jax import competition_env_jax as _env
from generals.core import game

N_SPATIAL_V2 = 14
N_GLOBAL_V2 = 12
TRUNCATION = float(_env.TRUNCATION)


class ObsMemoryV2Jax(NamedTuple):
    """Device-resident OBS-V2 memory (per player)."""

    seen_own: jnp.ndarray      # [H,W] float32
    last_army: jnp.ndarray     # [H,W] float32
    seen_enemy: jnp.ndarray    # [H,W] float32 - ever observed enemy-owned
    enemy_general_seen: jnp.ndarray  # [H,W] float32 - revealed enemy general


def empty_memory_v2() -> ObsMemoryV2Jax:
    z = jnp.zeros((MAX_HW, MAX_HW), dtype=jnp.float32)
    return ObsMemoryV2Jax(seen_own=z, last_army=z, seen_enemy=z, enemy_general_seen=z)


def update_memory_v2(memory: ObsMemoryV2Jax, obs) -> ObsMemoryV2Jax:
    owned = obs.owned_cells.astype(jnp.float32)
    armies = obs.armies.astype(jnp.float32)
    opp = obs.opponent_cells.astype(jnp.float32)
    enemy_general = (obs.generals & obs.opponent_cells).astype(jnp.float32)
    h, w = armies.shape
    seen_own = memory.seen_own.at[:h, :w].max(owned)
    last = jnp.where(owned > 0, armies, memory.last_army[:h, :w])
    last_army = memory.last_army.at[:h, :w].set(last)
    seen_enemy = memory.seen_enemy.at[:h, :w].max(opp)
    enemy_general_seen = memory.enemy_general_seen.at[:h, :w].max(enemy_general)
    return ObsMemoryV2Jax(
        seen_own=seen_own,
        last_army=last_army,
        seen_enemy=seen_enemy,
        enemy_general_seen=enemy_general_seen,
    )


def observe_one_v2(state: game.GameState, player: int, memory: ObsMemoryV2Jax):
    """Official fog observation -> (spatial[14,21,21], global[12], new_memory)."""
    obs = game.get_observation(state, player)
    memory = update_memory_v2(memory, obs)
    armies = obs.armies.astype(jnp.float32)
    owned = obs.owned_cells.astype(jnp.float32)
    opponent = obs.opponent_cells.astype(jnp.float32)
    mountains = obs.mountains.astype(jnp.float32)
    fog = obs.fog_cells.astype(jnp.float32)
    castles = obs.castles.astype(jnp.float32)
    structures_in_fog = obs.structures_in_fog.astype(jnp.float32)
    enemy_general = (obs.generals & obs.opponent_cells).astype(jnp.float32)
    h, w = armies.shape

    spatial = jnp.zeros((N_SPATIAL_V2, MAX_HW, MAX_HW), dtype=jnp.float32)
    spatial = spatial.at[0, :h, :w].set(1.0)
    spatial = spatial.at[1, :h, :w].set(owned)
    spatial = spatial.at[2, :h, :w].set(opponent)
    spatial = spatial.at[3, :h, :w].set(armies / 100.0)
    spatial = spatial.at[4, :h, :w].set(mountains)
    spatial = spatial.at[5].set(memory.seen_own)
    spatial = spatial.at[6].set(memory.last_army / 100.0)
    cost = _env._bc.build_cost_grid(state, player).astype(jnp.float32)
    spatial = spatial.at[7, :h, :w].set(jnp.clip(cost / 100.0, 0.0, 2.0))
    # OBS-V2 additions
    spatial = spatial.at[8, :h, :w].set(fog)
    spatial = spatial.at[9, :h, :w].set(enemy_general)
    spatial = spatial.at[10, :h, :w].set(castles)
    spatial = spatial.at[11, :h, :w].set(structures_in_fog)
    spatial = spatial.at[12].set(memory.enemy_general_seen)
    spatial = spatial.at[13].set(memory.seen_enemy)

    turn = state.time.astype(jnp.float32)
    area = float(MAX_HW * MAX_HW)
    global_vec = jnp.array(
        [
            turn / TRUNCATION,
            (turn % 50.0) / 50.0,
            jnp.clip((float(DEATHTOUCH_TURN) - turn) / float(DEATHTOUCH_TURN), -1.0, 1.0),
            jnp.where(turn >= float(DEATHTOUCH_TURN), 1.0, 0.0),
            jnp.maximum(0.0, turn - float(DEATHTOUCH_TURN)) / TRUNCATION,
            jnp.clip((TRUNCATION - turn) / TRUNCATION, 0.0, 1.0),
            h / float(MAX_HW),
            w / float(MAX_HW),
            obs.owned_land_count.astype(jnp.float32) / area,
            obs.owned_army_count.astype(jnp.float32) / (area * 10.0),
            obs.opponent_land_count.astype(jnp.float32) / area,
            obs.opponent_army_count.astype(jnp.float32) / (area * 10.0),
        ],
        dtype=jnp.float32,
    )
    return spatial, global_vec, memory


observe_batch_v2_p0 = jax.jit(jax.vmap(lambda s, m: observe_one_v2(s, 0, m)))
observe_batch_v2_p1 = jax.jit(jax.vmap(lambda s, m: observe_one_v2(s, 1, m)))
