"""Deterministic observation memory and channel packing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from generals_bot.competition_native_jax.castles import padded_price_map
from generals_bot.competition_native_jax.constants import MAX_HW
from generals_bot.competition_native_jax.deathtouch import horizon_features
from generals_bot.observation import Observation
from generals_bot.protocol import (
    OWNER_ME,
    OWNER_OPP,
    TYPE_CASTLE,
    TYPE_FOG,
    TYPE_GENERAL,
    TYPE_MOUNTAIN,
    TYPE_STRUCTURE_IN_FOG,
)


N_SPATIAL = 8
N_GLOBAL = 8
N_SPATIAL_V2 = 14
N_GLOBAL_V2 = 12


@dataclass
class ObsMemory:
    """Player-specific deterministic memory (seen land / last army)."""

    height: int = MAX_HW
    width: int = MAX_HW
    seen_own: np.ndarray = field(default_factory=lambda: np.zeros((MAX_HW, MAX_HW), dtype=np.float32))
    last_army: np.ndarray = field(default_factory=lambda: np.zeros((MAX_HW, MAX_HW), dtype=np.float32))
    turn: int = 0

    def reset(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.seen_own.fill(0.0)
        self.last_army.fill(0.0)
        self.turn = 0

    def update(self, observation: Observation) -> None:
        self.turn = int(getattr(observation, "turn", self.turn))
        h, w = observation.height, observation.width
        for r in range(h):
            for c in range(w):
                if observation.owner_grid[r][c] == OWNER_ME:
                    self.seen_own[r, c] = 1.0
                    self.last_army[r, c] = float(observation.army_grid[r][c])


def encode_observation(observation: Observation, memory: ObsMemory) -> tuple[np.ndarray, np.ndarray]:
    """Return spatial [C,21,21] and global [G] float32 tensors."""
    memory.update(observation)
    spatial = np.zeros((N_SPATIAL, MAX_HW, MAX_HW), dtype=np.float32)
    h, w = observation.height, observation.width
    playable = np.zeros((MAX_HW, MAX_HW), dtype=np.float32)
    playable[:h, :w] = 1.0
    # Outside playable area marked as mountain padding
    spatial[0] = playable
    for r in range(h):
        for c in range(w):
            spatial[1, r, c] = 1.0 if observation.owner_grid[r][c] == OWNER_ME else 0.0
            spatial[2, r, c] = 1.0 if observation.owner_grid[r][c] == OWNER_OPP else 0.0
            spatial[3, r, c] = float(observation.army_grid[r][c]) / 100.0
            spatial[4, r, c] = 1.0 if observation.type_grid[r][c] == TYPE_MOUNTAIN else 0.0
    spatial[5] = memory.seen_own
    spatial[6] = memory.last_army / 100.0
    prices = padded_price_map(observation).astype(np.float32)
    spatial[7] = np.clip(prices / 100.0, 0.0, 2.0) * playable

    hz = horizon_features(memory.turn)
    global_vec = np.array(
        [
            hz["x_turn"],
            hz["x_turn_mod50"],
            hz["x_death_countdown"],
            hz["x_dt_active"],
            hz["x_turns_since_dt"],
            hz["x_cap_remaining"],
            float(h) / MAX_HW,
            float(w) / MAX_HW,
        ],
        dtype=np.float32,
    )
    return spatial, global_vec


@dataclass
class ObsMemoryV2:
    """OBS-V2 serving memory: v1 fields + enemy-seen + revealed enemy general."""

    height: int = MAX_HW
    width: int = MAX_HW
    seen_own: np.ndarray = field(default_factory=lambda: np.zeros((MAX_HW, MAX_HW), dtype=np.float32))
    last_army: np.ndarray = field(default_factory=lambda: np.zeros((MAX_HW, MAX_HW), dtype=np.float32))
    seen_enemy: np.ndarray = field(default_factory=lambda: np.zeros((MAX_HW, MAX_HW), dtype=np.float32))
    enemy_general_seen: np.ndarray = field(default_factory=lambda: np.zeros((MAX_HW, MAX_HW), dtype=np.float32))
    turn: int = 0

    def reset(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.seen_own.fill(0.0)
        self.last_army.fill(0.0)
        self.seen_enemy.fill(0.0)
        self.enemy_general_seen.fill(0.0)
        self.turn = 0


def encode_observation_v2(
    observation: Observation, memory: ObsMemoryV2
) -> tuple[np.ndarray, np.ndarray]:
    """OBS-V2 serving encoder -> spatial [14,21,21], global [12].

    Plane/global semantics are IDENTICAL to the training path
    (obs_v2_jax.observe_one_v2): both consume only fog-applied legal
    observation content.
    """
    memory.turn = int(getattr(observation, "turn", memory.turn))
    h, w = observation.height, observation.width
    spatial = np.zeros((N_SPATIAL_V2, MAX_HW, MAX_HW), dtype=np.float32)
    playable = np.zeros((MAX_HW, MAX_HW), dtype=np.float32)
    playable[:h, :w] = 1.0
    spatial[0] = playable
    for r in range(h):
        for c in range(w):
            owner = observation.owner_grid[r][c]
            cell_type = observation.type_grid[r][c]
            spatial[1, r, c] = 1.0 if owner == OWNER_ME else 0.0
            spatial[2, r, c] = 1.0 if owner == OWNER_OPP else 0.0
            spatial[3, r, c] = float(observation.army_grid[r][c]) / 100.0
            spatial[4, r, c] = 1.0 if cell_type == TYPE_MOUNTAIN else 0.0
            spatial[8, r, c] = 1.0 if cell_type == TYPE_FOG else 0.0
            spatial[9, r, c] = 1.0 if (cell_type == TYPE_GENERAL and owner == OWNER_OPP) else 0.0
            spatial[10, r, c] = 1.0 if cell_type == TYPE_CASTLE else 0.0
            spatial[11, r, c] = 1.0 if cell_type == TYPE_STRUCTURE_IN_FOG else 0.0
            if owner == OWNER_ME:
                memory.seen_own[r, c] = 1.0
                memory.last_army[r, c] = float(observation.army_grid[r][c])
            if owner == OWNER_OPP:
                memory.seen_enemy[r, c] = 1.0
            if cell_type == TYPE_GENERAL and owner == OWNER_OPP:
                memory.enemy_general_seen[r, c] = 1.0
    spatial[5] = memory.seen_own
    spatial[6] = memory.last_army / 100.0
    prices = padded_price_map(observation).astype(np.float32)
    spatial[7] = np.clip(prices / 100.0, 0.0, 2.0) * playable
    spatial[12] = memory.enemy_general_seen
    spatial[13] = memory.seen_enemy

    hz = horizon_features(memory.turn)
    area = float(MAX_HW * MAX_HW)
    global_vec = np.array(
        [
            hz["x_turn"],
            hz["x_turn_mod50"],
            hz["x_death_countdown"],
            hz["x_dt_active"],
            hz["x_turns_since_dt"],
            hz["x_cap_remaining"],
            float(h) / MAX_HW,
            float(w) / MAX_HW,
            float(observation.my_land) / area,
            float(observation.my_army) / (area * 10.0),
            float(observation.opp_land) / area,
            float(observation.opp_army) / (area * 10.0),
        ],
        dtype=np.float32,
    )
    return spatial, global_vec
