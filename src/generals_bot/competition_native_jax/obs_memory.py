"""Deterministic observation memory and channel packing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from generals_bot.competition_native_jax.castles import padded_price_map
from generals_bot.competition_native_jax.constants import MAX_HW
from generals_bot.competition_native_jax.deathtouch import horizon_features
from generals_bot.observation import Observation
from generals_bot.protocol import OWNER_ME, OWNER_OPP, TYPE_MOUNTAIN


N_SPATIAL = 8
N_GLOBAL = 8


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
