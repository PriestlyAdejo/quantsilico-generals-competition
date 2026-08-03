"""Shared observation tensor encoding for learned policies."""

from __future__ import annotations

import torch
from torch import Tensor

from generals_bot.observation import Observation
from generals_bot.protocol import OWNER_ME, OWNER_OPP, TYPE_FOG, TYPE_MOUNTAIN

# Channels: visibility, plain, mountain, castle, general, fog_struct,
# owner_me, owner_opp, army_norm, padding
NUM_CELL_CHANNELS = 10
MAX_HW = 21


def encode_observation(observation: Observation, device: torch.device | None = None) -> Tensor:
    """Return float tensor shaped (C, MAX_HW, MAX_HW) with mountain padding."""
    device = device or torch.device("cpu")
    grid = torch.zeros(NUM_CELL_CHANNELS, MAX_HW, MAX_HW, dtype=torch.float32, device=device)
    # Default padding as mountains / invisible
    grid[2, :, :] = 1.0
    grid[9, :, :] = 1.0
    h, w = observation.height, observation.width
    for r in range(h):
        for c in range(w):
            grid[9, r, c] = 0.0
            t = observation.type_grid[r][c]
            o = observation.owner_grid[r][c]
            a = observation.army_grid[r][c]
            if t != TYPE_FOG:
                grid[0, r, c] = 1.0
            if t == 1:
                grid[1, r, c] = 1.0
            if t == TYPE_MOUNTAIN:
                grid[2, r, c] = 1.0
            else:
                grid[2, r, c] = 0.0
            if t == 3:
                grid[3, r, c] = 1.0
            if t == 4:
                grid[4, r, c] = 1.0
            if t == 5:
                grid[5, r, c] = 1.0
            if o == OWNER_ME:
                grid[6, r, c] = 1.0
            if o == OWNER_OPP:
                grid[7, r, c] = 1.0
            grid[8, r, c] = min(a, 1000) / 1000.0
    return grid


def encode_globals(observation: Observation, device: torch.device | None = None) -> Tensor:
    device = device or torch.device("cpu")
    turn_frac = observation.turn / 1200.0
    return torch.tensor(
        [
            turn_frac,
            float(observation.turn % 2),
            float(observation.turn % 50) / 50.0,
            observation.my_land / 500.0,
            observation.my_army / 5000.0,
            observation.opp_land / 500.0,
            observation.opp_army / 5000.0,
            (observation.my_land - observation.opp_land) / 500.0,
            (observation.my_army - observation.opp_army) / 5000.0,
        ],
        dtype=torch.float32,
        device=device,
    )
