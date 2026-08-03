"""Shared observation tensor encoding for learned policies.

Hot path uses NumPy vectorisation into fixed 21×21 padded tensors.
A slow reference encoder is retained for equivalence tests only.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from torch import Tensor

from generals_bot.observation import Observation
from generals_bot.protocol import OWNER_ME, OWNER_OPP, TYPE_FOG, TYPE_MOUNTAIN

# Channels: visibility, plain, mountain, castle, general, fog_struct,
# owner_me, owner_opp, army_norm, padding
NUM_CELL_CHANNELS = 10
MAX_HW = 21
GLOBAL_DIM = 9


@lru_cache(maxsize=8)
def padding_mask(height: int, width: int) -> np.ndarray:
    """Boolean mask True on padded cells outside HxW (cached by board size)."""
    mask = np.ones((MAX_HW, MAX_HW), dtype=bool)
    mask[:height, :width] = False
    return mask


@lru_cache(maxsize=1)
def coordinate_grids() -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices((MAX_HW, MAX_HW), dtype=np.int32)
    return rows, cols


@lru_cache(maxsize=1)
def neighbour_index_tables() -> dict[str, np.ndarray]:
    """Cached N/S/E/W neighbour flat indices for the 21×21 graph (-1 = invalid)."""
    rows, cols = coordinate_grids()
    flat = rows * MAX_HW + cols
    north = np.full((MAX_HW, MAX_HW), -1, dtype=np.int32)
    south = np.full((MAX_HW, MAX_HW), -1, dtype=np.int32)
    west = np.full((MAX_HW, MAX_HW), -1, dtype=np.int32)
    east = np.full((MAX_HW, MAX_HW), -1, dtype=np.int32)
    north[1:, :] = flat[:-1, :]
    south[:-1, :] = flat[1:, :]
    west[:, 1:] = flat[:, :-1]
    east[:, :-1] = flat[:, 1:]
    return {
        "north": north,
        "south": south,
        "west": west,
        "east": east,
        "self": flat,
    }


def encode_observation_reference(
    observation: Observation,
    device: torch.device | None = None,
) -> Tensor:
    """Slow per-cell reference encoder for equivalence tests only."""
    device = device or torch.device("cpu")
    grid = torch.zeros(NUM_CELL_CHANNELS, MAX_HW, MAX_HW, dtype=torch.float32, device=device)
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


def encode_grids_batch_numpy(
    type_batch: np.ndarray,
    owner_batch: np.ndarray,
    army_batch: np.ndarray,
) -> np.ndarray:
    """Vectorised encode for equal-sized boards → (B, C, MAX_HW, MAX_HW)."""
    batch, h, w = type_batch.shape
    out = np.zeros((batch, NUM_CELL_CHANNELS, MAX_HW, MAX_HW), dtype=np.float32)
    out[:, 2, :, :] = 1.0
    out[:, 9, :, :] = 1.0
    out[:, 9, :h, :w] = 0.0
    tg = type_batch.astype(np.int32, copy=False)
    og = owner_batch.astype(np.int32, copy=False)
    ag = army_batch.astype(np.int32, copy=False)
    out[:, 0, :h, :w] = (tg != TYPE_FOG).astype(np.float32)
    out[:, 1, :h, :w] = (tg == 1).astype(np.float32)
    out[:, 2, :h, :w] = (tg == TYPE_MOUNTAIN).astype(np.float32)
    out[:, 3, :h, :w] = (tg == 3).astype(np.float32)
    out[:, 4, :h, :w] = (tg == 4).astype(np.float32)
    out[:, 5, :h, :w] = (tg == 5).astype(np.float32)
    out[:, 6, :h, :w] = (og == OWNER_ME).astype(np.float32)
    out[:, 7, :h, :w] = (og == OWNER_OPP).astype(np.float32)
    out[:, 8, :h, :w] = np.minimum(ag, 1000).astype(np.float32) / 1000.0
    return out


def encode_grids_numpy(
    type_grid: np.ndarray,
    owner_grid: np.ndarray,
    army_grid: np.ndarray,
) -> np.ndarray:
    """Vectorised encode from HxW int arrays → (C, MAX_HW, MAX_HW) float32."""
    return encode_grids_batch_numpy(
        type_grid[np.newaxis],
        owner_grid[np.newaxis],
        army_grid[np.newaxis],
    )[0]


def encode_observation(observation: Observation, device: torch.device | None = None) -> Tensor:
    """Fast vectorised observation encode → (C, MAX_HW, MAX_HW)."""
    device = device or torch.device("cpu")
    type_grid = np.asarray(observation.type_grid, dtype=np.int32)
    owner_grid = np.asarray(observation.owner_grid, dtype=np.int32)
    army_grid = np.asarray(observation.army_grid, dtype=np.int32)
    arr = encode_grids_numpy(type_grid, owner_grid, army_grid)
    tensor = torch.from_numpy(np.ascontiguousarray(arr))
    if device.type != "cpu":
        tensor = tensor.to(device, non_blocking=True)
    return tensor


def encode_globals_numpy(observation: Observation) -> np.ndarray:
    turn_frac = observation.turn / 1200.0
    return np.asarray(
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
        dtype=np.float32,
    )


def encode_globals(observation: Observation, device: torch.device | None = None) -> Tensor:
    device = device or torch.device("cpu")
    arr = encode_globals_numpy(observation)
    tensor = torch.from_numpy(arr)
    if device.type != "cpu":
        tensor = tensor.to(device, non_blocking=True)
    return tensor


def encode_batch_from_arrays(
    type_batch: np.ndarray,
    owner_batch: np.ndarray,
    army_batch: np.ndarray,
    globals_batch: np.ndarray,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Encode a batch of equal HxW boards.

    type/owner/army: (B, H, W) int32
    globals_batch: (B, GLOBAL_DIM) float32
    returns cells (B, C, MAX_HW, MAX_HW), globals (B, GLOBAL_DIM)
    """
    device = device or torch.device("cpu")
    cells = encode_grids_batch_numpy(type_batch, owner_batch, army_batch)
    cell_t = torch.from_numpy(np.ascontiguousarray(cells))
    glob_t = torch.from_numpy(np.ascontiguousarray(np.asarray(globals_batch, dtype=np.float32)))
    if device.type != "cpu":
        cell_t = cell_t.to(device, non_blocking=True)
        glob_t = glob_t.to(device, non_blocking=True)
    return cell_t, glob_t
