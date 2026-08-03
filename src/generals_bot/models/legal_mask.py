"""Legal-action mask tensors over the flat ACTION_DIM space."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from generals_bot.action import Action
from generals_bot.castle_cost import castle_price_at, own_structures
from generals_bot.legal import enumerate_legal_actions, is_passable
from generals_bot.models.action_index import (
    ACTION_DIM,
    PASS_INDEX,
    action_to_index,
    build_index,
    move_index,
)
from generals_bot.models.observation_encoder import MAX_HW
from generals_bot.observation import Observation
from generals_bot.protocol import DIRECTIONS, OWNER_ME, TYPE_PLAIN


def legal_mask_from_actions(actions: list[Action], device: torch.device | None = None) -> Tensor:
    device = device or torch.device("cpu")
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool, device=device)
    for action in actions:
        mask[action_to_index(action)] = True
    if not bool(mask.any()):
        mask[PASS_INDEX] = True
    return mask


def legal_mask_numpy(
    type_grid: np.ndarray,
    owner_grid: np.ndarray,
    army_grid: np.ndarray,
    structures: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build a boolean ACTION_DIM mask from HxW arrays (vectorised where practical)."""
    h, w = type_grid.shape
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[PASS_INDEX] = True

    owned = owner_grid == OWNER_ME
    movable = owned & (army_grid > 1)
    dirs = DIRECTIONS
    for r in range(h):
        for c in range(w):
            if not movable[r, c]:
                continue
            for direction, (dr, dc) in enumerate(dirs):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if not is_passable(int(type_grid[nr, nc])):
                    continue
                for split in (0, 1):
                    mask[move_index(r, c, direction, split)] = True

    if structures is None:
        structures = []
        for r in range(h):
            for c in range(w):
                if owned[r, c] and int(type_grid[r, c]) in (3, 4):  # castle/general
                    structures.append((r, c))

    plain_owned = owned & (type_grid == TYPE_PLAIN)
    for r in range(h):
        for c in range(w):
            if not plain_owned[r, c]:
                continue
            price = castle_price_at(r, c, structures)
            if int(army_grid[r, c]) >= price:
                mask[build_index(r, c)] = True
    return mask


def legal_mask_observation(
    observation: Observation,
    device: torch.device | None = None,
) -> Tensor:
    """Preferred path: enumerate via existing legal module for exact parity."""
    return legal_mask_from_actions(enumerate_legal_actions(observation), device=device)


def apply_action_mask(logits: Tensor, mask: Tensor) -> Tensor:
    """Mask illegal logits to -inf; guarantee at least pass is legal."""
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if not bool(mask.any(dim=-1).all()):
        mask = mask.clone()
        mask[..., PASS_INDEX] = True
    return logits.masked_fill(~mask, float("-inf"))


def select_legal_action(logits: Tensor, mask: Tensor) -> int:
    masked = apply_action_mask(logits, mask)
    return int(torch.argmax(masked, dim=-1).reshape(-1)[0].item())
