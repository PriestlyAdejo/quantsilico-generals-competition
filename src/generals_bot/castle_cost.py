"""Castle price calculation matching the official build_castles modifier."""

from __future__ import annotations

from generals_bot.observation import Observation
from generals_bot.protocol import OWNER_ME, TYPE_CASTLE, TYPE_GENERAL
from generals_bot.rules import (
    CASTLE_BASE_COST,
    CASTLE_PROXIMITY_DECAY,
    CASTLE_PROXIMITY_PENALTY,
)


def castle_price_at(
    row: int,
    col: int,
    structures: list[tuple[int, int]],
) -> int:
    """Price to build at ``(row, col)`` given owned structure coordinates."""
    cost = CASTLE_BASE_COST
    for sr, sc in structures:
        distance = abs(sr - row) + abs(sc - col)
        surcharge = CASTLE_PROXIMITY_PENALTY - CASTLE_PROXIMITY_DECAY * distance
        if surcharge > 0:
            cost += surcharge
    return cost


def own_structures(observation: Observation) -> list[tuple[int, int]]:
    """Return coordinates of visible owned generals and castles."""
    structures: list[tuple[int, int]] = []
    for r in range(observation.height):
        for c in range(observation.width):
            if observation.owner_grid[r][c] != OWNER_ME:
                continue
            cell_type = observation.type_grid[r][c]
            if cell_type in (TYPE_GENERAL, TYPE_CASTLE):
                structures.append((r, c))
    return structures


def castle_cost_grid(observation: Observation) -> list[list[int]]:
    """Full (H, W) castle price grid from the agent's visible own structures."""
    structures = own_structures(observation)
    return [
        [castle_price_at(r, c, structures) for c in range(observation.width)]
        for r in range(observation.height)
    ]
