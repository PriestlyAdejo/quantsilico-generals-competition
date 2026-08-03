"""Legal-action enumeration from a partial observation."""

from __future__ import annotations

from generals_bot.action import PASS_ACTION, Action
from generals_bot.castle_cost import castle_price_at, own_structures
from generals_bot.observation import Observation
from generals_bot.protocol import (
    DIRECTIONS,
    OWNER_ME,
    TYPE_CASTLE,
    TYPE_FOG,
    TYPE_GENERAL,
    TYPE_MOUNTAIN,
    TYPE_PLAIN,
    TYPE_STRUCTURE_IN_FOG,
)


def is_passable(cell_type: int) -> bool:
    """Return whether a destination type can be entered."""
    return cell_type not in (TYPE_MOUNTAIN, TYPE_STRUCTURE_IN_FOG)


def is_buildable_cell(observation: Observation, row: int, col: int) -> bool:
    """True if the cell is owned plain land (not general/castle)."""
    if observation.owner_grid[row][col] != OWNER_ME:
        return False
    cell_type = observation.type_grid[row][col]
    return cell_type == TYPE_PLAIN


def enumerate_legal_actions(observation: Observation) -> list[Action]:
    """Enumerate all legal moves, builds and the pass action.

    Moves into fog are allowed when the destination is not a known mountain
    or structure-in-fog. Builds use prices from visible owned structures.
    """
    actions: list[Action] = [PASS_ACTION]
    structures = own_structures(observation)
    height, width = observation.height, observation.width

    for r in range(height):
        for c in range(width):
            if observation.owner_grid[r][c] != OWNER_ME:
                continue
            army = observation.army_grid[r][c]
            cell_type = observation.type_grid[r][c]

            if army > 1:
                for direction, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    dest_type = observation.type_grid[nr][nc]
                    if not is_passable(dest_type):
                        continue
                    actions.append(Action.move(r, c, direction, split=0))
                    actions.append(Action.move(r, c, direction, split=1))

            if cell_type == TYPE_PLAIN:
                price = castle_price_at(r, c, structures)
                if army >= price:
                    actions.append(Action.build(r, c))
            elif cell_type in (TYPE_GENERAL, TYPE_CASTLE, TYPE_FOG):
                # Owned cells are visible; fog here should not occur for own.
                pass

    return actions


def is_legal_action(observation: Observation, action: Action) -> bool:
    """Return whether ``action`` is in the legal set for ``observation``."""
    if action.kind == PASS_ACTION.kind:
        return (
            action.row == 0
            and action.col == 0
            and action.direction == 0
            and action.split == 0
        ) or action == PASS_ACTION

    height, width = observation.height, observation.width
    if action.kind == Action.move(0, 0, 0).kind:
        r, c = action.row, action.col
        if not (0 <= r < height and 0 <= c < width):
            return False
        if observation.owner_grid[r][c] != OWNER_ME:
            return False
        if observation.army_grid[r][c] <= 1:
            return False
        if action.split not in (0, 1):
            return False
        if action.direction not in range(4):
            return False
        dr, dc = DIRECTIONS[action.direction]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < height and 0 <= nc < width):
            return False
        return is_passable(observation.type_grid[nr][nc])

    if action.kind == Action.build(0, 0).kind:
        r, c = action.row, action.col
        if not (0 <= r < height and 0 <= c < width):
            return False
        if not is_buildable_cell(observation, r, c):
            return False
        price = castle_price_at(r, c, own_structures(observation))
        return observation.army_grid[r][c] >= price

    return False
