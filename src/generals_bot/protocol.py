"""Wire-protocol constants, parsers and serializers for competition agents.

Agent-side inverse of ``third_party/generals-bots/competition/protocol.py``.
"""

from __future__ import annotations

from generals_bot.action import KIND_BUILD, KIND_MOVE, KIND_PASS, PASS_ACTION, Action
from generals_bot.observation import Observation

TYPE_FOG = 0
TYPE_PLAIN = 1
TYPE_MOUNTAIN = 2
TYPE_CASTLE = 3
TYPE_GENERAL = 4
TYPE_STRUCTURE_IN_FOG = 5

OWNER_NEUTRAL = 0
OWNER_ME = 1
OWNER_OPP = 2

# Re-export action kinds for callers that import protocol constants.
KIND_MOVE = KIND_MOVE
KIND_PASS = KIND_PASS
KIND_BUILD = KIND_BUILD

DIR_UP = 0
DIR_DOWN = 1
DIR_LEFT = 2
DIR_RIGHT = 3

DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # up
    (1, 0),  # down
    (0, -1),  # left
    (0, 1),  # right
)


def parse_handshake(line: str) -> tuple[int, int, int]:
    """Parse ``player_id H W`` handshake line."""
    parts = line.split()
    if len(parts) != 3:
        raise ValueError(f"invalid handshake: {line!r}")
    player_id, height, width = (int(x) for x in parts)
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid board size: {height}x{width}")
    return player_id, height, width


def _read_grid(lines: list[str], height: int, width: int) -> tuple[tuple[int, ...], ...]:
    if len(lines) != height:
        raise ValueError(f"expected {height} grid lines, got {len(lines)}")
    grid: list[tuple[int, ...]] = []
    for row_idx, line in enumerate(lines):
        values = tuple(int(x) for x in line.split())
        if len(values) != width:
            raise ValueError(
                f"row {row_idx}: expected {width} values, got {len(values)}"
            )
        grid.append(values)
    return tuple(grid)


def parse_observation_frame(
    scalars_line: str,
    grid_lines: list[str],
    *,
    height: int,
    width: int,
) -> Observation:
    """Parse one observation frame after the scalars line has been read."""
    parts = scalars_line.split()
    if len(parts) != 5:
        raise ValueError(f"invalid scalars line: {scalars_line!r}")
    turn, my_land, my_army, opp_land, opp_army = (int(x) for x in parts)
    if len(grid_lines) != 3 * height:
        raise ValueError(
            f"expected {3 * height} grid lines, got {len(grid_lines)}"
        )
    type_grid = _read_grid(grid_lines[0:height], height, width)
    owner_grid = _read_grid(grid_lines[height : 2 * height], height, width)
    army_grid = _read_grid(grid_lines[2 * height : 3 * height], height, width)
    return Observation(
        height=height,
        width=width,
        turn=turn,
        my_land=my_land,
        my_army=my_army,
        opp_land=opp_land,
        opp_army=opp_army,
        type_grid=type_grid,
        owner_grid=owner_grid,
        army_grid=army_grid,
    )


def serialize_action(action: Action) -> str:
    """Serialize an action to a single protocol line (no trailing newline)."""
    return (
        f"{int(action.kind)} {int(action.row)} {int(action.col)} "
        f"{int(action.direction)} {int(action.split)}"
    )


def parse_action_line(line: str) -> Action:
    """Parse one action line into an :class:`Action`."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"invalid action line: {line!r}")
    kind, row, col, direction, split = (int(x) for x in parts)
    return Action(kind=kind, row=row, col=col, direction=direction, split=split)


def pass_line() -> str:
    """Return the canonical pass action line (no trailing newline)."""
    return serialize_action(PASS_ACTION)
