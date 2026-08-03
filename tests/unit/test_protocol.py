"""Unit tests for protocol parsing and serialization."""

from __future__ import annotations

import pytest

from generals_bot.action import PASS_ACTION, Action
from generals_bot.protocol import (
    parse_action_line,
    parse_handshake,
    parse_observation_frame,
    serialize_action,
)


def test_parse_handshake() -> None:
    assert parse_handshake("0 18 21") == (0, 18, 21)


def test_parse_handshake_rejects_bad() -> None:
    with pytest.raises(ValueError):
        parse_handshake("0 18")


def test_serialize_pass() -> None:
    assert serialize_action(PASS_ACTION) == "1 0 0 0 0"


def test_roundtrip_action() -> None:
    action = Action.move(3, 5, 1, split=1)
    assert parse_action_line(serialize_action(action)) == action


def test_parse_observation_frame() -> None:
    scalars = "12 4 10 3 8"
    # 2x2 board
    type_lines = ["1 1", "2 4"]
    owner_lines = ["1 0", "0 1"]
    army_lines = ["5 0", "0 3"]
    obs = parse_observation_frame(
        scalars,
        type_lines + owner_lines + army_lines,
        height=2,
        width=2,
    )
    assert obs.turn == 12
    assert obs.my_land == 4
    assert obs.type_grid[1][1] == 4
    assert obs.owner_grid[0][0] == 1
    assert obs.army_grid[0][0] == 5
