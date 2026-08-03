"""Legal-action enumeration tests."""

from __future__ import annotations

from generals_bot.action import PASS_ACTION, Action
from generals_bot.legal import enumerate_legal_actions, is_legal_action
from generals_bot.observation import Observation


def _obs(
    *,
    types: list[list[int]],
    owners: list[list[int]],
    armies: list[list[int]],
) -> Observation:
    h, w = len(types), len(types[0])
    return Observation(
        height=h,
        width=w,
        turn=1,
        my_land=sum(1 for row in owners for v in row if v == 1),
        my_army=sum(armies[r][c] for r in range(h) for c in range(w) if owners[r][c] == 1),
        opp_land=0,
        opp_army=0,
        type_grid=tuple(tuple(r) for r in types),
        owner_grid=tuple(tuple(r) for r in owners),
        army_grid=tuple(tuple(r) for r in armies),
    )


def test_pass_always_legal() -> None:
    obs = _obs(
        types=[[4, 1], [1, 2]],
        owners=[[1, 1], [0, 0]],
        armies=[[3, 1], [0, 0]],
    )
    legal = enumerate_legal_actions(obs)
    assert PASS_ACTION in legal
    assert is_legal_action(obs, PASS_ACTION)


def test_mountain_blocks_move() -> None:
    obs = _obs(
        types=[[4, 2], [1, 1]],
        owners=[[1, 0], [0, 0]],
        armies=[[5, 0], [0, 0]],
    )
    legal = enumerate_legal_actions(obs)
    # From (0,0) right into mountain must be absent
    rights = [a for a in legal if a.kind == 0 and a.row == 0 and a.col == 0 and a.direction == 3]
    assert rights == []


def test_source_army_must_exceed_one() -> None:
    obs = _obs(
        types=[[4, 1]],
        owners=[[1, 0]],
        armies=[[1, 0]],
    )
    legal = enumerate_legal_actions(obs)
    moves = [a for a in legal if a.kind == 0]
    assert moves == []


def test_build_requires_price() -> None:
    # General at (0,0); plain at (0,7) costs 35; army 34 insufficient, 35 ok
    types = [[4] + [1] * 7]
    owners = [[1] * 8]
    armies_low = [[40] + [0] * 6 + [34]]
    armies_ok = [[40] + [0] * 6 + [35]]
    obs_low = _obs(types=types, owners=owners, armies=armies_low)
    obs_ok = _obs(types=types, owners=owners, armies=armies_ok)
    assert Action.build(0, 7) not in enumerate_legal_actions(obs_low)
    assert Action.build(0, 7) in enumerate_legal_actions(obs_ok)
