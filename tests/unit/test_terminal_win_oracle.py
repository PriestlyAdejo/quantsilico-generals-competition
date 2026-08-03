"""Unit tests for terminal-win oracle and hunt plan."""

from __future__ import annotations

from generals_bot.action import Action
from generals_bot.observation import Observation
from generals_bot.policies.general_hunt_plan import GeneralHuntPlan, update_hunt_plan
from generals_bot.policies.terminal_win_oracle import (
    find_immediate_touch_actions,
    heuristic_touch_wins,
    immediate_terminal_win_proposals,
)
from generals_bot.protocol import OWNER_ME, OWNER_NEUTRAL, OWNER_OPP, TYPE_GENERAL, TYPE_PLAIN
from generals_bot.risk.shield import SurvivalShield
from generals_bot.policies.base import Proposal


def _obs(
    *,
    turn: int,
    h: int = 3,
    w: int = 3,
    me: tuple[int, int] = (1, 0),
    me_army: int = 5,
    eg: tuple[int, int] = (1, 1),
    eg_army: int = 20,
) -> Observation:
    types = [[TYPE_PLAIN] * w for _ in range(h)]
    owners = [[OWNER_NEUTRAL] * w for _ in range(h)]
    armies = [[0] * w for _ in range(h)]
    types[me[0]][me[1]] = TYPE_PLAIN
    owners[me[0]][me[1]] = OWNER_ME
    armies[me[0]][me[1]] = me_army
    types[0][0] = TYPE_GENERAL
    owners[0][0] = OWNER_ME
    armies[0][0] = 3
    types[eg[0]][eg[1]] = TYPE_GENERAL
    owners[eg[0]][eg[1]] = OWNER_OPP
    armies[eg[0]][eg[1]] = eg_army
    return Observation(
        turn=turn,
        height=h,
        width=w,
        my_land=2,
        my_army=me_army + 3,
        opp_land=1,
        opp_army=eg_army,
        type_grid=tuple(tuple(r) for r in types),
        owner_grid=tuple(tuple(r) for r in owners),
        army_grid=tuple(tuple(r) for r in armies),
    )


def test_adjacent_before_deathtouch_requires_margin() -> None:
    obs = _obs(turn=100, me_army=5, eg_army=20)
    # Right from (1,0) onto (1,1): direction 1 is typically right — check DIRECTIONS
    from generals_bot.protocol import DIRECTIONS

    # Find direction that moves (1,0)->(1,1)
    direction = next(i for i, (dr, dc) in enumerate(DIRECTIONS) if (1 + dr, 0 + dc) == (1, 1))
    action = Action.move(1, 0, direction, split=0)
    assert not heuristic_touch_wins(obs, action, (1, 1))


def test_adjacent_after_deathtouch_bypasses_margin() -> None:
    obs = _obs(turn=800, me_army=5, eg_army=20)
    from generals_bot.protocol import DIRECTIONS

    direction = next(i for i, (dr, dc) in enumerate(DIRECTIONS) if (1 + dr, 0 + dc) == (1, 1))
    action = Action.move(1, 0, direction, split=0)
    assert heuristic_touch_wins(obs, action, (1, 1))
    cands = find_immediate_touch_actions(obs, (1, 1))
    assert cands
    assert cands[0].deathtouch_active


def test_terminal_win_outranks_defend() -> None:
    obs = _obs(turn=850, me_army=8, eg_army=50)
    wins = immediate_terminal_win_proposals(obs, (1, 1))
    assert wins
    defend = Proposal(
        action=Action.move(0, 0, 1, split=0),
        option="DEFEND",
        module="defence",
        hard_priority=100,
        score=2000.0,
        confidence=0.95,
        explanation_code="reinforce",
    )
    from generals_bot.legal import enumerate_legal_actions

    legal = enumerate_legal_actions(obs)
    selected = SurvivalShield().select(obs, wins + [defend], legal)
    assert selected is not None
    assert selected.option == "IMMEDIATE_TERMINAL_WIN"


def test_hunt_plan_persists_general() -> None:
    obs = _obs(turn=200, me_army=20, eg_army=5)
    plan = GeneralHuntPlan()
    plan = update_hunt_plan(plan, obs, known_general=(1, 1), emergency=False)
    assert plan.active
    assert plan.general == (1, 1)
    assert plan.source is not None
    assert plan.route


def test_multiple_attacks_one_terminal_win() -> None:
    # Build mutable then freeze
    types = [[TYPE_PLAIN] * 3 for _ in range(3)]
    owners = [[OWNER_NEUTRAL] * 3 for _ in range(3)]
    armies = [[0] * 3 for _ in range(3)]
    types[0][0] = TYPE_GENERAL
    owners[0][0] = OWNER_ME
    armies[0][0] = 3
    owners[1][0] = OWNER_ME
    armies[1][0] = 10
    owners[2][2] = OWNER_ME
    armies[2][2] = 30
    types[1][1] = TYPE_GENERAL
    owners[1][1] = OWNER_OPP
    armies[1][1] = 99
    obs = Observation(
        turn=900,
        height=3,
        width=3,
        my_land=3,
        my_army=43,
        opp_land=1,
        opp_army=99,
        type_grid=tuple(tuple(r) for r in types),
        owner_grid=tuple(tuple(r) for r in owners),
        army_grid=tuple(tuple(r) for r in armies),
    )
    cands = find_immediate_touch_actions(obs, (1, 1))
    assert cands
    from generals_bot.protocol import DIRECTIONS

    for c in cands:
        dr, dc = DIRECTIONS[c.action.direction]
        assert (c.action.row + dr, c.action.col + dc) == (1, 1)
