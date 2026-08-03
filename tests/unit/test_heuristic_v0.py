"""Determinism and legality tests for heuristic v0."""

from __future__ import annotations

from generals_bot.legal import is_legal_action
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import TraceLevel
from generals_bot.policies.heuristic_v0 import HeuristicV0Policy


def _obs() -> Observation:
    return Observation(
        height=3,
        width=3,
        turn=10,
        my_land=3,
        my_army=20,
        opp_land=1,
        opp_army=5,
        type_grid=((4, 1, 1), (1, 2, 1), (1, 1, 1)),
        owner_grid=((1, 1, 0), (1, 0, 0), (0, 0, 2)),
        army_grid=((8, 5, 0), (4, 0, 0), (0, 0, 5)),
    )


def test_heuristic_v0_deterministic() -> None:
    policy = HeuristicV0Policy()
    state = policy.initial_state(GameContext(0, 3, 3))
    obs = _obs()
    d1 = policy.act(obs, state, deterministic=True, trace=TraceLevel.NONE, deadline=None)
    d2 = policy.act(obs, state, deterministic=True, trace=TraceLevel.NONE, deadline=None)
    assert d1.action == d2.action
    assert is_legal_action(obs, d1.action)


def test_heuristic_v0_always_legal() -> None:
    policy = HeuristicV0Policy()
    state = policy.initial_state(GameContext(0, 3, 3))
    obs = _obs()
    decision = policy.act(obs, state, deterministic=True, trace=TraceLevel.DECISION, deadline=None)
    assert is_legal_action(obs, decision.action)
    assert decision.legal_action_count >= 1
