"""Tests for heuristic v1 determinism and legality."""

from __future__ import annotations

from generals_bot.legal import is_legal_action
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import TraceLevel
from generals_bot.policies.heuristic_config import AGGRESSIVE
from generals_bot.policies.heuristic_v1 import HeuristicV1Policy
from generals_bot.selector import create_policy, list_policies


def _obs() -> Observation:
    return Observation(
        height=4,
        width=4,
        turn=60,
        my_land=4,
        my_army=30,
        opp_land=2,
        opp_army=10,
        type_grid=((4, 1, 1, 0), (1, 2, 1, 1), (1, 1, 1, 1), (0, 1, 1, 1)),
        owner_grid=((1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 2), (0, 0, 0, 2)),
        army_grid=((10, 6, 0, 0), (5, 0, 0, 0), (4, 0, 0, 3), (0, 0, 0, 7)),
    )


def test_heuristic_v1_deterministic_and_legal() -> None:
    policy = HeuristicV1Policy()
    state = policy.initial_state(GameContext(0, 4, 4))
    obs = _obs()
    d1 = policy.act(obs, state, deterministic=True, trace=TraceLevel.DECISION, deadline=None)
    d2 = policy.act(obs, state, deterministic=True, trace=TraceLevel.NONE, deadline=None)
    assert d1.action == d2.action
    assert is_legal_action(obs, d1.action)
    assert d1.proposals


def test_variants_registered() -> None:
    names = list_policies()
    assert "heuristic_v1" in names
    assert "heuristic_aggressive" in names
    policy = create_policy("heuristic_aggressive")
    assert isinstance(policy, HeuristicV1Policy)
    assert policy.config == AGGRESSIVE
