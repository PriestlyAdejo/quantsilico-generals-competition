"""Unit tests for ablation references and modules."""

from __future__ import annotations

from generals_bot.legal import is_legal_action
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import TraceLevel
from generals_bot.policies.heuristic_v2_ablations import FLAGS, create_ablation
from generals_bot.policies.heuristic_v2f_reference import HeuristicV2FReferencePolicy
from generals_bot.selector import create_policy


def _obs() -> Observation:
    return Observation(
        height=4,
        width=4,
        turn=60,
        my_land=4,
        my_army=20,
        opp_land=2,
        opp_army=8,
        type_grid=((4, 1, 1, 0), (1, 1, 1, 1), (1, 1, 1, 1), (0, 1, 1, 1)),
        owner_grid=((1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)),
        army_grid=((8, 4, 0, 0), (3, 0, 0, 0), (2, 0, 0, 0), (0, 0, 0, 0)),
    )


def test_v2f_reference_loads() -> None:
    p = create_policy("heuristic_v2f_best_reference")
    assert isinstance(p, HeuristicV2FReferencePolicy)
    st = p.initial_state(GameContext(0, 4, 4))
    d = p.act(_obs(), st, deterministic=True, trace=TraceLevel.NONE, deadline=None)
    assert is_legal_action(_obs(), d.action)


def test_ablation_flags_hashed() -> None:
    hashes = {FLAGS[k].config_hash() for k in FLAGS}
    assert len(hashes) == len(FLAGS)


def test_ablation_garrison_legal() -> None:
    p = create_ablation("heuristic_v2f_plus_garrison")
    st = p.initial_state(GameContext(0, 4, 4))
    d = p.act(_obs(), st, deterministic=True, trace=TraceLevel.DECISION, deadline=None)
    assert is_legal_action(_obs(), d.action)
    assert d.policy_id == "heuristic_v2f_plus_garrison"
