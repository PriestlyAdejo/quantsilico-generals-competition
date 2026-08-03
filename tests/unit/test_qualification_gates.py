"""Tests for three-level qualification gates and shield tie-break order."""

from __future__ import annotations

from generals_bot.action import Action
from generals_bot.evaluation.qualification_gates import evaluate_screening_smoke
from generals_bot.policies.base import Proposal
from generals_bot.risk.shield import proposal_rank_key


def test_screening_passes_planner_like() -> None:
    r = evaluate_screening_smoke(
        expander={"wins": 5, "losses": 0, "enemy_general_discovery_rate": 0.5},
        hunter={"wins": 3, "draws": 0, "losses": 5, "games": 8},
    )
    assert r.passed


def test_screening_rejects_garrison_like() -> None:
    r = evaluate_screening_smoke(
        expander={"wins": 0, "losses": 0, "enemy_general_discovery_rate": 0.0},
        hunter={"wins": 0, "draws": 0, "losses": 8, "games": 8},
    )
    assert not r.passed
    assert any("wins" in x or "discovery" in x or "hunter" in x for x in r.reasons)


def test_screening_rejects_intercept_hunter_collapse() -> None:
    r = evaluate_screening_smoke(
        expander={"wins": 6, "losses": 0, "enemy_general_discovery_rate": 0.38},
        hunter={"wins": 0, "draws": 0, "losses": 8, "games": 8},
    )
    assert not r.passed
    assert any("hunter" in x for x in r.reasons)


def test_proposal_rank_key_is_total_order() -> None:
    a = Proposal(
        action=Action.move(1, 2, 0, split=0),
        option="A",
        module="m",
        hard_priority=10,
        score=1.0,
        confidence=0.5,
        explanation_code="t",
    )
    b = Proposal(
        action=Action.move(1, 2, 0, split=1),
        option="A",
        module="m",
        hard_priority=10,
        score=1.0,
        confidence=0.5,
        explanation_code="t",
    )
    assert proposal_rank_key(a) != proposal_rank_key(b)
    assert proposal_rank_key(a) < proposal_rank_key(b)
