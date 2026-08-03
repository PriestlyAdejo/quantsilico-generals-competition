"""Unit tests for qualification hardening: phase, Deathtouch, v2, rewards."""

from __future__ import annotations

from generals_bot.action import KIND_MOVE
from generals_bot.evaluation.qualification import (
    QualificationGameRecord,
    classify_expander_failure,
    summarise_wdl,
)
from generals_bot.legal import is_legal_action
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import TraceLevel
from generals_bot.policies.heuristic_v2_qualifier import HeuristicV2QualifierPolicy
from generals_bot.policies.phase_controller import (
    StrategicPhase,
    is_dominant_position,
    select_phase,
)
from generals_bot.selector import create_policy, list_policies
from generals_bot.training.reward_audit import DRAW_PENALTY, reward_audit_report


def _obs_deathtouch_adjacent() -> Observation:
    """Own stack adjacent to heavily defended enemy general after turn 800."""
    return Observation(
        height=4,
        width=4,
        turn=850,
        my_land=8,
        my_army=40,
        opp_land=3,
        opp_army=50,
        type_grid=((4, 1, 1, 0), (1, 1, 1, 1), (1, 1, 4, 1), (0, 1, 1, 1)),
        owner_grid=((1, 1, 1, 0), (1, 1, 1, 0), (1, 1, 2, 0), (0, 0, 0, 0)),
        # Own cell (1,2) has 3 army → sendable 2; enemy general has 40 → classic margin fails.
        army_grid=((12, 5, 4, 0), (4, 3, 3, 0), (2, 2, 40, 0), (0, 0, 0, 0)),
    )


def test_v2_registered() -> None:
    assert "heuristic_v2_qualifier" in list_policies()
    p = create_policy("heuristic_v2_qualifier")
    assert isinstance(p, HeuristicV2QualifierPolicy)


def test_dominant_position() -> None:
    assert is_dominant_position(my_land=70, opp_land=20)
    assert not is_dominant_position(my_land=40, opp_land=40)


def test_phase_mandatory_deathtouch() -> None:
    obs = Observation(
        height=2,
        width=2,
        turn=801,
        my_land=10,
        my_army=20,
        opp_land=5,
        opp_army=10,
        type_grid=((1, 1), (1, 1)),
        owner_grid=((1, 1), (0, 0)),
        army_grid=((5, 5), (0, 0)),
    )
    phase, reason = select_phase(
        obs,
        prev=StrategicPhase.EXPANSION,
        enemy_contact=True,
        enemy_general_known=True,
        own_general_threatened=False,
        dominant=True,
        mobile_ratio=0.5,
        candidate_mask_size=3,
    )
    assert phase == StrategicPhase.DEATHTOUCH_HUNT
    assert "deathtouch" in reason


def test_v2_deathtouch_allows_insufficient_army() -> None:
    policy = HeuristicV2QualifierPolicy()
    state = policy.initial_state(GameContext(0, 4, 4))
    obs = _obs_deathtouch_adjacent()
    # Seed memory so known enemy general is remembered from visibility
    decision = policy.act(obs, state, deterministic=True, trace=TraceLevel.DECISION, deadline=None)
    assert is_legal_action(obs, decision.action)
    # Prefer touch or approach; must not refuse solely due to army margin.
    touch_proposals = [
        p
        for p in decision.proposals
        if p.module in {"deathtouch", "general_hunt", "attack"}
        and p.action.kind == KIND_MOVE
    ]
    assert touch_proposals, "expected Deathtouch/hunt proposals despite defender army"
    assert decision.shield_result.get("phase") in {
        StrategicPhase.DEATHTOUCH_HUNT.value,
        StrategicPhase.GENERAL_HUNT.value,
        StrategicPhase.EMERGENCY_DEFENCE.value,
        StrategicPhase.CONVERSION.value,
    }


def test_failure_classifier_deathtouch() -> None:
    rec = QualificationGameRecord(
        policy="heuristic_v1",
        opponent="official_expander",
        seed=0,
        position=0,
        winner=-1,
        terminal_turn=1200,
        terminal_reason="DRAW_TURN_LIMIT",
        draws=1,
        first_enemy_contact_turn=100,
        enemy_general_discovered=True,
        turn_enemy_general_discovered=400,
        dominant_at_terminal=True,
    )
    assert classify_expander_failure(rec) == "DEATHTOUCH_NOT_EXPLOITED"


def test_summarise_wdl_not_score_alone() -> None:
    records = [
        QualificationGameRecord(
            policy="x",
            opponent="y",
            seed=i,
            position=0,
            winner=-1,
            terminal_turn=1200,
            terminal_reason="DRAW_TURN_LIMIT",
            draws=1,
        )
        for i in range(3)
    ]
    summary = summarise_wdl(records)
    assert summary["wins"] == 0
    assert summary["draws"] == 3
    assert summary["score_rate"] == 0.5
    assert "Do not use score_rate alone" in summary["note"]


def test_reward_audit_draw_penalty_config() -> None:
    report = reward_audit_report()
    assert report["campaign_gate"]
    assert DRAW_PENALTY.terminal_reward(winner=-1) == -0.2
    assert DRAW_PENALTY.terminal_reward(winner=0, perspective=0) == 1.0
