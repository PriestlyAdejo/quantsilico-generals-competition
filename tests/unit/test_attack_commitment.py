"""Unit tests for AttackCommitmentState machine and PREPARE soft-gate."""

from __future__ import annotations

from generals_bot.action import Action, KIND_BUILD, KIND_MOVE
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.attack_commitment import (
    AttackCommitmentState,
    AttackReadinessConfig,
    evaluate_readiness_ok,
    filter_proposals_for_commitment,
    update_attack_commitment,
)
from generals_bot.policies.base import Proposal, TraceLevel
from generals_bot.policies.heuristic_v2_ablations import FLAGS, create_ablation
from generals_bot.selector import create_policy, list_policies


def test_commitment_none_to_prepare() -> None:
    nxt = update_attack_commitment(
        AttackCommitmentState.NONE,
        known_eg=(1, 1),
        eg_confidence=1.0,
        belief_age=0,
        readiness_ok=False,
        emergency=False,
        route_illegal=False,
        eg_captured=False,
        terminal=False,
        combat_margin_negative=False,
        convert_ready=False,
        turn=10,
    )
    assert nxt == AttackCommitmentState.PREPARE


def test_commitment_prepare_to_commit_requires_readiness() -> None:
    stuck = update_attack_commitment(
        AttackCommitmentState.PREPARE,
        known_eg=(1, 1),
        eg_confidence=1.0,
        belief_age=0,
        readiness_ok=False,
        emergency=False,
        route_illegal=False,
        eg_captured=False,
        terminal=False,
        combat_margin_negative=False,
        convert_ready=False,
        turn=20,
    )
    assert stuck == AttackCommitmentState.PREPARE
    ready = update_attack_commitment(
        AttackCommitmentState.PREPARE,
        known_eg=(1, 1),
        eg_confidence=1.0,
        belief_age=0,
        readiness_ok=True,
        emergency=False,
        route_illegal=False,
        eg_captured=False,
        terminal=False,
        combat_margin_negative=False,
        convert_ready=False,
        turn=25,
    )
    assert ready == AttackCommitmentState.COMMIT


def test_commit_to_retreat_only_on_negative_margin() -> None:
    stay = update_attack_commitment(
        AttackCommitmentState.COMMIT,
        known_eg=(1, 1),
        eg_confidence=1.0,
        belief_age=0,
        readiness_ok=True,
        emergency=False,
        route_illegal=False,
        eg_captured=False,
        terminal=False,
        combat_margin_negative=False,
        convert_ready=False,
        turn=40,
    )
    assert stay == AttackCommitmentState.COMMIT
    retreat = update_attack_commitment(
        AttackCommitmentState.COMMIT,
        known_eg=(1, 1),
        eg_confidence=1.0,
        belief_age=0,
        readiness_ok=True,
        emergency=False,
        route_illegal=False,
        eg_captured=False,
        terminal=False,
        combat_margin_negative=True,
        convert_ready=False,
        turn=41,
    )
    assert retreat == AttackCommitmentState.RETREAT


def test_reset_on_stale_belief_or_emergency() -> None:
    stale = update_attack_commitment(
        AttackCommitmentState.COMMIT,
        known_eg=(1, 1),
        eg_confidence=0.1,
        belief_age=999,
        readiness_ok=True,
        emergency=False,
        route_illegal=False,
        eg_captured=False,
        terminal=False,
        combat_margin_negative=False,
        convert_ready=False,
        turn=50,
    )
    assert stale == AttackCommitmentState.NONE
    emergency = update_attack_commitment(
        AttackCommitmentState.COMMIT,
        known_eg=(1, 1),
        eg_confidence=1.0,
        belief_age=0,
        readiness_ok=True,
        emergency=True,
        route_illegal=False,
        eg_captured=False,
        terminal=False,
        combat_margin_negative=False,
        convert_ready=False,
        turn=51,
    )
    assert emergency == AttackCommitmentState.RETREAT


def test_prepare_demotes_approach_allows_build() -> None:
    approach = Proposal(
        action=Action(kind=KIND_MOVE, row=0, col=0, direction=0, split=0),
        option="GENERAL_HUNT",
        module="general_hunt",
        hard_priority=93,
        score=900.0,
        confidence=0.8,
        explanation_code="approach_enemy_general",
    )
    build = Proposal(
        action=Action(kind=KIND_BUILD, row=1, col=1),
        option="BUILD",
        module="castle",
        hard_priority=26,
        score=180.0,
        confidence=0.55,
        explanation_code="selective_castle",
    )
    out = filter_proposals_for_commitment(
        [approach, build],
        AttackCommitmentState.PREPARE,
        known_eg=(3, 3),
        emergency=False,
    )
    assert len(out) == 2
    demoted = next(p for p in out if p.explanation_code == "approach_enemy_general")
    assert demoted.hard_priority <= 45
    assert any(p.option == "BUILD" for p in out)


def test_commit_strips_build_and_offroute_collect() -> None:
    build = Proposal(
        action=Action(kind=KIND_BUILD, row=1, col=1),
        option="BUILD",
        module="castle",
        hard_priority=26,
        score=180.0,
        confidence=0.55,
        explanation_code="selective_castle",
    )
    # Move away from EG at (0,0): from (2,2) direction right increases dist.
    collect = Proposal(
        action=Action(kind=KIND_MOVE, row=2, col=2, direction=1, split=0),
        option="COLLECT",
        module="collection",
        hard_priority=15,
        score=50.0,
        confidence=0.55,
        explanation_code="concentrate_mobile_army",
    )
    out = filter_proposals_for_commitment(
        [build, collect],
        AttackCommitmentState.COMMIT,
        known_eg=(0, 0),
        emergency=False,
    )
    assert out == []


def test_readiness_requires_dwell_and_stack() -> None:
    cfg = AttackReadinessConfig(dwell_turns_prepare_to_commit=3, min_attack_stack=18)
    assert not evaluate_readiness_ok(
        cfg=cfg,
        eg_confidence=1.0,
        belief_age=0,
        mobile_army=30,
        route_length=10,
        gathering=False,
        own_general_threatened=False,
        prepare_dwell=1,
    )
    assert evaluate_readiness_ok(
        cfg=cfg,
        eg_confidence=1.0,
        belief_age=0,
        mobile_army=30,
        route_length=10,
        gathering=False,
        own_general_threatened=False,
        prepare_dwell=3,
    )


def test_tactical_v2_flag_registered() -> None:
    assert "heuristic_v2f_tactical_attack_v2" in FLAGS
    assert FLAGS["heuristic_v2f_tactical_attack_v2"].use_attack_commitment is True
    assert "heuristic_v2f_tactical_attack_v2" in list_policies()
    p = create_policy("heuristic_v2f_tactical_attack_v2")
    st = p.initial_state(GameContext(0, 4, 4))
    assert st.data.get("attack_commitment") == AttackCommitmentState.NONE.value


def test_tactical_v2_act_sets_prepare_when_eg_known() -> None:
    obs = Observation(
        height=4,
        width=4,
        turn=60,
        my_land=4,
        my_army=20,
        opp_land=2,
        opp_army=8,
        type_grid=((4, 1, 1, 0), (1, 1, 1, 1), (1, 1, 1, 1), (0, 1, 1, 4)),
        owner_grid=((1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 2)),
        army_grid=((8, 4, 0, 0), (3, 0, 0, 0), (2, 0, 0, 0), (0, 0, 0, 5)),
    )
    p = create_ablation("heuristic_v2f_tactical_attack_v2")
    st = p.initial_state(GameContext(0, 4, 4))
    st.data["known_enemy_general"] = (3, 3)
    d = p.act(obs, st, deterministic=True, trace=TraceLevel.DECISION, deadline=None)
    commitment = d.new_state.data.get("attack_commitment")
    assert commitment in {
        AttackCommitmentState.PREPARE.value,
        AttackCommitmentState.COMMIT.value,
        AttackCommitmentState.RETREAT.value,
        AttackCommitmentState.NONE.value,
    }
    # With modest army and short dwell, expect PREPARE soft-gate (not instant COMMIT).
    assert commitment == AttackCommitmentState.PREPARE.value
    reason = (d.new_state.data.get("diagnostics") or {}).get("phase_reason") or d.new_state.data.get(
        "phase_reason"
    )
    if d.new_state.data.get("phase") == "GENERAL_HUNT":
        assert reason == "attack_prepare_soft_gate"
