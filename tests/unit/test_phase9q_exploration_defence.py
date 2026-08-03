"""Scenario tests for exploration planner and threat assessment."""

from __future__ import annotations

from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import TraceLevel
from generals_bot.policies.exploration_planner import ExplorationPlanner, ExplorationState
from generals_bot.policies.heuristic_v2_qualifier import HeuristicV2QualifierPolicy
from generals_bot.policies.phase_controller import StrategicPhase, select_phase
from generals_bot.policies.threat_assessment import ThreatMemory, assess_threat
from generals_bot.map_memory import MapMemory


def _blank_obs(**kwargs) -> Observation:
    base = dict(
        height=5,
        width=5,
        turn=100,
        my_land=5,
        my_army=20,
        opp_land=3,
        opp_army=10,
        type_grid=((4, 1, 1, 0, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (0, 1, 1, 1, 1)),
        owner_grid=((1, 1, 1, 0, 0), (1, 1, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        army_grid=((10, 3, 2, 0, 0), (4, 2, 0, 0, 0), (3, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    base.update(kwargs)
    return Observation(**base)


def test_fog_alone_does_not_trigger_emergency() -> None:
    obs = _blank_obs(
        type_grid=((4, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        owner_grid=((1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        army_grid=((8, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    assessment, mem = assess_threat(obs, ThreatMemory())
    assert not assessment.emergency
    assert assessment.phase_override != "EMERGENCY_DEFENCE"


def test_adjacent_enemy_triggers_emergency() -> None:
    obs = _blank_obs(
        type_grid=((4, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)),
        owner_grid=((1, 2, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        army_grid=((8, 5, 0, 0, 0), (3, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    assessment, mem = assess_threat(obs, ThreatMemory())
    assert assessment.emergency
    assert mem.activations == 1


def test_emergency_decays_without_evidence() -> None:
    mem = ThreatMemory(emergency_active=True, entered_turn=10, last_evidence_turn=10, confidence=0.9)
    obs = _blank_obs(turn=40)  # no enemy visible
    assessment, mem2 = assess_threat(obs, mem, max_emergency_without_evidence=20)
    # After enough turns without evidence, should exit or decay heavily
    for t in range(41, 70):
        assessment, mem2 = assess_threat(
            _blank_obs(turn=t), mem2, max_emergency_without_evidence=20
        )
    assert not assessment.emergency


def test_distant_enemy_is_caution_not_emergency() -> None:
    obs = _blank_obs(
        type_grid=((4, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1), (1, 1, 1, 2, 1)),
        owner_grid=((1, 1, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 2, 0)),
        army_grid=((8, 2, 0, 0, 0), (2, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 6, 0)),
    )
    assessment, _ = assess_threat(obs, ThreatMemory(), caution_dist=4, emergency_enter_dist=2)
    # Manhattan from (0,0) to (4,3) = 7 > caution_dist 4 → neither, or adjust
    assert not assessment.emergency


def test_exploration_target_persists() -> None:
    planner = ExplorationPlanner(stall_turns=100)
    state = ExplorationState()
    obs = _blank_obs(turn=50)
    mem = MapMemory.create(5, 5)
    mem.update(obs)
    # Force fog elsewhere
    obs2 = _blank_obs(
        turn=51,
        type_grid=((4, 1, 0, 0, 0), (1, 1, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        owner_grid=((1, 1, 0, 0, 0), (1, 1, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        army_grid=((8, 3, 0, 0, 0), (3, 2, 0, 0, 0), (2, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    mem.update(obs2)
    mask = mem.possible_enemy_general_mask(obs2)
    state = planner.update(
        state, obs2, mem, mask, last_enemy=None, newly_revealed=True, enemy_general_known=False
    )
    first = state.current_target
    assert first is not None
    state2 = planner.update(
        state, obs2, mem, mask, last_enemy=None, newly_revealed=False, enemy_general_known=False
    )
    assert state2.current_target == first


def test_search_for_contact_phase() -> None:
    obs = _blank_obs(turn=160, opp_land=5, my_land=30)
    phase, reason = select_phase(
        obs,
        prev=StrategicPhase.EXPANSION,
        enemy_contact=False,
        enemy_general_known=False,
        own_general_threatened=False,
        dominant=False,
        mobile_ratio=0.5,
        candidate_mask_size=10,
    )
    assert phase == StrategicPhase.SEARCH_FOR_CONTACT


def test_v2_deterministic_legal() -> None:
    from generals_bot.legal import is_legal_action

    policy = HeuristicV2QualifierPolicy()
    state = policy.initial_state(GameContext(0, 5, 5))
    obs = _blank_obs(turn=90)
    d1 = policy.act(obs, state, deterministic=True, trace=TraceLevel.DECISION, deadline=None)
    d2 = policy.act(obs, state, deterministic=True, trace=TraceLevel.NONE, deadline=None)
    assert d1.action == d2.action
    assert is_legal_action(obs, d1.action)
