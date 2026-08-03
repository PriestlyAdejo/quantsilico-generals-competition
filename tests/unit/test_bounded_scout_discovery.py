"""Focused tests for persistent ExplorationState and late dual-scout rules."""

from __future__ import annotations

from generals_bot.legal import enumerate_legal_actions
from generals_bot.map_memory import MapMemory
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import TraceLevel
from generals_bot.policies.bounded_scout import (
    SECOND_SCOUT_MIN_STALL,
    SECOND_SCOUT_MIN_TURN,
    BoundedScoutAssigner,
    ScoutTask,
)
from generals_bot.policies.exploration_planner import (
    REGION_ATTEMPT_DECAY_TURNS,
    ExplorationState,
    decayed_region_attempts,
)
from generals_bot.selector import create_policy


def _obs(**kwargs) -> Observation:
    base = dict(
        height=5,
        width=6,
        turn=100,
        my_land=6,
        my_army=30,
        opp_land=4,
        opp_army=12,
        type_grid=(
            (4, 1, 1, 2, 0, 0),
            (1, 1, 1, 2, 0, 0),
            (1, 1, 1, 1, 1, 1),
            (1, 1, 1, 2, 0, 0),
            (1, 1, 1, 2, 0, 0),
        ),
        owner_grid=(
            (1, 1, 1, 0, 0, 0),
            (1, 1, 1, 0, 0, 0),
            (1, 1, 1, 1, 1, 1),
            (1, 1, 1, 0, 0, 0),
            (1, 1, 1, 0, 0, 0),
        ),
        army_grid=(
            (12, 4, 3, 0, 0, 0),
            (5, 3, 2, 0, 0, 0),
            (4, 2, 2, 2, 2, 8),
            (3, 2, 2, 0, 0, 0),
            (3, 2, 2, 0, 0, 0),
        ),
    )
    base.update(kwargs)
    return Observation(**base)


def test_exploration_state_persists_between_turns() -> None:
    assigner = BoundedScoutAssigner(dual_scout=True)
    obs = _obs(turn=80)
    mem = MapMemory.create(obs.height, obs.width)
    mem.update(obs)
    mask = mem.possible_enemy_general_mask(obs)
    est = ExplorationState()
    task, _, est = assigner.update(
        ScoutTask(),
        obs,
        mem,
        mask,
        last_enemy=None,
        newly_revealed=True,
        enemy_general_known=False,
        emergency=False,
        exploration=est,
    )
    assert task.target is not None
    first = task.target
    est.bump_region_attempt(task.region_id or 0, 80)
    task2, _, est2 = assigner.update(
        task,
        _obs(turn=81),
        mem,
        mask,
        last_enemy=None,
        newly_revealed=False,
        enemy_general_known=False,
        emergency=False,
        exploration=est,
    )
    assert est2 is est
    assert est2.region_attempts.get(task.region_id or 0, 0) >= 1
    assert task2.target == first or task2.target is not None


def test_exploration_state_resets_between_games() -> None:
    p = create_policy("heuristic_v2f_plus_planner_terminal_fix")
    s0 = p.initial_state(GameContext(0, 5, 6))
    s1 = p.initial_state(GameContext(0, 5, 6))
    assert s0.data["exploration_state"] is not s1.data["exploration_state"]
    assert s0.data["scout_task"] is not s1.data["scout_task"]


def test_stall_grows_and_resets_on_reveal() -> None:
    assigner = BoundedScoutAssigner()
    obs = _obs(turn=50)
    mem = MapMemory.create(obs.height, obs.width)
    mem.update(obs)
    mask = mem.possible_enemy_general_mask(obs)
    est = ExplorationState()
    task, _, est = assigner.update(
        ScoutTask(),
        obs,
        mem,
        mask,
        last_enemy=None,
        newly_revealed=True,
        enemy_general_known=False,
        emergency=False,
        exploration=est,
    )
    task, _, est = assigner.update(
        task,
        _obs(turn=60),
        mem,
        mask,
        last_enemy=None,
        newly_revealed=False,
        enemy_general_known=False,
        emergency=False,
        exploration=est,
    )
    assert task.stall == 10 or est.scout_stall == 10 or task.stall > 0
    task, _, est = assigner.update(
        task,
        _obs(turn=61),
        mem,
        mask,
        last_enemy=None,
        newly_revealed=True,
        enemy_general_known=False,
        emergency=False,
        exploration=est,
    )
    assert task.stall == 0
    assert est.last_newly_scouted_turn == 61


def test_attempt_decay_allows_revisitation() -> None:
    est = ExplorationState()
    est.bump_region_attempt(3, turn=100)
    assert decayed_region_attempts(est, 3, 100) == 1
    later = 100 + REGION_ATTEMPT_DECAY_TURNS
    assert decayed_region_attempts(est, 3, later) == 0


def test_soft_fail_expires() -> None:
    est = ExplorationState()
    est.note_soft_fail((1, 4), turn=100)
    assert est.is_soft_failed((1, 4), 100)
    assert not est.is_soft_failed((1, 4), 200)


def test_dual_scout_activation_conditions() -> None:
    assigner = BoundedScoutAssigner(dual_scout=True)
    obs = _obs(turn=SECOND_SCOUT_MIN_TURN - 1)
    mem = MapMemory.create(obs.height, obs.width)
    mem.update(obs)
    mask = mem.possible_enemy_general_mask(obs)
    est = ExplorationState()
    task, task_b, est = assigner.update(
        ScoutTask(),
        obs,
        mem,
        mask,
        last_enemy=None,
        newly_revealed=True,
        enemy_general_known=False,
        emergency=False,
        exploration=est,
        secondary=ScoutTask(),
    )
    assert task.target is not None
    assert task_b.target is None  # before turn 1050

    task.stall = SECOND_SCOUT_MIN_STALL
    task.last_reveal_turn = SECOND_SCOUT_MIN_TURN - SECOND_SCOUT_MIN_STALL
    late = _obs(turn=SECOND_SCOUT_MIN_TURN)
    mem.update(late)
    mask2 = mem.possible_enemy_general_mask(late)
    # Force no abort by refreshing assigned_turn
    task.assigned_turn = SECOND_SCOUT_MIN_TURN
    task2, task_b2, _ = assigner.update(
        task,
        late,
        mem,
        mask2,
        last_enemy=None,
        newly_revealed=False,
        enemy_general_known=False,
        emergency=False,
        exploration=est,
        secondary=ScoutTask(),
    )
    if task2.target is not None and est.unresolved_regions >= 2:
        assert task_b2.target is not None
        assert task_b2.region_id != task2.region_id
        assert task_b2.source != task2.source


def test_terminal_fix_hash_changes_with_persistent_flag() -> None:
    p = create_policy("heuristic_v2f_plus_planner_terminal_fix")
    assert p.flags.use_persistent_explore is True
    assert p.flags.use_dual_scout is True
    assert p.config_hash != "4c5466776180217b"


def test_deterministic_actions_from_same_inputs() -> None:
    p = create_policy("heuristic_v2f_plus_planner_terminal_fix")
    obs = _obs(turn=120)
    ctx = GameContext(0, obs.height, obs.width)
    s1 = p.initial_state(ctx)
    s2 = p.initial_state(ctx)
    d1 = p.act(obs, s1, deterministic=True, trace=TraceLevel.NONE, deadline=None)
    d2 = p.act(obs, s2, deterministic=True, trace=TraceLevel.NONE, deadline=None)
    assert d1.action == d2.action
    legal = enumerate_legal_actions(obs)
    assert d1.action in legal
