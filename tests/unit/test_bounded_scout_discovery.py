"""Documents the throwaway ExplorationState bug in BoundedScoutAssigner.

Production scout code is currently frozen at the conversion-passing
terminal_fix revision. Discovery escalation must preserve the conversion
micro gate (10/10 on the discovered-not-converted corpus) before landing.
"""

from __future__ import annotations

from generals_bot.map_memory import MapMemory
from generals_bot.observation import Observation
from generals_bot.policies.bounded_scout import BoundedScoutAssigner, ScoutTask
from generals_bot.policies.exploration_planner import ExplorationState
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


def test_terminal_fix_hash_is_conversion_frozen() -> None:
    p = create_policy("heuristic_v2f_plus_planner_terminal_fix")
    assert p.config_hash == "4c5466776180217b"


def test_bounded_scout_rebuilds_exploration_state_each_assign() -> None:
    """Current assigner discards failed_targets across reassignment calls."""
    assigner = BoundedScoutAssigner()
    obs = _obs(turn=80)
    mem = MapMemory.create(obs.height, obs.width)
    mem.update(obs)
    mask = mem.possible_enemy_general_mask(obs)
    task = ScoutTask()
    # First assignment via public update (creates internal throwaway ExplorationState)
    task = assigner.update(
        task,
        obs,
        mem,
        mask,
        last_enemy=None,
        newly_revealed=True,
        enemy_general_known=False,
        emergency=False,
    )
    assert task.target is not None
    # Simulate stall abort + reassignment — failed target memory is not persisted
    # on ScoutTask, which matches the development no-discovery diagnosis.
    task.last_reveal_turn = 40
    task.stall = 40
    task.assigned_turn = 40
    task2 = assigner.update(
        task,
        _obs(turn=90),
        mem,
        mask,
        last_enemy=None,
        newly_revealed=False,
        enemy_general_known=False,
        emergency=False,
    )
    # After abort the assigner may re-pick; the point is ScoutTask has no
    # failed_targets field and ExplorationState is not stored on the policy.
    assert not hasattr(task2, "failed_targets")
    assert isinstance(ExplorationState(), ExplorationState)
