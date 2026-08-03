"""Bounded scout-task assignment — does not take over the whole policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from generals_bot.action import KIND_MOVE, Action
from generals_bot.map_memory import MapMemory
from generals_bot.observation import Observation
from generals_bot.policies.base import Proposal
from generals_bot.policies.exploration_planner import (
    ExplorationPlanner,
    ExplorationState,
    partition_fog_regions,
)
from generals_bot.policies.general_garrison import find_own_general
from generals_bot.protocol import DIRECTIONS, OWNER_ME, OWNER_OPP, TYPE_FOG

DEFAULT_TTL = 40
STALL_ABORT = 20


@dataclass
class ScoutTask:
    source: tuple[int, int] | None = None
    target: tuple[int, int] | None = None
    region_id: int | None = None
    assigned_turn: int = 0
    ttl: int = DEFAULT_TTL
    stall: int = 0
    last_reveal_turn: int = 0
    abort_reason: str = ""
    progress: int = 0
    recent: list[tuple[int, int]] = field(default_factory=list)

    def to_diagnostics(self) -> dict:
        return {
            "scout_source": self.source,
            "scout_target": self.target,
            "scout_region_id": self.region_id,
            "scout_ttl": self.ttl,
            "scout_stall": self.stall,
            "scout_abort_reason": self.abort_reason,
            "scout_progress": self.progress,
        }


def _pick_source(obs: Observation, gen: tuple[int, int] | None) -> tuple[int, int] | None:
    """Largest stack that is not the general (or general only if surplus)."""
    best = None
    best_a = 0
    for r in range(obs.height):
        for c in range(obs.width):
            if obs.owner_grid[r][c] != OWNER_ME:
                continue
            a = obs.army_grid[r][c]
            if a <= 2:
                continue
            if gen is not None and (r, c) == gen and a < 8:
                continue
            if a > best_a:
                best_a = a
                best = (r, c)
    return best


class BoundedScoutAssigner:
    """Assign one scout stack to one fog target; other modules keep operating."""

    def __init__(self) -> None:
        self._planner = ExplorationPlanner(stall_turns=STALL_ABORT)

    def update(
        self,
        task: ScoutTask,
        obs: Observation,
        memory: MapMemory,
        gen_mask: list[list[bool]],
        *,
        last_enemy: tuple[int, int] | None,
        newly_revealed: bool,
        enemy_general_known: bool,
        emergency: bool,
    ) -> ScoutTask:
        if enemy_general_known or emergency:
            task.abort_reason = "higher_priority_threat_or_known_general"
            task.target = None
            task.source = None
            return task

        if newly_revealed:
            task.last_reveal_turn = obs.turn
            task.stall = 0
            task.progress += 1
        else:
            task.stall = obs.turn - task.last_reveal_turn

        # Abort conditions
        if task.target is not None:
            age = obs.turn - task.assigned_turn
            if age >= task.ttl:
                task.abort_reason = "ttl_expired"
                task.target = None
            elif task.stall >= STALL_ABORT:
                task.abort_reason = "stall_window"
                task.target = None
            elif task.source is not None:
                sr, sc = task.source
                if obs.owner_grid[sr][sc] != OWNER_ME or obs.army_grid[sr][sc] <= 2:
                    task.abort_reason = "source_too_weak"
                    task.target = None
                    task.source = None

        if task.target is not None:
            return task

        # Fresh assignment via region planner
        est = ExplorationState(
            last_reveal_turn=task.last_reveal_turn,
            stalled_turns=task.stall,
        )
        est = self._planner.update(
            est,
            obs,
            memory,
            gen_mask,
            last_enemy=last_enemy,
            newly_revealed=newly_revealed,
            enemy_general_known=False,
        )
        gen = find_own_general(obs)
        task.source = _pick_source(obs, gen)
        task.target = est.current_target
        task.region_id = est.target_region_id
        task.assigned_turn = obs.turn
        task.ttl = DEFAULT_TTL
        task.abort_reason = ""
        if task.target is None:
            task.abort_reason = "no_unresolved_target"
        return task

    def proposals(self, task: ScoutTask, obs: Observation, legal: list[Action]) -> list[Proposal]:
        if task.target is None or task.source is None:
            return []
        ar, ac = task.target
        sr, sc = task.source
        out: list[Proposal] = []
        for action in legal:
            if action.kind != KIND_MOVE or action.split == 1:
                continue
            # Prefer moves from the assigned source; allow nearby helpers weakly
            from_src = (action.row, action.col) == (sr, sc)
            if not from_src and abs(action.row - sr) + abs(action.col - sc) > 2:
                continue
            dr, dc = DIRECTIONS[action.direction]
            nr, nc = action.row + dr, action.col + dc
            sendable = obs.army_grid[action.row][action.col] - 1
            if sendable < 1:
                continue
            dist_src = abs(action.row - ar) + abs(action.col - ac)
            dist_dst = abs(nr - ar) + abs(nc - ac)
            into = obs.type_grid[nr][nc] == TYPE_FOG or obs.owner_grid[nr][nc] == OWNER_OPP
            if into and ((nr, nc) == (ar, ac) or dist_dst <= 1):
                out.append(
                    Proposal(
                        action=action,
                        option="SCOUT_TASK",
                        module="bounded_scout",
                        hard_priority=58 if from_src else 40,
                        score=700.0 + sendable * 6,
                        confidence=0.75,
                        explanation_code="bounded_scout_enter",
                    )
                )
            elif dist_dst < dist_src and sendable >= 2:
                out.append(
                    Proposal(
                        action=action,
                        option="SCOUT_TASK",
                        module="bounded_scout",
                        hard_priority=36 if from_src else 22,
                        score=250.0 + sendable * 4 + 20.0 * (dist_src - dist_dst),
                        confidence=0.65,
                        explanation_code="bounded_scout_approach",
                    )
                )
        return out
