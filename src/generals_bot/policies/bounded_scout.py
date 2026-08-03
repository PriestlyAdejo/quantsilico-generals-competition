"""Bounded scout-task assignment with persistent exploration memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from generals_bot.action import KIND_MOVE, Action
from generals_bot.map_memory import MapMemory
from generals_bot.observation import Observation
from generals_bot.policies.base import Proposal
from generals_bot.policies.exploration_planner import (
    ExplorationPlanner,
    ExplorationState,
    FogRegion,
    decayed_region_attempts,
    partition_fog_regions,
    score_region,
)
from generals_bot.policies.general_garrison import find_own_general
from generals_bot.protocol import DIRECTIONS, OWNER_ME, OWNER_OPP, TYPE_FOG

DEFAULT_TTL = 40
STALL_ABORT = 25
SECOND_SCOUT_MIN_TURN = 1050
SECOND_SCOUT_MIN_STALL = 40


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


def _pick_source(
    obs: Observation,
    gen: tuple[int, int] | None,
    *,
    target: tuple[int, int] | None = None,
    exclude: set[tuple[int, int]] | None = None,
) -> tuple[int, int] | None:
    """Largest usable stack; never prefer a stripped general; honour excludes."""
    exclude = exclude or set()
    best = None
    best_key = None
    for r in range(obs.height):
        for c in range(obs.width):
            if (r, c) in exclude:
                continue
            if obs.owner_grid[r][c] != OWNER_ME:
                continue
            a = obs.army_grid[r][c]
            if a <= 2:
                continue
            if gen is not None and (r, c) == gen and a < 12:
                continue
            dist = 0 if target is None else abs(r - target[0]) + abs(c - target[1])
            key = (-a, dist, r, c)
            if best_key is None or key < best_key:
                best_key = key
                best = (r, c)
    return best


class BoundedScoutAssigner:
    """Assign scout stack(s) using persistent ExplorationState on the policy."""

    def __init__(self, *, dual_scout: bool = False) -> None:
        self._planner = ExplorationPlanner(stall_turns=STALL_ABORT)
        self.dual_scout = dual_scout

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
        exploration: ExplorationState | None = None,
        secondary: ScoutTask | None = None,
    ) -> tuple[ScoutTask, ScoutTask, ExplorationState]:
        est = exploration if exploration is not None else ExplorationState()
        secondary = secondary or ScoutTask()

        if enemy_general_known:
            task.abort_reason = "higher_priority_threat_or_known_general"
            task.target = None
            task.source = None
            secondary.abort_reason = "higher_priority_threat_or_known_general"
            secondary.target = None
            secondary.source = None
            est = self._planner.update(
                est,
                obs,
                memory,
                gen_mask,
                last_enemy=last_enemy,
                newly_revealed=newly_revealed,
                enemy_general_known=True,
            )
            return task, secondary, est

        if emergency:
            # Pause scouting; keep exploration memory intact.
            task.abort_reason = "higher_priority_threat_or_known_general"
            task.target = None
            task.source = None
            secondary.abort_reason = "higher_priority_threat_or_known_general"
            secondary.target = None
            secondary.source = None
            return task, secondary, est

        if newly_revealed:
            task.last_reveal_turn = obs.turn
            task.stall = 0
            task.progress += 1
            secondary.last_reveal_turn = obs.turn
            secondary.stall = 0
            est.last_newly_scouted_turn = obs.turn
        else:
            if task.last_reveal_turn <= 0:
                task.last_reveal_turn = max(0, obs.turn - task.stall)
            task.stall = obs.turn - task.last_reveal_turn
            secondary.stall = obs.turn - max(secondary.last_reveal_turn, 0)
        est.scout_stall = task.stall
        stall_for_dual = task.stall

        task = self._maybe_abort(task, obs, est)
        secondary = self._maybe_abort(secondary, obs, est)

        # Keep planner target in sync with the live scout task.
        if task.target is not None:
            est.current_target = task.target
            est.target_region_id = task.region_id
        else:
            est.current_target = None
            est.target_region_id = None
        est.stalled_turns = task.stall
        if newly_revealed:
            est.last_reveal_turn = obs.turn
        elif est.last_reveal_turn <= 0:
            est.last_reveal_turn = task.last_reveal_turn

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
        if task.target is None:
            task.target = est.current_target
            task.region_id = est.target_region_id
            task.assigned_turn = obs.turn
            task.ttl = DEFAULT_TTL
            task.abort_reason = ""
            task.last_reveal_turn = obs.turn
            task.stall = 0
            if task.target is None:
                task.abort_reason = "no_unresolved_target"
                task.source = None
            else:
                task.source = _pick_source(obs, gen, target=task.target)
                if task.region_id is not None:
                    est.prior_regions.append(task.region_id)
                    if len(est.prior_regions) > 32:
                        est.prior_regions = est.prior_regions[-32:]
        elif task.source is None or obs.owner_grid[task.source[0]][task.source[1]] != OWNER_ME:
            task.source = _pick_source(obs, gen, target=task.target)

        need_second = (
            self.dual_scout
            and task.target is not None
            and obs.turn >= SECOND_SCOUT_MIN_TURN
            and stall_for_dual >= SECOND_SCOUT_MIN_STALL
        )
        if need_second and secondary.target is None:
            regions = partition_fog_regions(obs, memory, gen_mask)
            secondary = self._assign_secondary(
                secondary,
                obs,
                gen,
                regions,
                est,
                primary_region=task.region_id,
                exclude_source=task.source,
            )
        elif secondary.target is not None:
            if secondary.source is None or obs.owner_grid[secondary.source[0]][secondary.source[1]] != OWNER_ME:
                exclude = {task.source} if task.source is not None else set()
                if gen is not None:
                    exclude.add(gen)
                secondary.source = _pick_source(
                    obs, gen, target=secondary.target, exclude=exclude
                )

        return task, secondary, est

    def _maybe_abort(
        self,
        task: ScoutTask,
        obs: Observation,
        est: ExplorationState,
    ) -> ScoutTask:
        if task.target is None:
            return task
        age = obs.turn - task.assigned_turn
        aborted = False
        if age >= task.ttl:
            task.abort_reason = "ttl_expired"
            aborted = True
        elif task.stall >= STALL_ABORT:
            task.abort_reason = "stall_window"
            aborted = True
        elif task.source is not None:
            sr, sc = task.source
            if obs.owner_grid[sr][sc] != OWNER_ME or obs.army_grid[sr][sc] <= 2:
                task.abort_reason = "source_too_weak"
                aborted = True
        if aborted:
            if task.abort_reason in {"stall_window", "ttl_expired"}:
                est.note_soft_fail(task.target, obs.turn)
                if task.region_id is not None:
                    est.bump_region_attempt(task.region_id, obs.turn)
            task.target = None
            task.source = None
        return task

    def _assign_secondary(
        self,
        secondary: ScoutTask,
        obs: Observation,
        gen: tuple[int, int] | None,
        regions: list[FogRegion],
        est: ExplorationState,
        *,
        primary_region: int | None,
        exclude_source: tuple[int, int] | None,
    ) -> ScoutTask:
        if len(regions) < 2:
            secondary.abort_reason = "no_second_region"
            return secondary
        stacks = [
            (r, c, obs.army_grid[r][c])
            for r in range(obs.height)
            for c in range(obs.width)
            if obs.owner_grid[r][c] == OWNER_ME and obs.army_grid[r][c] > 2
        ]
        scored = sorted(
            regions,
            key=lambda reg: -score_region(
                reg,
                last_enemy=None,
                stacks=stacks,
                attempts=decayed_region_attempts(est, reg.region_id, obs.turn),
                turn=obs.turn,
            ),
        )
        exclude: set[tuple[int, int]] = set()
        if exclude_source is not None:
            exclude.add(exclude_source)
        if gen is not None:
            exclude.add(gen)
        for reg in scored:
            if primary_region is not None and reg.region_id == primary_region:
                continue
            candidates = list(reg.frontier) + list(reg.cells[:4])
            for cell in candidates:
                if est.is_soft_failed(cell, obs.turn):
                    continue
                src = _pick_source(obs, gen, target=cell, exclude=exclude)
                if src is None:
                    continue
                secondary.target = cell
                secondary.region_id = reg.region_id
                secondary.source = src
                secondary.assigned_turn = obs.turn
                secondary.ttl = DEFAULT_TTL
                secondary.last_reveal_turn = obs.turn
                secondary.stall = 0
                secondary.abort_reason = ""
                return secondary
        secondary.abort_reason = "no_second_region"
        return secondary

    def proposals(self, task: ScoutTask, obs: Observation, legal: list[Action]) -> list[Proposal]:
        if task.target is None or task.source is None:
            return []
        ar, ac = task.target
        sr, sc = task.source
        out: list[Proposal] = []
        for action in legal:
            if action.kind != KIND_MOVE or action.split == 1:
                continue
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
