"""Persistent deterministic exploration planner for qualification heuristics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from generals_bot.map_memory import MapMemory
from generals_bot.observation import Observation
from generals_bot.protocol import (
    DIRECTIONS,
    OWNER_ME,
    OWNER_OPP,
    TYPE_FOG,
    TYPE_MOUNTAIN,
    TYPE_STRUCTURE_IN_FOG,
)

STALL_TURNS = 25
RECENT_LIMIT = 24
SOFT_FAIL_COOLDOWN = 45
REGION_ATTEMPT_DECAY_TURNS = 80


@dataclass
class FogRegion:
    region_id: int
    cells: list[tuple[int, int]]
    size: int
    candidate_mass: int
    near_enemy: bool
    frontier: list[tuple[int, int]]


@dataclass
class ExplorationState:
    current_target: tuple[int, int] | None = None
    target_region_id: int | None = None
    assigned_turn: int = 0
    stalled_turns: int = 0
    last_reveal_turn: int = 0
    last_newly_scouted_turn: int = 0
    scout_stall: int = 0
    recent_cells: list[tuple[int, int]] = field(default_factory=list)
    # Soft cell cooldowns only — never a permanent blacklist.
    soft_fail_until: dict[tuple[int, int], int] = field(default_factory=dict)
    region_attempts: dict[int, int] = field(default_factory=dict)
    region_attempt_turn: dict[int, int] = field(default_factory=dict)
    prior_regions: list[int] = field(default_factory=list)
    last_change_reason: str = "init"
    unresolved_regions: int = 0
    target_progress: int = 0

    def to_diagnostics(self) -> dict:
        return {
            "scout_target": self.current_target,
            "target_region_id": self.target_region_id,
            "stalled_turns": self.stalled_turns,
            "scout_stall": self.scout_stall,
            "last_reveal_turn": self.last_reveal_turn,
            "last_newly_scouted_turn": self.last_newly_scouted_turn,
            "unresolved_regions": self.unresolved_regions,
            "target_change_reason": self.last_change_reason,
            "soft_fail_active": sum(1 for until in self.soft_fail_until.values() if until > 0),
            "region_attempt_keys": sorted(self.region_attempts.keys()),
        }

    def note_soft_fail(self, cell: tuple[int, int], turn: int) -> None:
        self.soft_fail_until[cell] = turn + SOFT_FAIL_COOLDOWN

    def is_soft_failed(self, cell: tuple[int, int], turn: int) -> bool:
        until = self.soft_fail_until.get(cell)
        if until is None:
            return False
        if turn >= until:
            self.soft_fail_until.pop(cell, None)
            return False
        return True

    def bump_region_attempt(self, region_id: int, turn: int) -> None:
        self.region_attempts[region_id] = self.region_attempts.get(region_id, 0) + 1
        self.region_attempt_turn[region_id] = turn

    def prune_expired(self, turn: int) -> None:
        expired = [c for c, until in self.soft_fail_until.items() if turn >= until]
        for c in expired:
            self.soft_fail_until.pop(c, None)


def decayed_region_attempts(state: ExplorationState, region_id: int, turn: int) -> int:
    raw = state.region_attempts.get(region_id, 0)
    if raw <= 0:
        return 0
    last = state.region_attempt_turn.get(region_id, turn)
    decay = max(0, (turn - last) // REGION_ATTEMPT_DECAY_TURNS)
    return max(0, raw - decay)


def score_region(
    region: FogRegion,
    *,
    last_enemy: tuple[int, int] | None,
    stacks: list[tuple[int, int, int]],
    attempts: int,
    turn: int,
) -> float:
    score = float(region.candidate_mass) * 5.0 + float(region.size) * 0.15
    if region.near_enemy:
        score += 80.0
    if last_enemy is not None and region.frontier:
        fr, fc = region.frontier[0]
        score += 40.0 / (1.0 + abs(fr - last_enemy[0]) + abs(fc - last_enemy[1]))
    if stacks and region.frontier:
        fr, fc = region.frontier[0]
        best = min(abs(sr - fr) + abs(sc - fc) for sr, sc, _ in stacks)
        score += 30.0 / (1.0 + best)
        score += 0.05 * max(a for _, _, a in stacks)
    if turn >= 800:
        score += 100.0 / max(1, region.size)
    if turn >= 1050:
        score += 150.0 / max(1, region.size)
    score -= attempts * 25.0
    return score


# Back-compat alias.
_score_region = score_region


def _passable_unknown(memory: MapMemory, obs: Observation, r: int, c: int) -> bool:
    if not (0 <= r < obs.height and 0 <= c < obs.width):
        return False
    if memory.known_terrain[r][c] == TYPE_MOUNTAIN:
        return False
    t = obs.type_grid[r][c]
    if t == TYPE_STRUCTURE_IN_FOG:
        return False
    if t == TYPE_FOG:
        return True
    if not memory.ever_seen[r][c]:
        return True
    return False


def partition_fog_regions(
    obs: Observation,
    memory: MapMemory,
    gen_mask: list[list[bool]],
) -> list[FogRegion]:
    """Connected components of unresolved fog / never-seen passable cells."""
    h, w = obs.height, obs.width
    visited = [[False] * w for _ in range(h)]
    regions: list[FogRegion] = []
    rid = 0
    for r in range(h):
        for c in range(w):
            if visited[r][c] or not _passable_unknown(memory, obs, r, c):
                continue
            cells: list[tuple[int, int]] = []
            q = deque([(r, c)])
            visited[r][c] = True
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRECTIONS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc]:
                        if _passable_unknown(memory, obs, nr, nc):
                            visited[nr][nc] = True
                            q.append((nr, nc))
            frontier: list[tuple[int, int]] = []
            candidate_mass = 0
            near_enemy = False
            for cr, cc in cells:
                if gen_mask[cr][cc]:
                    candidate_mass += 1
                for dr, dc in DIRECTIONS:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if obs.owner_grid[nr][nc] == OWNER_ME:
                        frontier.append((cr, cc))
                    if obs.owner_grid[nr][nc] == OWNER_OPP or memory.last_owner[nr][nc] == OWNER_OPP:
                        near_enemy = True
            # unique frontier
            frontier = list(dict.fromkeys(frontier))
            regions.append(
                FogRegion(
                    region_id=rid,
                    cells=cells,
                    size=len(cells),
                    candidate_mass=candidate_mass,
                    near_enemy=near_enemy,
                    frontier=frontier or cells[:1],
                )
            )
            rid += 1
    return regions


def _mobile_stacks(obs: Observation) -> list[tuple[int, int, int]]:
    stacks: list[tuple[int, int, int]] = []
    for r in range(obs.height):
        for c in range(obs.width):
            if obs.owner_grid[r][c] != OWNER_ME:
                continue
            a = obs.army_grid[r][c]
            if a > 1:
                stacks.append((r, c, a))
    stacks.sort(key=lambda x: -x[2])
    return stacks


class ExplorationPlanner:
    """Persistent scout-target state machine with stall recovery and anti-loop memory."""

    def __init__(self, stall_turns: int = STALL_TURNS) -> None:
        self.stall_turns = stall_turns

    def update(
        self,
        state: ExplorationState,
        obs: Observation,
        memory: MapMemory,
        gen_mask: list[list[bool]],
        *,
        last_enemy: tuple[int, int] | None,
        newly_revealed: bool,
        enemy_general_known: bool,
    ) -> ExplorationState:
        if enemy_general_known:
            state.current_target = None
            state.target_region_id = None
            state.last_change_reason = "enemy_general_known"
            state.unresolved_regions = 0
            return state

        if newly_revealed:
            state.last_reveal_turn = obs.turn
            state.last_newly_scouted_turn = obs.turn
            state.stalled_turns = 0
            state.scout_stall = 0
            state.target_progress += 1
        else:
            state.stalled_turns = obs.turn - state.last_reveal_turn
            state.scout_stall = state.stalled_turns

        state.prune_expired(obs.turn)
        regions = partition_fog_regions(obs, memory, gen_mask)
        state.unresolved_regions = len(regions)
        stacks = _mobile_stacks(obs)

        # Invalidate target if cleared / invalid
        if state.current_target is not None:
            tr, tc = state.current_target
            if not _passable_unknown(memory, obs, tr, tc):
                state.soft_fail_until.pop(state.current_target, None)
                state.current_target = None
                state.target_region_id = None
                state.last_change_reason = "target_cleared_or_invalid"

        # Stall recovery — soft cooldown only
        stalled = state.stalled_turns >= self.stall_turns and obs.turn > 40
        if stalled and state.current_target is not None:
            state.note_soft_fail(state.current_target, obs.turn)
            if state.target_region_id is not None:
                state.bump_region_attempt(state.target_region_id, obs.turn)
            state.current_target = None
            state.target_region_id = None
            state.last_change_reason = "stall_recovery"
            state.stalled_turns = 0
            state.last_reveal_turn = obs.turn  # reset stall clock after switch
            state.scout_stall = 0

        # Retain existing valid target
        if state.current_target is not None:
            return state

        if not regions:
            state.last_change_reason = "no_unresolved_regions"
            return state

        scored = sorted(
            regions,
            key=lambda reg: -score_region(
                reg,
                last_enemy=last_enemy,
                stacks=stacks,
                attempts=decayed_region_attempts(state, reg.region_id, obs.turn),
                turn=obs.turn,
            ),
        )
        # Prefer frontiers not recently visited / not soft-failed
        for reg in scored:
            candidates = list(reg.frontier) + list(reg.cells[:3])
            for cell in candidates:
                if state.is_soft_failed(cell, obs.turn):
                    continue
                if cell in state.recent_cells[-8:]:
                    continue
                state.current_target = cell
                state.target_region_id = reg.region_id
                state.assigned_turn = obs.turn
                state.target_progress = 0
                state.last_change_reason = "new_region_target"
                return state
        # Fallback: best region frontier even if recently visited
        reg = scored[0]
        cell = reg.frontier[0] if reg.frontier else reg.cells[0]
        state.current_target = cell
        state.target_region_id = reg.region_id
        state.assigned_turn = obs.turn
        state.last_change_reason = "fallback_region_target"
        return state

    def note_move(self, state: ExplorationState, cell: tuple[int, int]) -> None:
        state.recent_cells.append(cell)
        if len(state.recent_cells) > RECENT_LIMIT:
            state.recent_cells = state.recent_cells[-RECENT_LIMIT:]
