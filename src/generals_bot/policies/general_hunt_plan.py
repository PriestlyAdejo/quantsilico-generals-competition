"""Persistent General Hunt plan after enemy general discovery."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

from generals_bot.action import KIND_MOVE, Action
from generals_bot.observation import Observation
from generals_bot.policies.base import Proposal
from generals_bot.policies.general_garrison import find_own_general
from generals_bot.protocol import DIRECTIONS, OWNER_ME, OWNER_OPP, TYPE_MOUNTAIN, TYPE_STRUCTURE_IN_FOG
from generals_bot.rules import DEATHTOUCH_TURN, DRAW_TURN


@dataclass
class GeneralHuntPlan:
    general: tuple[int, int] | None = None
    discovery_turn: int = 0
    source: tuple[int, int] | None = None
    route: list[tuple[int, int]] = field(default_factory=list)
    route_hash: str = ""
    route_index: int = 0
    last_progress_turn: int = 0
    replan_count: int = 0
    blocked_reason: str = ""
    deathtouch: bool = False
    active: bool = False

    def to_diagnostics(self) -> dict:
        d = asdict(self)
        d["route_len"] = len(self.route)
        return d


def _passable_known(obs: Observation, r: int, c: int) -> bool:
    t = obs.type_grid[r][c]
    return t not in (TYPE_MOUNTAIN, TYPE_STRUCTURE_IN_FOG)


def _bfs_route(
    obs: Observation,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    if start == goal:
        return [start]
    h, w = obs.height, obs.width
    q = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in parent:
                continue
            if not _passable_known(obs, nr, nc):
                continue
            parent[(nr, nc)] = (r, c)
            if (nr, nc) == goal:
                path = [(nr, nc)]
                cur: tuple[int, int] | None = (r, c)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            q.append((nr, nc))
    return []


def _route_hash(route: list[tuple[int, int]]) -> str:
    return "|".join(f"{r},{c}" for r, c in route)


def _pick_attack_source(
    obs: Observation,
    goal: tuple[int, int],
    *,
    prefer_near: bool = False,
    min_army: int = 2,
) -> tuple[int, int] | None:
    gen = find_own_general(obs)
    best = None
    best_key = None
    for r in range(obs.height):
        for c in range(obs.width):
            if obs.owner_grid[r][c] != OWNER_ME:
                continue
            army = obs.army_grid[r][c]
            if army < min_army:
                continue
            if gen is not None and (r, c) == gen and army < 8:
                continue
            dist = abs(r - goal[0]) + abs(c - goal[1])
            key = (dist, -army, r, c) if prefer_near else (-army, dist, r, c)
            if best_key is None or key < best_key:
                best_key = key
                best = (r, c)
    return best


def _visible_enemy_general_army(obs: Observation, goal: tuple[int, int]) -> int:
    gr, gc = goal
    if 0 <= gr < obs.height and 0 <= gc < obs.width and obs.owner_grid[gr][gc] == OWNER_OPP:
        return int(obs.army_grid[gr][gc])
    return 0


def update_hunt_plan(
    plan: GeneralHuntPlan,
    obs: Observation,
    *,
    known_general: tuple[int, int] | None,
    emergency: bool,
) -> GeneralHuntPlan:
    if known_general is None:
        plan.active = False
        plan.blocked_reason = "no_known_general"
        return plan

    if plan.general is None:
        plan.general = known_general
        plan.discovery_turn = obs.turn
        plan.active = True
        plan.last_progress_turn = obs.turn
    else:
        plan.general = known_general
        plan.active = True

    plan.deathtouch = obs.turn >= DEATHTOUCH_TURN
    if emergency:
        plan.blocked_reason = "own_general_emergency"
        return plan

    goal = plan.general
    assert goal is not None
    eg_army = _visible_enemy_general_army(obs, goal)
    # Before Deathtouch, gather a decisive stack; after, tip with any remaining army.
    advance_army = 2 if plan.deathtouch else max(15, eg_army + 3)

    need_replan = False
    reason = ""
    if not plan.route or plan.source is None:
        need_replan = True
        reason = "initial"
    else:
        sr, sc = plan.source
        source_weak = obs.owner_grid[sr][sc] != OWNER_ME or obs.army_grid[sr][sc] <= 1
        if source_weak:
            best = None
            best_a = 0
            best_i = plan.route_index
            for i, cell in enumerate(plan.route):
                if i < max(0, plan.route_index - 2):
                    continue
                r, c = cell
                if obs.owner_grid[r][c] != OWNER_ME:
                    continue
                a = obs.army_grid[r][c]
                if a > best_a:
                    best_a = a
                    best = cell
                    best_i = i
            if best is not None and best_a > 1:
                plan.source = best
                plan.route_index = best_i
                plan.last_progress_turn = obs.turn
                plan.blocked_reason = "source_followed_along_route"
            else:
                need_replan = True
                reason = "source_lost"
        if not need_replan and plan.route_index + 1 < len(plan.route):
            nxt = plan.route[plan.route_index + 1]
            if not _passable_known(obs, nxt[0], nxt[1]):
                need_replan = True
                reason = "next_edge_invalid"

    if need_replan:
        prefer_near = reason.startswith("source_lost")
        src = _pick_attack_source(obs, goal, prefer_near=prefer_near, min_army=3)
        plan.source = src
        if src is None:
            plan.route = []
            plan.route_hash = ""
            plan.route_index = 0
            plan.blocked_reason = "no_attack_source"
            plan.replan_count += 1
            return plan
        route = _bfs_route(obs, src, goal)
        if not route:
            plan.route = []
            plan.route_hash = ""
            plan.route_index = 0
            plan.blocked_reason = "no_reachable_route"
            plan.replan_count += 1
            return plan
        plan.route = route
        plan.route_hash = _route_hash(route)
        plan.route_index = 0
        plan.replan_count += 1
        plan.blocked_reason = f"replan:{reason}" if reason else ""
        plan.last_progress_turn = obs.turn

    if plan.source is not None and plan.route and plan.source in plan.route:
        plan.route_index = plan.route.index(plan.source)

    if plan.source is not None:
        sr, sc = plan.source
        if obs.army_grid[sr][sc] < advance_army and not plan.deathtouch:
            plan.blocked_reason = f"gather_until_{advance_army}"
        elif not plan.blocked_reason.startswith("replan") and not plan.blocked_reason.startswith("source"):
            plan.blocked_reason = ""
    return plan


def hunt_plan_proposals(plan: GeneralHuntPlan, obs: Observation, legal: list[Action]) -> list[Proposal]:
    if not plan.active or plan.general is None or plan.source is None or not plan.route:
        return []
    if plan.blocked_reason == "own_general_emergency":
        return []

    sr, sc = plan.source
    goal = plan.general
    idx = plan.route_index
    target = goal if idx >= len(plan.route) - 1 else plan.route[min(idx + 1, len(plan.route) - 1)]

    late = obs.turn >= DRAW_TURN - 150
    very_late = obs.turn >= DRAW_TURN - 50
    pri = 112 if very_late else (110 if late else 105)
    gathering = plan.blocked_reason.startswith("gather_until_")

    out: list[Proposal] = []
    for action in legal:
        if action.kind != KIND_MOVE:
            continue
        dr, dc = DIRECTIONS[action.direction]
        nr, nc = action.row + dr, action.col + dc
        src_army = obs.army_grid[action.row][action.col]
        sendable = src_army - 1 if action.split == 0 else src_army // 2
        if sendable < 1:
            continue

        if gathering:
            if (action.row, action.col) == (sr, sc):
                continue
            dist_src = abs(action.row - sr) + abs(action.col - sc)
            dist_dst = abs(nr - sr) + abs(nc - sc)
            if dist_dst >= dist_src:
                continue
            out.append(
                Proposal(
                    action=action,
                    option="GENERAL_HUNT",
                    module="general_hunt_plan",
                    hard_priority=pri - 1,
                    score=15_000.0 + sendable * 6 + 100.0 * (dist_src - dist_dst),
                    confidence=0.85,
                    explanation_code="hunt_plan_gather",
                )
            )
            continue

        if (action.row, action.col) != (sr, sc):
            continue
        if (nr, nc) == goal:
            out.append(
                Proposal(
                    action=action,
                    option="GENERAL_HUNT",
                    module="general_hunt_plan",
                    hard_priority=pri + 1,
                    score=50_000.0 + sendable,
                    confidence=0.95,
                    explanation_code="hunt_plan_touch",
                )
            )
            continue
        dist_src = abs(action.row - goal[0]) + abs(action.col - goal[1])
        dist_dst = abs(nr - goal[0]) + abs(nc - goal[1])
        on_route = (nr, nc) == target or (nr, nc) in plan.route[idx : idx + 3]
        if dist_dst >= dist_src and not on_route:
            continue
        if action.split == 1 and not very_late and src_army < 12:
            continue
        out.append(
            Proposal(
                action=action,
                option="GENERAL_HUNT",
                module="general_hunt_plan",
                hard_priority=pri,
                score=20_000.0
                + sendable * 8
                + (500.0 if on_route else 0.0)
                + 80.0 * max(0, dist_src - dist_dst),
                confidence=0.9,
                explanation_code="hunt_plan_step",
                explanation_values={"route_index": float(idx), "replan_count": float(plan.replan_count)},
            )
        )
    return out
