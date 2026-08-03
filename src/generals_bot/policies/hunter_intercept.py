"""Narrow Hunter-style corridor interception — not a global emergency mode."""

from __future__ import annotations

from generals_bot.action import KIND_MOVE, Action
from generals_bot.observation import Observation
from generals_bot.policies.base import Proposal
from generals_bot.policies.general_garrison import find_own_general
from generals_bot.protocol import DIRECTIONS, OWNER_ME, OWNER_OPP


def visible_enemies_near_general(
    obs: Observation, gr: int, gc: int, *, max_dist: int = 6
) -> list[tuple[int, int, int, int]]:
    """Return (r, c, dist, army) for visible enemies within max_dist of general."""
    out: list[tuple[int, int, int, int]] = []
    for r in range(obs.height):
        for c in range(obs.width):
            if obs.owner_grid[r][c] != OWNER_OPP:
                continue
            d = abs(r - gr) + abs(c - gc)
            if d <= max_dist:
                out.append((r, c, d, obs.army_grid[r][c]))
    out.sort(key=lambda x: (x[2], -x[3]))
    return out


def intercept_proposals(obs: Observation, legal: list[Action]) -> list[Proposal]:
    """Legacy intercept used by ablation matrix (keep for reproducibility)."""
    gen = find_own_general(obs)
    if gen is None:
        return []
    gr, gc = gen
    enemies = visible_enemies_near_general(obs, gr, gc, max_dist=6)
    if not enemies:
        return []
    er, ec, edist, earmy = enemies[0]
    if edist > 5 and earmy < 3:
        return []
    out: list[Proposal] = []
    for action in legal:
        if action.kind != KIND_MOVE:
            continue
        if (action.row, action.col) == (gr, gc):
            gen_army = obs.army_grid[gr][gc]
            sendable = gen_army - 1 if action.split == 0 else gen_army // 2
            if gen_army - sendable < 4:
                continue
        dr, dc = DIRECTIONS[action.direction]
        nr, nc = action.row + dr, action.col + dc
        dist_src_e = abs(action.row - er) + abs(action.col - ec)
        dist_dst_e = abs(nr - er) + abs(nc - ec)
        dist_dst_g = abs(nr - gr) + abs(nc - gc)
        toward_enemy = dist_dst_e < dist_src_e
        between = dist_dst_e + dist_dst_g <= edist + 1
        if not (toward_enemy or between or (nr, nc) == (er, ec)):
            continue
        src_army = obs.army_grid[action.row][action.col]
        sendable = src_army - 1 if action.split == 0 else src_army // 2
        if sendable < 1:
            continue
        split_bonus = 30.0 if action.split == 1 and src_army >= 8 else 0.0
        out.append(
            Proposal(
                action=action,
                option="INTERCEPT",
                module="hunter_intercept",
                hard_priority=85 if edist <= 3 else 65,
                score=900.0 + sendable * 8 + split_bonus + 40.0 * max(0, dist_src_e - dist_dst_e),
                confidence=0.8,
                explanation_code="intercept_rush_corridor",
                explanation_values={"enemy_dist": float(edist), "enemy_army": float(earmy)},
            )
        )
    return out


def _on_corridor(nr: int, nc: int, er: int, ec: int, gr: int, gc: int) -> bool:
    return abs(nr - er) + abs(nr - gr) + abs(nc - ec) + abs(nc - gc) == abs(er - gr) + abs(ec - gc)


def corridor_intercept_proposals_v2(obs: Observation, legal: list[Action]) -> list[Proposal]:
    """Block/contest one corridor cell; avoid one-army feeds; keep scouts free."""
    gen = find_own_general(obs)
    if gen is None:
        return []
    gr, gc = gen
    enemies = visible_enemies_near_general(obs, gr, gc, max_dist=7)
    enemies = [e for e in enemies if e[3] >= 2]
    if not enemies:
        return []
    er, ec, edist, earmy = enemies[0]
    if edist > 6:
        return []
    if edist > 4 and earmy < 5:
        return []

    out: list[Proposal] = []
    for action in legal:
        if action.kind != KIND_MOVE:
            continue
        src_army = obs.army_grid[action.row][action.col]
        sendable = src_army - 1 if action.split == 0 else src_army // 2
        if sendable < 2:
            continue
        if (action.row, action.col) == (gr, gc):
            if src_army - sendable < 5:
                continue
            if action.split == 0 and src_army >= 10:
                continue
        dr, dc = DIRECTIONS[action.direction]
        nr, nc = action.row + dr, action.col + dc
        dist_src_e = abs(action.row - er) + abs(action.col - ec)
        dist_dst_e = abs(nr - er) + abs(nc - ec)
        toward = dist_dst_e < dist_src_e
        corridor = _on_corridor(nr, nc, er, ec, gr, gc)
        if not (toward or corridor or (nr, nc) == (er, ec)):
            continue
        merge_bonus = 0.0
        if obs.owner_grid[nr][nc] == OWNER_ME and obs.army_grid[nr][nc] >= 2:
            merge_bonus = 80.0
        split_bonus = 25.0 if action.split == 1 and src_army >= 8 else 0.0
        pri = 88 if edist <= 3 else (72 if edist <= 5 else 55)
        out.append(
            Proposal(
                action=action,
                option="CORRIDOR_INTERCEPT",
                module="hunter_intercept_v2",
                hard_priority=pri,
                score=950.0
                + sendable * 10
                + merge_bonus
                + split_bonus
                + (50.0 if corridor else 0.0)
                + 35.0 * max(0, dist_src_e - dist_dst_e),
                confidence=0.85,
                explanation_code="corridor_intercept_v2",
                explanation_values={"enemy_dist": float(edist), "enemy_army": float(earmy)},
            )
        )
    return out
