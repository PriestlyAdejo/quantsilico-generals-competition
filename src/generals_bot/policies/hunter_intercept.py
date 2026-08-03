"""Narrow Hunter-style corridor interception — not a global emergency mode."""

from __future__ import annotations

from generals_bot.action import KIND_MOVE, Action
from generals_bot.observation import Observation
from generals_bot.policies.base import Proposal
from generals_bot.policies.general_garrison import find_own_general
from generals_bot.protocol import DIRECTIONS, OWNER_OPP


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
    """Intercept on the approach corridor; preserve general tile."""
    gen = find_own_general(obs)
    if gen is None:
        return []
    gr, gc = gen
    enemies = visible_enemies_near_general(obs, gr, gc, max_dist=6)
    if not enemies:
        return []
    er, ec, edist, earmy = enemies[0]
    # Only act when a real nearby stack exists
    if edist > 5 and earmy < 3:
        return []
    out: list[Proposal] = []
    for action in legal:
        if action.kind != KIND_MOVE:
            continue
        # Never move the last garrison off the general for intercept
        if (action.row, action.col) == (gr, gc):
            gen_army = obs.army_grid[gr][gc]
            sendable = gen_army - 1 if action.split == 0 else gen_army // 2
            if gen_army - sendable < 4:
                continue
        dr, dc = DIRECTIONS[action.direction]
        nr, nc = action.row + dr, action.col + dc
        # Prefer stepping onto/toward the enemy corridor cell or between enemy and general
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
        # Prefer half-moves from large stacks to keep branching
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
