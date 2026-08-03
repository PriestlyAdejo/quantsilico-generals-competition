"""Lightweight general garrison — does not change strategic phase."""

from __future__ import annotations

from generals_bot.action import KIND_MOVE, Action
from generals_bot.observation import Observation
from generals_bot.policies.base import Proposal
from generals_bot.protocol import DIRECTIONS, OWNER_ME, TYPE_GENERAL


def find_own_general(obs: Observation) -> tuple[int, int] | None:
    for r in range(obs.height):
        for c in range(obs.width):
            if obs.owner_grid[r][c] == OWNER_ME and obs.type_grid[r][c] == TYPE_GENERAL:
                return r, c
    return None


def garrison_reserve(*, turn: int, threatened: bool) -> int:
    """Minimum army to leave on the general tile. Independent of EMERGENCY phase."""
    if threatened:
        return 10
    if turn < 40:
        return 2
    if turn < 200:
        return 4
    return 5


def filter_general_stripping(
    proposals: list[Proposal],
    obs: Observation,
    *,
    reserve: int,
) -> list[Proposal]:
    gen = find_own_general(obs)
    if gen is None:
        return proposals
    gr, gc = gen
    gen_army = obs.army_grid[gr][gc]
    kept: list[Proposal] = []
    for p in proposals:
        a = p.action
        if a.kind == KIND_MOVE and (a.row, a.col) == (gr, gc):
            sendable = gen_army - 1 if a.split == 0 else gen_army // 2
            if gen_army - sendable < reserve:
                continue
        kept.append(p)
    return kept


def reinforcement_proposals(obs: Observation, legal: list[Action], *, threatened: bool) -> list[Proposal]:
    """Soft reinforce toward general without hard_priority that cancels exploration."""
    gen = find_own_general(obs)
    if gen is None:
        return []
    gr, gc = gen
    out: list[Proposal] = []
    for action in legal:
        if action.kind != KIND_MOVE:
            continue
        if (action.row, action.col) == (gr, gc):
            continue
        dr, dc = DIRECTIONS[action.direction]
        nr, nc = action.row + dr, action.col + dc
        dist_src = abs(action.row - gr) + abs(action.col - gc)
        dist_dst = abs(nr - gr) + abs(nc - gc)
        if dist_dst >= dist_src:
            continue
        # Soft priority: below fog_sweep (90) and attacks; above casual expand
        pri = 42 if threatened else 28
        out.append(
            Proposal(
                action=action,
                option="GARRISON",
                module="general_garrison",
                hard_priority=pri,
                score=300.0 + obs.army_grid[action.row][action.col] * 4.0 / max(1, dist_dst + 1),
                confidence=0.7,
                explanation_code="garrison_reinforce_soft",
            )
        )
    return out
