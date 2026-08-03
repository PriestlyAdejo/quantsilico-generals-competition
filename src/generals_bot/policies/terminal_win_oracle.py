"""Engine-assisted terminal-win oracle for known-general finishing moves."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from generals_bot.action import KIND_MOVE, PASS_ACTION, Action
from generals_bot.legal import enumerate_legal_actions, is_legal_action
from generals_bot.observation import Observation
from generals_bot.policies.base import Proposal
from generals_bot.protocol import DIRECTIONS, OWNER_OPP
from generals_bot.rules import DEATHTOUCH_TURN


@dataclass(frozen=True)
class TerminalWinCandidate:
    action: Action
    enemy_general: tuple[int, int]
    deathtouch_active: bool
    sendable: int
    dest_army: int
    verified_by_engine: bool
    reason: str


def _action_to_jax(action: Action) -> jnp.ndarray:
    return jnp.array(
        [action.kind, action.row, action.col, action.direction, action.split],
        dtype=jnp.int32,
    )


def _sendable(obs: Observation, action: Action) -> int:
    src = obs.army_grid[action.row][action.col]
    return src - 1 if action.split == 0 else src // 2


def _dest(_obs: Observation, action: Action) -> tuple[int, int]:
    dr, dc = DIRECTIONS[action.direction]
    return action.row + dr, action.col + dc


def heuristic_touch_wins(
    obs: Observation,
    action: Action,
    enemy_general: tuple[int, int],
) -> bool:
    """Rule-level check without stepping the engine (unit-test friendly)."""
    if action.kind != KIND_MOVE:
        return False
    if not is_legal_action(obs, action):
        return False
    nr, nc = _dest(obs, action)
    if (nr, nc) != enemy_general:
        return False
    sendable = _sendable(obs, action)
    if sendable < 1:
        return False
    dt = obs.turn >= DEATHTOUCH_TURN
    dest_army = 0
    if obs.owner_grid[nr][nc] == OWNER_OPP:
        dest_army = obs.army_grid[nr][nc]
    if dt:
        return True
    return sendable > dest_army


def find_immediate_touch_actions(
    obs: Observation,
    enemy_general: tuple[int, int] | None,
    *,
    legal: list[Action] | None = None,
) -> list[TerminalWinCandidate]:
    if enemy_general is None:
        return []
    legal = legal if legal is not None else enumerate_legal_actions(obs)
    dt = obs.turn >= DEATHTOUCH_TURN
    out: list[TerminalWinCandidate] = []
    for action in legal:
        if not heuristic_touch_wins(obs, action, enemy_general):
            continue
        nr, nc = _dest(obs, action)
        dest_army = obs.army_grid[nr][nc] if obs.owner_grid[nr][nc] == OWNER_OPP else 0
        out.append(
            TerminalWinCandidate(
                action=action,
                enemy_general=enemy_general,
                deathtouch_active=dt,
                sendable=_sendable(obs, action),
                dest_army=dest_army,
                verified_by_engine=False,
                reason="heuristic_touch",
            )
        )
    out.sort(
        key=lambda c: (-c.sendable, c.action.split, c.action.row, c.action.col, c.action.direction)
    )
    return out


def verify_touch_with_opponent_pass(
    state,
    *,
    perspective: int,
    action: Action,
) -> bool:
    """Step official transition with opponent PASS; true if perspective wins."""
    from generals import GeneralsEnv

    from generals_bot.evaluation.match import make_transition

    env = GeneralsEnv(mode="competition")
    transition = make_transition(env)
    mine = _action_to_jax(action)
    foe = _action_to_jax(PASS_ACTION)
    stacked = jnp.stack([mine, foe] if perspective == 0 else [foe, mine])
    _new_state, info = transition(state, stacked)
    return bool(info.is_done) and int(info.winner) == perspective


def immediate_terminal_win_proposals(
    obs: Observation,
    enemy_general: tuple[int, int] | None,
    *,
    legal: list[Action] | None = None,
    hard_priority: int = 110,
) -> list[Proposal]:
    """Hard proposals that must outrank DEFEND(100) and scouting."""
    cands = find_immediate_touch_actions(obs, enemy_general, legal=legal)
    out: list[Proposal] = []
    for c in cands:
        out.append(
            Proposal(
                action=c.action,
                option="IMMEDIATE_TERMINAL_WIN",
                module="terminal_win_oracle",
                hard_priority=hard_priority,
                score=1_000_000.0 + float(c.sendable),
                confidence=1.0,
                explanation_code="immediate_terminal_win",
                explanation_values={
                    "deathtouch": float(c.deathtouch_active),
                    "sendable": float(c.sendable),
                    "dest_army": float(c.dest_army),
                    "combat_margin_bypass": 1.0 if c.deathtouch_active else 0.0,
                },
            )
        )
    return out
