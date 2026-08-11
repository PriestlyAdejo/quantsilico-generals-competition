"""Internal attack-commitment state machine (soft-gate; not StrategicPhase)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from generals_bot.action import KIND_MOVE, Action
from generals_bot.observation import Observation
from generals_bot.policies.base import Proposal
from generals_bot.protocol import DIRECTIONS


class AttackCommitmentState(StrEnum):
    NONE = "NONE"
    PREPARE = "PREPARE"
    COMMIT = "COMMIT"
    CONVERT = "CONVERT"
    RETREAT = "RETREAT"


@dataclass(frozen=True)
class AttackReadinessConfig:
    """Versioned readiness thresholds for tactical attack V2.

    Defaults (documented readiness_ok heuristic used by the ablation policy):
    - enemy_general_confidence: minimum belief confidence to stay engaged
    - belief_age_max: max turns since EG last confirmed before belief is stale
    - min_attack_stack: mobile army (sendable total) required before COMMIT
    - combat_margin: required sendable surplus vs visible EG army (caller encodes)
    - route_length_max: refuse COMMIT when hunt route longer than this
    - own_general_reserve: leave at least this on own general (caller / garrison)
    - short_horizon_counterattack_risk: soft risk score threshold (caller encodes)
    - dwell_turns_prepare_to_commit: hysteresis — turns in PREPARE before COMMIT
    - dwell_turns_retreat: hysteresis — turns in RETREAT before re-engage

    readiness_ok (policy): hunt_plan advance_army met (not gather_until_*),
    own general not threatened, mobile army >= min_attack_stack, route length OK,
    confidence/age OK, and prepare-dwell satisfied.
    """

    version: int = 1
    enemy_general_confidence: float = 0.55
    belief_age_max: int = 80
    min_attack_stack: int = 18
    combat_margin: int = 3
    route_length_max: int = 28
    own_general_reserve: int = 5
    short_horizon_counterattack_risk: float = 0.65
    dwell_turns_prepare_to_commit: int = 3
    dwell_turns_retreat: int = 4


DEFAULT_ATTACK_READINESS = AttackReadinessConfig()


def update_attack_commitment(
    prev: AttackCommitmentState,
    *,
    known_eg: tuple[int, int] | None,
    eg_confidence: float,
    belief_age: int,
    readiness_ok: bool,
    emergency: bool,
    route_illegal: bool,
    eg_captured: bool,
    terminal: bool,
    combat_margin_negative: bool,
    convert_ready: bool,
    turn: int,
) -> AttackCommitmentState:
    """Advance commitment state. Caller resets on new game; dwell lives in policy state."""
    del turn  # reserved for future time-based gates; dwell handled by caller
    cfg = DEFAULT_ATTACK_READINESS
    belief_invalid = (
        known_eg is None
        or eg_confidence < cfg.enemy_general_confidence
        or belief_age > cfg.belief_age_max
    )

    if eg_captured or terminal or belief_invalid:
        return AttackCommitmentState.NONE

    if emergency or route_illegal:
        if prev in {
            AttackCommitmentState.COMMIT,
            AttackCommitmentState.CONVERT,
            AttackCommitmentState.PREPARE,
            AttackCommitmentState.RETREAT,
        }:
            return AttackCommitmentState.RETREAT
        return AttackCommitmentState.NONE

    if prev == AttackCommitmentState.NONE:
        return AttackCommitmentState.PREPARE

    if prev == AttackCommitmentState.PREPARE:
        if readiness_ok:
            return AttackCommitmentState.COMMIT
        return AttackCommitmentState.PREPARE

    if prev == AttackCommitmentState.COMMIT:
        # COMMIT→RETREAT only when margin is materially negative (caller flag).
        if combat_margin_negative:
            return AttackCommitmentState.RETREAT
        if convert_ready:
            return AttackCommitmentState.CONVERT
        return AttackCommitmentState.COMMIT

    if prev == AttackCommitmentState.CONVERT:
        if combat_margin_negative:
            return AttackCommitmentState.RETREAT
        return AttackCommitmentState.CONVERT

    # RETREAT
    if readiness_ok:
        return AttackCommitmentState.COMMIT
    return AttackCommitmentState.PREPARE


def _dist_to_eg(action: Action, eg: tuple[int, int], *, after: bool) -> int:
    er, ec = eg
    if action.kind != KIND_MOVE:
        return abs(action.row - er) + abs(action.col - ec)
    if after:
        dr, dc = DIRECTIONS[action.direction]
        nr, nc = action.row + dr, action.col + dc
        return abs(nr - er) + abs(nc - ec)
    return abs(action.row - er) + abs(action.col - ec)


def is_approach_enemy_general(p: Proposal) -> bool:
    return p.explanation_code == "approach_enemy_general" or (
        p.option in {"GENERAL_HUNT", "DEATHTOUCH"}
        and p.module in {"general_hunt", "fog_sweep", "general_hunt_plan"}
        and p.hard_priority > 50
        and p.explanation_code
        in {
            "approach_enemy_general",
            "approach_enemy_region",
            "hunt_plan_step",
            "hunt_plan_gather",
            "sweep_into_enemy_or_fog",
        }
    )


def filter_proposals_for_commitment(
    proposals: list[Proposal],
    commitment: AttackCommitmentState,
    *,
    known_eg: tuple[int, int] | None,
    emergency: bool,
) -> list[Proposal]:
    """Apply PREPARE/COMMIT proposal soft-gates without changing StrategicPhase."""
    if commitment == AttackCommitmentState.NONE:
        return proposals

    out: list[Proposal] = []
    for p in proposals:
        if p.option == "IMMEDIATE_TERMINAL_WIN":
            out.append(p)
            continue

        if commitment == AttackCommitmentState.PREPARE:
            # Suppress hard GENERAL_HUNT approach priorities above 50 / approach proposals.
            if is_approach_enemy_general(p) or (
                p.option in {"GENERAL_HUNT", "DEATHTOUCH"} and p.hard_priority > 50
            ):
                # Demote rather than drop terminal-adjacent touch proposals with pri>=95
                if p.explanation_code == "touch_enemy_general":
                    out.append(p)
                    continue
                demoted = Proposal(
                    action=p.action,
                    option=p.option,
                    module=p.module,
                    hard_priority=min(p.hard_priority, 45),
                    score=p.score * 0.25,
                    confidence=p.confidence * 0.5,
                    explanation_code=p.explanation_code,
                    explanation_values={
                        **p.explanation_values,
                        "attack_commitment_demote": 1.0,
                    },
                    rejection_reasons=p.rejection_reasons + ("prepare_soft_gate",),
                )
                out.append(demoted)
                continue
            # Allow contextual castles (BUILD) during PREPARE.
            out.append(p)
            continue

        if commitment == AttackCommitmentState.COMMIT and not emergency:
            # Strip off-route COLLECT that increases distance to EG.
            if p.module == "collection" and known_eg is not None and p.action.kind == KIND_MOVE:
                if _dist_to_eg(p.action, known_eg, after=True) > _dist_to_eg(
                    p.action, known_eg, after=False
                ):
                    continue
            # Strip BUILD during commit unless emergency.
            if p.option == "BUILD" or p.module == "castle":
                continue
            out.append(p)
            continue

        if commitment == AttackCommitmentState.RETREAT:
            # Allow defence; demote aggressive hunt.
            if p.option in {"GENERAL_HUNT", "DEATHTOUCH"} and p.hard_priority > 50:
                if p.explanation_code != "touch_enemy_general":
                    continue
            out.append(p)
            continue

        # CONVERT / emergency COMMIT: pass through (defence allowed).
        out.append(p)
    return out


def evaluate_readiness_ok(
    *,
    cfg: AttackReadinessConfig,
    eg_confidence: float,
    belief_age: int,
    mobile_army: int,
    route_length: int,
    gathering: bool,
    own_general_threatened: bool,
    prepare_dwell: int,
    counterattack_risk: float = 0.0,
) -> bool:
    """Rough readiness heuristic aligned with hunt_plan advance_army + reserves."""
    if eg_confidence < cfg.enemy_general_confidence:
        return False
    if belief_age > cfg.belief_age_max:
        return False
    if own_general_threatened:
        return False
    if gathering:
        return False
    if mobile_army < cfg.min_attack_stack:
        return False
    if route_length <= 0 or route_length > cfg.route_length_max:
        return False
    if counterattack_risk > cfg.short_horizon_counterattack_risk:
        return False
    if prepare_dwell < cfg.dwell_turns_prepare_to_commit:
        return False
    return True


def mobile_army_total(obs: Observation) -> int:
    from generals_bot.protocol import OWNER_ME

    mobile = 0
    for r in range(obs.height):
        for c in range(obs.width):
            if obs.owner_grid[r][c] != OWNER_ME:
                continue
            a = int(obs.army_grid[r][c])
            if a > 1:
                mobile += a - 1
    return mobile
