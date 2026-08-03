"""Credible threat assessment with emergency hysteresis and defensive caution."""

from __future__ import annotations

from dataclasses import dataclass

from generals_bot.observation import Observation
from generals_bot.protocol import DIRECTIONS, OWNER_ME, OWNER_OPP, TYPE_GENERAL, TYPE_FOG


@dataclass
class ThreatAssessment:
    emergency: bool
    caution: bool
    confidence: float
    trigger_type: str
    nearest_enemy_dist: int | None
    nearest_enemy_army: int
    estimated_arrival: int | None
    general_reserve: int
    exit_reason: str = ""
    phase_override: str | None = None  # EMERGENCY_DEFENCE | DEFENSIVE_CAUTION | None


@dataclass
class ThreatMemory:
    emergency_active: bool = False
    caution_active: bool = False
    entered_turn: int = -1
    last_evidence_turn: int = -1
    confidence: float = 0.0
    trigger_type: str = ""
    activations: int = 0
    false_emergency_suspects: int = 0

    def to_diagnostics(self) -> dict:
        return {
            "emergency_active": self.emergency_active,
            "caution_active": self.caution_active,
            "threat_confidence": self.confidence,
            "threat_trigger": self.trigger_type,
            "threat_entered_turn": self.entered_turn,
            "threat_last_evidence_turn": self.last_evidence_turn,
            "emergency_activations": self.activations,
        }


def _find_own_general(obs: Observation) -> tuple[int, int] | None:
    for r in range(obs.height):
        for c in range(obs.width):
            if obs.owner_grid[r][c] == OWNER_ME and obs.type_grid[r][c] == TYPE_GENERAL:
                return r, c
    return None


def _nearest_visible_enemy(obs: Observation, gr: int, gc: int) -> tuple[int | None, int]:
    best_d: int | None = None
    best_a = 0
    for r in range(obs.height):
        for c in range(obs.width):
            if obs.owner_grid[r][c] != OWNER_OPP:
                continue
            d = abs(r - gr) + abs(c - gc)
            if best_d is None or d < best_d:
                best_d = d
                best_a = obs.army_grid[r][c]
            elif d == best_d:
                best_a = max(best_a, obs.army_grid[r][c])
    return best_d, best_a


def assess_threat(
    obs: Observation,
    memory: ThreatMemory,
    *,
    emergency_enter_dist: int = 2,
    emergency_enter_army: int = 2,
    caution_dist: int = 4,
    max_emergency_without_evidence: int = 20,
    confidence_decay: float = 0.08,
) -> tuple[ThreatAssessment, ThreatMemory]:
    """Classify EMERGENCY vs CAUTION using visible evidence only (no fog-only triggers)."""
    gen = _find_own_general(obs)
    if gen is None:
        assessment = ThreatAssessment(
            emergency=False,
            caution=False,
            confidence=0.0,
            trigger_type="no_general",
            nearest_enemy_dist=None,
            nearest_enemy_army=0,
            estimated_arrival=None,
            general_reserve=2,
            exit_reason="no_general",
        )
        return assessment, memory

    gr, gc = gen
    dist, army = _nearest_visible_enemy(obs, gr, gc)
    fresh_emergency = False
    fresh_caution = False
    trigger = "none"

    # Credible emergency: visible enemy close with meaningful force
    if dist is not None and dist <= emergency_enter_dist and army >= emergency_enter_army:
        fresh_emergency = True
        trigger = "visible_enemy_adjacent_or_near"
    elif dist is not None and dist == 1:
        # Any adjacent enemy is emergency regardless of small army (source-catching)
        fresh_emergency = True
        trigger = "adjacent_enemy_any_army"

    if dist is not None and dist <= caution_dist and not fresh_emergency:
        fresh_caution = True
        if trigger == "none":
            trigger = "visible_enemy_within_caution_range"

    # Fog alone never starts emergency. Adjacent fog after prior enemy evidence can sustain caution.
    if not fresh_emergency and not fresh_caution and memory.last_evidence_turn >= 0:
        age = obs.turn - memory.last_evidence_turn
        if age <= 8:
            for dr, dc in DIRECTIONS:
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < obs.height and 0 <= nc < obs.width:
                    if obs.type_grid[nr][nc] == TYPE_FOG and memory.caution_active:
                        fresh_caution = True
                        trigger = "recent_threat_adjacent_fog"
                        break

    if fresh_emergency:
        memory.confidence = min(1.0, memory.confidence + 0.45)
        memory.last_evidence_turn = obs.turn
        memory.trigger_type = trigger
        if not memory.emergency_active:
            memory.emergency_active = True
            memory.caution_active = False
            memory.entered_turn = obs.turn
            memory.activations += 1
    elif fresh_caution:
        memory.confidence = min(1.0, memory.confidence + 0.2)
        memory.last_evidence_turn = obs.turn
        memory.trigger_type = trigger
        if not memory.emergency_active:
            memory.caution_active = True
            if memory.entered_turn < 0:
                memory.entered_turn = obs.turn
    else:
        memory.confidence = max(0.0, memory.confidence - confidence_decay)

    exit_reason = ""
    # Hysteresis: stay in emergency with lower bar, but expire without evidence
    if memory.emergency_active:
        if fresh_emergency:
            pass
        elif dist is not None and dist <= caution_dist and memory.confidence >= 0.25:
            # remain briefly while enemy still in caution range
            pass
        elif (
            memory.last_evidence_turn >= 0
            and obs.turn - memory.last_evidence_turn > max_emergency_without_evidence
        ) or memory.confidence < 0.15:
            memory.emergency_active = False
            memory.caution_active = dist is not None and dist <= caution_dist + 1
            exit_reason = "threat_decay_or_timeout"
            memory.confidence = max(0.0, memory.confidence * 0.5)
        # Detect possible false emergency: entered with no subsequent enemy within 3 for long
        if (
            exit_reason
            and memory.entered_turn >= 0
            and obs.turn - memory.entered_turn <= 5
            and (dist is None or dist > 5)
        ):
            memory.false_emergency_suspects += 1

    if not memory.emergency_active and memory.caution_active and not fresh_caution:
        if memory.confidence < 0.1 or (
            memory.last_evidence_turn >= 0 and obs.turn - memory.last_evidence_turn > 30
        ):
            memory.caution_active = False
            exit_reason = exit_reason or "caution_decay"

    # Dynamic reserve
    reserve = 2
    if obs.turn < 120:
        reserve = 2
    elif memory.caution_active:
        reserve = 6
    elif obs.turn < 600:
        reserve = 4
    else:
        reserve = 5
    if memory.emergency_active:
        reserve = max(reserve, 12 if army >= 5 else 8)

    phase_override = None
    if memory.emergency_active:
        phase_override = "EMERGENCY_DEFENCE"
    elif memory.caution_active:
        phase_override = "DEFENSIVE_CAUTION"

    assessment = ThreatAssessment(
        emergency=memory.emergency_active,
        caution=memory.caution_active,
        confidence=memory.confidence,
        trigger_type=memory.trigger_type or trigger,
        nearest_enemy_dist=dist,
        nearest_enemy_army=army,
        estimated_arrival=dist,
        general_reserve=reserve,
        exit_reason=exit_reason,
        phase_override=phase_override,
    )
    return assessment, memory
