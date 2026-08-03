"""Explicit strategic phase controller for qualification heuristics."""

from __future__ import annotations

from enum import StrEnum

from generals_bot.observation import Observation
from generals_bot.rules import DEATHTOUCH_TURN, DRAW_TURN


class StrategicPhase(StrEnum):
    OPENING = "OPENING"
    EXPANSION = "EXPANSION"
    SEARCH_FOR_CONTACT = "SEARCH_FOR_CONTACT"
    CONSOLIDATION = "CONSOLIDATION"
    CONTACT = "CONTACT"
    CLEAR_ENEMY_REGION = "CLEAR_ENEMY_REGION"
    CONVERSION = "CONVERSION"
    GENERAL_HUNT = "GENERAL_HUNT"
    DEATHTOUCH_HUNT = "DEATHTOUCH_HUNT"
    DRAW_AVOIDANCE = "DRAW_AVOIDANCE"
    DEFENSIVE_CAUTION = "DEFENSIVE_CAUTION"
    EMERGENCY_DEFENCE = "EMERGENCY_DEFENCE"


def is_dominant_position(*, my_land: int, opp_land: int, passable: int | None = None) -> bool:
    """True when own land share implies Expander conversion should dominate expansion."""
    total = my_land + opp_land
    if total <= 0:
        return False
    if my_land / total >= 0.65:
        return True
    if opp_land <= 20 and my_land > opp_land * 2:
        return True
    if passable and my_land >= 0.65 * passable:
        return True
    return False


def select_phase(
    obs: Observation,
    *,
    prev: StrategicPhase | None,
    enemy_contact: bool,
    enemy_general_known: bool,
    own_general_threatened: bool,
    defensive_caution: bool = False,
    dominant: bool,
    mobile_ratio: float,
    candidate_mask_size: int,
    contact_lost: bool = False,
) -> tuple[StrategicPhase, str]:
    """Return (phase, reason). Mandatory transitions ignore option stickiness."""
    # Full emergency only from credible threat assessment (caller sets flag)
    if own_general_threatened:
        return StrategicPhase.EMERGENCY_DEFENCE, "credible_threat_to_own_general"
    # Caution does not cancel conversion/hunt — applied as soft overlay by caller when needed
    if obs.turn >= 1150 and not enemy_general_known:
        return StrategicPhase.DRAW_AVOIDANCE, "turn_ge_1150_unresolved"
    if obs.turn >= 1050 and not enemy_general_known:
        return StrategicPhase.DRAW_AVOIDANCE, "turn_ge_1050_draw_avoidance"
    if obs.turn >= DEATHTOUCH_TURN and enemy_general_known:
        return StrategicPhase.DEATHTOUCH_HUNT, "deathtouch_known_general"
    if obs.turn >= DEATHTOUCH_TURN and candidate_mask_size > 0:
        return StrategicPhase.DEATHTOUCH_HUNT, "deathtouch_candidate_sweep"
    if enemy_general_known:
        return StrategicPhase.GENERAL_HUNT, "enemy_general_visible_or_known"
    if dominant and enemy_contact:
        return StrategicPhase.CONVERSION, "dominant_position_conversion"
    if enemy_contact and contact_lost:
        return StrategicPhase.CLEAR_ENEMY_REGION, "contact_lost_clear_last_region"
    if enemy_contact:
        return StrategicPhase.CONTACT, "enemy_tile_or_route_observed"
    if defensive_caution and prev not in {
        StrategicPhase.DEATHTOUCH_HUNT,
        StrategicPhase.GENERAL_HUNT,
        StrategicPhase.CONVERSION,
    }:
        # Soft label only when not already converting
        pass
    if not enemy_contact and obs.turn >= 150 and obs.opp_land > 0 and obs.my_land >= 25:
        return StrategicPhase.SEARCH_FOR_CONTACT, "no_contact_search"
    if mobile_ratio < 0.2 and obs.turn > 80:
        return StrategicPhase.CONSOLIDATION, "army_too_dispersed"
    if obs.turn < 40:
        return StrategicPhase.OPENING, "opening_route"
    if obs.turn >= DRAW_TURN - 50:
        return StrategicPhase.DRAW_AVOIDANCE, "near_draw_deadline"
    if defensive_caution:
        return StrategicPhase.DEFENSIVE_CAUTION, "uncertain_threat_caution"
    return StrategicPhase.EXPANSION, "default_expansion"
