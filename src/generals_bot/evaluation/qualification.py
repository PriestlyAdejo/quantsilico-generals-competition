"""Qualification metrics with explicit W/D/L and conversion diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TERMINAL_REASONS = (
    "WIN_GENERAL_CAPTURE",
    "LOSS_GENERAL_CAPTURED",
    "DRAW_TURN_LIMIT",
    "DRAW_OTHER",
    "UNKNOWN",
)

FAILURE_CLASSES = (
    "NO_CONTACT_DRAW",
    "FOG_NOT_CLEARED",
    "GENERAL_NEVER_LOCATED",
    "GENERAL_LOCATED_NOT_REACHED",
    "FRONTIER_STALL",
    "ARMY_TOO_DISPERSED",
    "LARGE_STACK_STRANDED",
    "RISK_GATE_STALL",
    "SHIELD_OVERRIDE_STALL",
    "CASTLE_OVERINVESTMENT",
    "CASTLE_UNDERINVESTMENT",
    "GENERAL_DEFENCE_COLLAPSE",
    "DEATHTOUCH_NOT_EXPLOITED",
    "LATE_DRAW_AVOIDANCE_FAILURE",
    "PROTOCOL_OR_INVALID_ACTION",
    "OTHER",
)


@dataclass
class QualificationGameRecord:
    policy: str
    opponent: str
    seed: int
    position: int
    winner: int | None
    terminal_turn: int
    terminal_reason: str
    wins: int = 0
    draws: int = 0
    losses: int = 0
    first_enemy_contact_turn: int | None = None
    enemy_general_discovered: bool = False
    turn_enemy_general_discovered: int | None = None
    turn_enemy_general_captured: int | None = None
    land_ratio_terminal: float | None = None
    army_ratio_terminal: float | None = None
    remaining_enemy_land: int | None = None
    candidate_general_cells_terminal: int | None = None
    last_newly_scouted_turn: int | None = None
    dominant_at_terminal: bool = False
    failure_class: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def outcome_from_winner(winner: int | None, *, perspective: int = 0) -> tuple[int, int, int]:
    if winner is None or winner < 0:
        return 0, 1, 0
    if winner == perspective:
        return 1, 0, 0
    return 0, 0, 1


def score_rate(wins: int, draws: int, losses: int) -> float:
    n = wins + draws + losses
    if n <= 0:
        return float("nan")
    return (wins + 0.5 * draws) / n


def summarise_wdl(records: list[QualificationGameRecord]) -> dict[str, Any]:
    wins = sum(r.wins for r in records)
    draws = sum(r.draws for r in records)
    losses = sum(r.losses for r in records)
    n = wins + draws + losses
    dominant = [r for r in records if r.dominant_at_terminal]
    discovered = [r for r in records if r.enemy_general_discovered]
    post_disc_wins = sum(r.wins for r in discovered)
    return {
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / n if n else float("nan"),
        "draw_rate": draws / n if n else float("nan"),
        "loss_rate": losses / n if n else float("nan"),
        "score_rate": score_rate(wins, draws, losses),
        "dominant_position_games": len(dominant),
        "dominant_position_conversion_rate": (
            sum(r.wins for r in dominant) / len(dominant) if dominant else float("nan")
        ),
        "enemy_general_discovery_rate": len(discovered) / n if n else float("nan"),
        "post_discovery_win_rate": post_disc_wins / len(discovered) if discovered else float("nan"),
        "failure_classes": _count_failures(records),
        "note": "Do not use score_rate alone; persistent draws yield 0.5 without wins.",
    }


def _count_failures(records: list[QualificationGameRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        if r.wins:
            continue
        key = r.failure_class or "OTHER"
        out[key] = out.get(key, 0) + 1
    return out


def classify_expander_failure(record: QualificationGameRecord) -> str:
    """Heuristic classifier from telemetry (not outcome alone)."""
    if record.wins:
        return ""
    if record.losses:
        return "GENERAL_DEFENCE_COLLAPSE"
    extras = record.extras
    if record.first_enemy_contact_turn is None:
        return "NO_CONTACT_DRAW"
    if not record.enemy_general_discovered:
        fog = extras.get("unscouted_regions", extras.get("candidate_mask_size", 99))
        if isinstance(fog, (int, float)) and fog > 0:
            return "FOG_NOT_CLEARED"
        return "GENERAL_NEVER_LOCATED"
    if record.enemy_general_discovered and record.turn_enemy_general_captured is None:
        if record.terminal_turn >= 800:
            return "DEATHTOUCH_NOT_EXPLOITED"
        return "GENERAL_LOCATED_NOT_REACHED"
    if record.dominant_at_terminal and record.terminal_turn >= 1050:
        return "LATE_DRAW_AVOIDANCE_FAILURE"
    if extras.get("army_concentration_ratio", 1.0) < 0.15:
        return "ARMY_TOO_DISPERSED"
    if extras.get("largest_stranded_stack", 0) >= 20:
        return "LARGE_STACK_STRANDED"
    if extras.get("pass_count", 0) > extras.get("move_count", 1):
        return "FRONTIER_STALL"
    return "OTHER"


def is_dominant_position(*, my_land: int, opp_land: int, passable: int | None = None) -> bool:
    """Re-export for evaluation callers; packaged policies use phase_controller."""
    from generals_bot.policies.phase_controller import is_dominant_position as _impl

    return _impl(my_land=my_land, opp_land=opp_land, passable=passable)
