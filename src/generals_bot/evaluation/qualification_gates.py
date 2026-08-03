"""Evaluate Phase 9Q three-level qualification gates from suite summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScreeningResult:
    passed: bool
    reasons: list[str]
    level: str = "SCREENING_SMOKE"


def evaluate_screening_smoke(
    *,
    expander: dict[str, Any],
    hunter: dict[str, Any],
    protocol_faults: int = 0,
    legal_action_rate: float = 1.0,
) -> ScreeningResult:
    """Screening gate: signal for A/B/C, not upload readiness."""
    reasons: list[str] = []
    wins = int(expander.get("wins", expander.get("W", 0)))
    losses = int(expander.get("losses", expander.get("L", 0)))
    discovery = float(expander.get("enemy_general_discovery_rate", expander.get("discovery_rate", 0.0)))
    h_wins = int(hunter.get("wins", hunter.get("W", 0)))
    h_losses = int(hunter.get("losses", hunter.get("L", 0)))
    h_games = int(hunter.get("games", h_wins + int(hunter.get("draws", hunter.get("D", 0))) + h_losses))

    if protocol_faults != 0:
        reasons.append(f"protocol_faults={protocol_faults}")
    if legal_action_rate < 1.0:
        reasons.append(f"legal_action_rate={legal_action_rate}")
    if losses > 1:
        reasons.append(f"expander_losses={losses}>1")
    if wins < 4:
        reasons.append(f"expander_wins={wins}<4")
    if discovery < 0.35:
        reasons.append(f"discovery={discovery:.3f}<0.35")
    if h_games >= 8 and h_wins == 0 and h_losses == 8:
        reasons.append("hunter=0/8")

    return ScreeningResult(passed=not reasons, reasons=reasons)


def evaluate_development_gate(
    *,
    groups: list[dict[str, Any]],
    discovery_rate: float,
    post_discovery_win_rate: float,
    hunter_comparable_to_v1: bool,
    protocol_faults: int = 0,
    legal_action_rate: float = 1.0,
) -> ScreeningResult:
    reasons: list[str] = []
    total_w = total_g = total_l = 0
    for g in groups:
        w = int(g.get("wins", 0))
        d = int(g.get("draws", 0))
        l = int(g.get("losses", 0))
        n = w + d + l
        total_w += w
        total_g += n
        total_l += l
        if n and (w / n) < 0.375:
            reasons.append(f"group_{g.get('name','?')}_win_rate={w/n:.3f}<0.375")
    if total_g == 0:
        reasons.append("no_games")
    else:
        if total_w / total_g < 0.50:
            reasons.append(f"combined_win_rate={total_w/total_g:.3f}<0.50")
        if total_l / total_g > 0.10:
            reasons.append(f"loss_rate={total_l/total_g:.3f}>0.10")
    if discovery_rate < 0.60:
        reasons.append(f"discovery={discovery_rate:.3f}<0.60")
    if post_discovery_win_rate < 0.80:
        reasons.append(f"post_discovery={post_discovery_win_rate:.3f}<0.80")
    if not hunter_comparable_to_v1:
        reasons.append("hunter_below_v1")
    if protocol_faults != 0:
        reasons.append(f"protocol_faults={protocol_faults}")
    if legal_action_rate < 1.0:
        reasons.append(f"legal_action_rate={legal_action_rate}")
    return ScreeningResult(passed=not reasons, reasons=reasons, level="DEVELOPMENT_GATE")
