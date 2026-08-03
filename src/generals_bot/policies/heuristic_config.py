"""Heuristic configuration knobs (private thresholds)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HeuristicConfig:
    name: str = "heuristic_v1"
    expand_weight: float = 1.0
    attack_weight: float = 1.0
    defend_weight: float = 1.0
    scout_weight: float = 1.0
    collect_weight: float = 1.0
    castle_weight: float = 1.0
    deathtouch_weight: float = 1.0
    aggression: float = 1.0
    prefer_castles: bool = False
    prefer_deathtouch: bool = False


V1 = HeuristicConfig()
AGGRESSIVE = HeuristicConfig(
    name="heuristic_aggressive",
    expand_weight=1.2,
    attack_weight=1.6,
    defend_weight=0.8,
    aggression=1.5,
)
DEFENSIVE = HeuristicConfig(
    name="heuristic_defensive",
    expand_weight=0.8,
    attack_weight=0.7,
    defend_weight=1.8,
    scout_weight=0.7,
    aggression=0.6,
)
CASTLE = HeuristicConfig(
    name="heuristic_castle",
    castle_weight=2.0,
    expand_weight=1.1,
    prefer_castles=True,
)
DEATHTOUCH = HeuristicConfig(
    name="heuristic_deathtouch",
    deathtouch_weight=2.5,
    attack_weight=1.4,
    prefer_deathtouch=True,
)
