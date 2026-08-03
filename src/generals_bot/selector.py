"""Policy registry / selector."""

from __future__ import annotations

from typing import Any

from generals_bot.policies.heuristic_config import (
    AGGRESSIVE,
    CASTLE,
    DEATHTOUCH,
    DEFENSIVE,
)
from generals_bot.policies.heuristic_v0 import HeuristicV0Policy
from generals_bot.policies.heuristic_v1 import HeuristicV1Policy
from generals_bot.policies.pass_policy import PassPolicy
from generals_bot.policies.random_policy import RandomPolicy


def create_policy(name: str, **kwargs: Any) -> Any:
    key = name.strip().lower().replace("-", "_")
    if key in {"pass", "pass_bot", "pass_policy"}:
        return PassPolicy()
    if key in {"legal_random", "random", "random_policy"}:
        return RandomPolicy(seed=int(kwargs.get("seed", 0)))
    if key in {"heuristic_v0", "heuristic0", "hv0"}:
        return HeuristicV0Policy()
    if key in {"heuristic_v1", "heuristic1", "hv1"}:
        return HeuristicV1Policy()
    if key in {"heuristic_aggressive", "aggressive"}:
        return HeuristicV1Policy(config=AGGRESSIVE)
    if key in {"heuristic_defensive", "defensive"}:
        return HeuristicV1Policy(config=DEFENSIVE)
    if key in {"heuristic_castle", "castle"}:
        return HeuristicV1Policy(config=CASTLE)
    if key in {"heuristic_deathtouch", "deathtouch"}:
        return HeuristicV1Policy(config=DEATHTOUCH)
    raise KeyError(f"unknown policy: {name}")


def list_policies() -> list[str]:
    return [
        "pass",
        "legal_random",
        "heuristic_v0",
        "heuristic_v1",
        "heuristic_aggressive",
        "heuristic_defensive",
        "heuristic_castle",
        "heuristic_deathtouch",
    ]
