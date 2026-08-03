"""Policy registry / selector."""

from __future__ import annotations

from typing import Any

from generals_bot.policies.heuristic_v0 import HeuristicV0Policy
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
    raise KeyError(f"unknown policy: {name}")


def list_policies() -> list[str]:
    return ["pass", "legal_random", "heuristic_v0"]
