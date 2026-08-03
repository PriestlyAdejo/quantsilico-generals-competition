"""Minimal PFSP / payoff utilities for Phase 7."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from generals_bot.evaluation.payoff_matrix import add_result, empty_payoff


def pfsp_sample(win_rates: np.ndarray, *, temperature: float = 1.0) -> int:
    """Prioritised fictitious self-play sampling over opponents."""
    scores = np.clip(1.0 - win_rates, 1e-6, None) ** (1.0 / max(temperature, 1e-6))
    probs = scores / scores.sum()
    return int(np.random.choice(len(probs), p=probs))


def build_smoke_payoff(agents: list[str], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    matrix = empty_payoff(agents)
    for a in agents:
        for b in agents:
            if a == b:
                continue
            # Placeholder smoke scores only — not claimed competition results.
            for _ in range(3):
                add_result(matrix, a, b, float(rng.random()))
    path = Path("experiments/manifests/pfsp_smoke_payoff.json")
    payload = {
        "schema_version": 1,
        "agents": agents,
        "note": "SMOKE synthetic payoff for pipeline wiring; not empirical match results.",
        "payoff": matrix,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
