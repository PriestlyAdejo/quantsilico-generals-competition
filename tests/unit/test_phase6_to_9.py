"""Phase 6–9 wiring tests."""

from __future__ import annotations

import numpy as np

from generals_bot.game_theory.pfsp import build_smoke_payoff, pfsp_sample
from generals_bot.observation import Observation
from generals_bot.training.autopilot import evaluate_promotion
from generals_bot.training.explain import integrated_gradients_smoke


def test_pfsp_sample() -> None:
    idx = pfsp_sample(np.asarray([0.9, 0.2, 0.5], dtype=np.float64))
    assert 0 <= idx < 3


def test_smoke_payoff_manifest() -> None:
    payload = build_smoke_payoff(["heuristic_v1", "heuristic_v0"])
    assert payload["payoff"]["counts"][0][1] == 3


def test_promotion_keeps_heuristic_champion() -> None:
    report = evaluate_promotion(bridge_decision="PASS", linux_parity=False)
    assert report["decision"] == "NO LEARNED CANDIDATE PROMOTED"
    assert report["champion"] == "heuristic_v1"
    assert report["UPLOAD_READY"] is False


def test_explain_smoke() -> None:
    obs = Observation(
        3,
        3,
        1,
        1,
        5,
        1,
        3,
        ((4, 1, 1), (1, 2, 1), (1, 1, 1)),
        ((1, 1, 0), (0, 0, 0), (0, 0, 2)),
        ((5, 2, 0), (0, 0, 0), (0, 0, 3)),
    )
    report = integrated_gradients_smoke(obs)
    assert report["status"] in {"OK", "SKIPPED_NO_CAPTUM"}
