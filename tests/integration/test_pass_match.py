"""Integration: pass-bot completes an official-style match without faults."""

from __future__ import annotations

from pathlib import Path

from generals_bot.evaluation.match import run_python_agent_match

REPO = Path(__file__).resolve().parents[2]
PASS_MAIN = REPO / "baselines" / "pass_bot" / "main.py"


def test_pass_vs_pass_competition_short() -> None:
    result = run_python_agent_match(
        PASS_MAIN,
        PASS_MAIN,
        seed=0,
        mode="competition",
        max_turns=30,
    )
    assert result.faults0 == 0
    assert result.faults1 == 0
    assert result.turns == 30
    assert result.winner == -1
