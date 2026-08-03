"""Tests for live-like evaluator fault classification."""

from __future__ import annotations

from pathlib import Path

from generals_bot.evaluation.runner import run_live_like_match

REPO = Path(__file__).resolve().parents[2]
PASS = REPO / "baselines" / "pass_bot" / "main.py"
HV1 = REPO / "baselines" / "heuristic_v1" / "main.py"
FIX = REPO / "baselines" / "fixtures"


def test_heuristic_v1_zero_protocol_faults() -> None:
    result = run_live_like_match(
        HV1,
        PASS,
        seed=0,
        max_turns=20,
        enforce_deadlines=True,
        candidate="heuristic_v1",
        opponent="pass",
    )
    assert result.protocol_fault_count0 == 0
    assert result.protocol_fault_count1 == 0
    assert result.faults0 == 0
    assert result.faults1 == 0


def test_malformed_classified_as_protocol_fault() -> None:
    result = run_live_like_match(
        FIX / "malformed" / "main.py",
        PASS,
        seed=1,
        max_turns=5,
        enforce_deadlines=True,
        candidate="malformed",
        opponent="pass",
    )
    assert result.protocol_fault_count0 >= 1
    assert result.illegal_action_count0 == 0


def test_illegal_action_not_protocol_fault() -> None:
    result = run_live_like_match(
        FIX / "illegal_action" / "main.py",
        PASS,
        seed=2,
        max_turns=5,
        enforce_deadlines=True,
        candidate="illegal_action",
        opponent="pass",
    )
    # Well-formed illegal moves are silent passes — not protocol faults.
    assert result.protocol_fault_count0 == 0
    assert result.turns == 5


def test_crash_fixture() -> None:
    result = run_live_like_match(
        FIX / "crash" / "main.py",
        PASS,
        seed=3,
        max_turns=3,
        enforce_deadlines=True,
        candidate="crash",
        opponent="pass",
    )
    assert result.crash0 or result.protocol_fault_count0 >= 1


def test_late_reply_fixture() -> None:
    result = run_live_like_match(
        FIX / "late_reply" / "main.py",
        PASS,
        seed=4,
        max_turns=3,
        enforce_deadlines=True,
        candidate="late_reply",
        opponent="pass",
    )
    # First action has 10s grace; subsequent 0.35s sleep exceeds 150ms.
    assert result.protocol_fault_count0 >= 1


def test_paired_seed_reproducible() -> None:
    a = run_live_like_match(PASS, PASS, seed=7, max_turns=15, candidate="pass", opponent="pass")
    b = run_live_like_match(PASS, PASS, seed=7, max_turns=15, candidate="pass", opponent="pass")
    assert a.turns == b.turns
    assert a.winner == b.winner
    assert a.faults0 == b.faults0 == 0
