"""Schema and manifest unit tests."""

from __future__ import annotations

from generals_bot.evaluation.confidence import wilson_interval
from generals_bot.evaluation.payoff_matrix import add_result, empty_payoff
from generals_bot.schemas import SCHEMA_VERSION, MatchResultRecord


def test_match_record_schema_version() -> None:
    rec = MatchResultRecord(seed=1, candidate="a", opponent="b")
    assert rec.to_dict()["schema_version"] == SCHEMA_VERSION


def test_wilson_and_payoff() -> None:
    lo, hi = wilson_interval(7, 10)
    assert 0.0 <= lo <= hi <= 1.0
    payoff = empty_payoff(["a", "b"])
    add_result(payoff, "a", "b", 1.0)
    add_result(payoff, "a", "b", 0.0)
    assert payoff["counts"][0][1] == 2
    assert payoff["matrix"][0][1] == 0.5
