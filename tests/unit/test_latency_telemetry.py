"""Unit tests for competition-size latency classification and telemetry schema."""

from __future__ import annotations

from generals_bot.training.latency_gate import classify
from generals_bot.training.telemetry_schema import annotate_history, PPO_UPDATE_FIELDS


def _row(p99: float, maximum: float) -> dict:
    return {"stats_ms": {"p99": p99, "maximum": maximum, "p50": 1.0, "p95": 1.0, "first_ms": 1.0, "n": 1}}


def test_latency_classify_pass_partial_fail() -> None:
    assert classify([_row(100.0, 110.0)]) == "PASS"
    assert classify([_row(130.0, 140.0)]) == "PARTIAL"
    assert classify([_row(150.0, 160.0)]) == "FAIL"
    assert classify([_row(100.0, 155.0)]) == "FAIL"


def test_telemetry_annotate_empty_is_not_recorded() -> None:
    out = annotate_history([], producer="unit")
    assert out["note"] == "NOT RECORDED"
    assert out["points"] == []
    assert set(out["missing"]) == set(PPO_UPDATE_FIELDS)


def test_telemetry_annotate_partial_fields() -> None:
    out = annotate_history([{"update": 0, "loss": 1.0}], producer="unit")
    assert "loss" not in out["missing"]
    assert "approx_kl" in out["missing"]
    assert out["points"][0]["loss"] == 1.0
