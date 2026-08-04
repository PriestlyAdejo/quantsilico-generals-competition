"""Tests for DEVELOPMENT audit, INITIAL readiness, and durable telemetry."""

from __future__ import annotations

import json
from pathlib import Path

from generals_bot.training.campaign_telemetry import (
    campaign_path,
    load_campaign,
    new_campaign_record,
    persist_campaign,
)
from generals_bot.training.development_audit import audit_development_arms
from generals_bot.training.initial_readiness import evaluate_initial_readiness

REPO = Path(__file__).resolve().parents[1]


def test_development_audit_ranks_eight_arms() -> None:
    source = REPO / "experiments" / "manifests" / "bounded_development_ppo.json"
    report = audit_development_arms(source)
    assert report["arm_count"] == 8
    assert report["best_cnn"] is not None
    assert report["best_cnn"]["engineering_ok"] is True
    assert report["best_cnn"]["architecture"] == "recurrent_cnn_v2"
    assert len(report["ranked_arms"]) == 8


def test_initial_readiness_ready_with_cnn() -> None:
    source = REPO / "experiments" / "manifests" / "bounded_development_ppo.json"
    audit = audit_development_arms(source)
    gate = evaluate_initial_readiness(audit)
    assert gate["kind"] == "INITIAL_READINESS_GATE"
    assert gate["decision"] == "READY"
    slots = {c["slot"] for c in gate["selected_candidates"]}
    assert "cnn" in slots
    assert len(gate["selected_candidates"]) <= 2


def test_durable_campaign_telemetry_roundtrip(tmp_path: Path, monkeypatch) -> None:
    import generals_bot.training.campaign_telemetry as ct

    monkeypatch.setattr(ct, "TELEMETRY_DIR", tmp_path)
    rec = new_campaign_record(
        campaign_id="test_campaign_1",
        stage="INITIAL",
        config_hash="abc",
        architecture="recurrent_cnn_v2",
    )
    path = persist_campaign(rec)
    assert path.is_file()
    loaded = load_campaign("test_campaign_1")
    assert loaded is not None
    assert loaded["campaign_id"] == "test_campaign_1"
    assert loaded["kind"] == "DURABLE_CAMPAIGN_TELEMETRY"
    # Incremental update
    loaded["env_steps"] = 128
    persist_campaign(loaded)
    again = load_campaign("test_campaign_1")
    assert again is not None
    assert again["env_steps"] == 128
    assert campaign_path("test_campaign_1").is_file()


def test_overnight_readiness_blocked_on_draw_only_initial() -> None:
    from generals_bot.training.overnight_readiness import evaluate_overnight_readiness

    initial = json.loads(
        (REPO / "experiments" / "manifests" / "adaptive_initial_campaign.json").read_text(encoding="utf-8")
    )
    gate = evaluate_overnight_readiness(initial)
    assert gate["kind"] == "OVERNIGHT_READINESS_GATE"
    assert gate["decision"] == "BLOCKED"
    assert gate["holdout_unused"] is True
    assert any("wins" in b for b in gate["blockers"])
