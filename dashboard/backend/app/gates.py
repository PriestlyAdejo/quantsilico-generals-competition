"""Current vs historical research gate aggregation for the console."""

from __future__ import annotations

from typing import Any

from dashboard.backend.app.readers.evidence import manifest


def current_gate_status() -> dict[str, str]:
    """Live project gates from current manifests — never upload-time observation."""
    readiness = manifest("learning_readiness_gate.json") or {}
    preppo = manifest("phase_9q_pre_ppo_submission_gate.json") or {}
    portal_obs = manifest("official_portal_observation_heuristic_v2_preppo_2026-08-04.json") or {}
    promo = manifest("learned_promotion_precheck.json") or {}
    hist = portal_obs.get("gate_status_at_observation") or {}

    learning = str(readiness.get("decision") or "UNKNOWN")

    # PRE_PPO: prefer explicit decision; else conversion_micro.passed from gate manifest.
    pre = preppo.get("decision")
    if not pre:
        conv = preppo.get("conversion_micro") or {}
        if isinstance(conv, dict) and "passed" in conv:
            pre = "PASS" if conv.get("passed") else "FAIL"
        else:
            pre = hist.get("PRE_PPO_SUBMISSION_GATE") or "UNKNOWN"

    portal = hist.get("PORTAL_SUBMISSION_GATE") or "UNKNOWN"
    if portal_obs.get("portal_verdict") == "QUALIFIED" or portal == "PASS":
        portal = "PASS"

    return {
        "learning_readiness": learning,
        "heuristic_development": "FAIL",
        "pre_ppo_submission": str(pre),
        "portal_submission": str(portal),
        "learned_promotion": str(promo.get("decision") or "NONE"),
    }


def historical_gate_observations() -> list[dict[str, Any]]:
    obs = manifest("official_portal_observation_heuristic_v2_preppo_2026-08-04.json") or {}
    hist = obs.get("gate_status_at_observation")
    if not hist:
        return []
    return [
        {
            "observed_at": obs.get("observed_at"),
            "source": "UPLOAD_RECORD",
            "learning_readiness": hist.get("LEARNING_READINESS_GATE"),
            "heuristic_development": hist.get("HEURISTIC_DEVELOPMENT_GATE"),
            "pre_ppo_submission": hist.get("PRE_PPO_SUBMISSION_GATE"),
            "portal_submission": hist.get("PORTAL_SUBMISSION_GATE"),
            "learned_promotion": hist.get("LEARNED_PROMOTION_GATE"),
            "raw_keys": hist,
            "note": "Immutable upload-time observation; must not override current gate manifests.",
        }
    ]


def gate_status_dto() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "GATE_STATUS_SPLIT",
        "current": current_gate_status(),
        "historical_observations": historical_gate_observations(),
    }


def legacy_gate_board_from_current(current: dict[str, str] | None = None) -> dict[str, str]:
    """Named FULL_*_GATE keys for UI chips that still expect legacy names."""
    c = current or current_gate_status()
    return {
        "HEURISTIC_DEVELOPMENT_GATE": c["heuristic_development"],
        "PRE_PPO_SUBMISSION_GATE": c["pre_ppo_submission"],
        "PORTAL_SUBMISSION_GATE": c["portal_submission"],
        "LEARNING_READINESS_GATE": c["learning_readiness"],
        "LEARNED_PROMOTION_GATE": c["learned_promotion"],
    }
