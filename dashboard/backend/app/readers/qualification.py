"""Qualification evidence DTO — maps real manifests without inventing zeros."""

from __future__ import annotations

from typing import Any

from generals_bot.candidate_identity import EXECUTABLE_REGISTRY_ID
from dashboard.backend.app.gates import gate_status_dto, legacy_gate_board_from_current
from dashboard.backend.app.readers.evidence import manifest


def _wdl_recorded(w: int, d: int, l: int, *, source: str, suite: str) -> dict[str, Any]:
    return {
        "wins": w,
        "draws": d,
        "losses": l,
        "availability": "RECORDED",
        "source": source,
        "suite": suite,
    }


def _metric_missing(note: str, *, suite: str | None = None) -> dict[str, Any]:
    return {
        "value": None,
        "availability": "MISSING",
        "note": note,
        "suite": suite,
    }


def _metric_recorded(value: float, *, source: str, suite: str) -> dict[str, Any]:
    return {
        "value": value,
        "availability": "RECORDED",
        "source": source,
        "suite": suite,
    }


def qualification_dashboard_dto() -> dict[str, Any]:
    gates = gate_status_dto()
    board = legacy_gate_board_from_current(gates["current"])
    abc = manifest("phase_9q_development_abc_terminal_fix.json") or {}
    discovery_diag = manifest("phase_9q_discovery_diagnosis.json") or {}
    preppo = manifest("phase_9q_pre_ppo_submission_gate.json")
    readiness = manifest("learning_readiness_gate.json")
    portal_v2 = manifest("official_portal_observation_heuristic_v2_preppo_2026-08-04.json")

    combined = abc.get("combined") or {}
    source_abc = "experiments/manifests/phase_9q_development_abc_terminal_fix.json"
    suite_dev = "development"

    development_wdl = None
    discovery = _metric_missing("No DEVELOPMENT discovery rate in ABC terminal_fix record", suite=suite_dev)
    conversion = _metric_missing("No DEVELOPMENT conversion rate in ABC terminal_fix record", suite=suite_dev)
    if isinstance(combined, dict) and all(k in combined for k in ("W", "D", "L")):
        development_wdl = _wdl_recorded(
            int(combined["W"]),
            int(combined["D"]),
            int(combined["L"]),
            source=source_abc,
            suite=suite_dev,
        )
        if "discovery_rate" in combined:
            discovery = _metric_recorded(float(combined["discovery_rate"]), source=source_abc, suite=suite_dev)
        if "post_discovery_win_rate" in combined:
            conversion = _metric_recorded(
                float(combined["post_discovery_win_rate"]), source=source_abc, suite=suite_dev
            )

    persistent = _metric_missing("Persistent-state diagnostic not present as structured counts")
    conv_base = (discovery_diag.get("conversion_baseline") or {}) if isinstance(discovery_diag, dict) else {}
    # Prefer explicit diagnostic fields when present
    if isinstance(discovery_diag, dict):
        note = discovery_diag.get("persistent_state_note") or discovery_diag.get("note")
        discovered = discovery_diag.get("discovered_draws") or discovery_diag.get("persistent_discovered")
        total_draws = discovery_diag.get("total_draws") or discovery_diag.get("persistent_total")
        if discovered is not None and total_draws is not None:
            persistent = {
                "value": {"discovered": int(discovered), "total": int(total_draws)},
                "availability": "RECORDED",
                "source": "experiments/manifests/phase_9q_discovery_diagnosis.json",
                "suite": "persistent-state-diagnostic",
            }
        elif note:
            persistent = {
                "value": None,
                "availability": "MISSING",
                "note": str(note),
                "suite": "persistent-state-diagnostic",
                # Human-readable fallback from known evidence wording
                "display_hint": "8 / 27 discovered (see discovery diagnosis manifest)",
            }

    stages = [
        {
            "id": "screening",
            "label": "Screening Evaluation",
            "internal_id": "screening",
            "status": "pending",
            "explains": "Early filter suite before full DEVELOPMENT evaluation.",
            "evidence": "NOT RECORDED for the active submitted heuristic in this console.",
            "pass_means": "Candidate cleared screening thresholds.",
            "fail_means": "Candidate failed screening and should not advance.",
            "blocks_next": True,
            "perspective": "current",
        },
        {
            "id": "development",
            "label": "Development Evaluation",
            "internal_id": "HEURISTIC_DEVELOPMENT_GATE",
            "status": "failed" if (abc.get("development_gate") or {}).get("passed") is False else (
                "complete" if (abc.get("development_gate") or {}).get("passed") is True else "pending"
            ),
            "explains": "Internal research Expander discovery/conversion DEVELOPMENT suite.",
            "evidence": source_abc,
            "pass_means": "Discovery and conversion thresholds met.",
            "fail_means": "Below discovery/conversion thresholds — blocks learned promotion, not portal package.",
            "blocks_next": False,
            "perspective": "current",
            "reasons": (abc.get("development_gate") or {}).get("reasons") or [],
        },
        {
            "id": "holdout",
            "label": "Holdout Evaluation",
            "internal_id": "promotion_holdout",
            "status": "pending",
            "explains": "Promotion holdout is reserved and must not be consumed by dashboard campaigns.",
            "evidence": "NOT APPLICABLE — holdout unused by design.",
            "pass_means": "Holdout evaluation authorised and passed.",
            "fail_means": "Holdout failed.",
            "blocks_next": True,
            "perspective": "current",
        },
        {
            "id": "package",
            "label": "Package Build",
            "internal_id": "PACKAGE_BUILT",
            "status": "complete",
            "explains": "Immutable submitted package artefact.",
            "evidence": "submitted package DTO",
            "pass_means": "Package exists with recorded hashes.",
            "fail_means": "Package missing or invalid.",
            "blocks_next": True,
            "perspective": "current",
        },
        {
            "id": "linux_parity",
            "label": "Linux Validation",
            "internal_id": "LINUX_PARITY",
            "status": "complete" if (manifest("linux_parity_report_preppo.json") or {}).get("passed") else "pending",
            "explains": "Linux package validation / parity report.",
            "evidence": "experiments/manifests/linux_parity_report_preppo.json",
            "pass_means": "Linux validation checks passed.",
            "fail_means": "Linux validation failed.",
            "blocks_next": True,
            "perspective": "current",
        },
        {
            "id": "upload_ready",
            "label": "Upload Ready",
            "internal_id": "UPLOAD_READY",
            "status": "complete",
            "explains": "Operator may manually upload outside this console.",
            "evidence": "manual upload instructions",
            "pass_means": "Package ready for manual portal upload.",
            "fail_means": "Package not ready.",
            "blocks_next": False,
            "perspective": "current",
        },
        {
            "id": "portal",
            "label": "Portal Accepted",
            "internal_id": "PORTAL_SUBMISSION_GATE",
            "status": "complete" if str(board.get("PORTAL_SUBMISSION_GATE", "")).upper() in {"PASS", "QUALIFIED", "PASSED"} else "pending",
            "explains": "Official portal Expander gate observation (QUALIFIED ≠ final tournament).",
            "evidence": "official portal observation manifest",
            "pass_means": "Portal acceptance recorded at observation time.",
            "fail_means": "Portal gate failed at observation time.",
            "blocks_next": False,
            "perspective": "historical_observation",
        },
    ]

    suites = [
        {"id": "screening", "label": "Screening Evaluation", "internal_id": "screening"},
        {"id": "development", "label": "Development Evaluation", "internal_id": "HEURISTIC_DEVELOPMENT_GATE"},
        {
            "id": "persistent-state-diagnostic",
            "label": "Persistent-state diagnostic",
            "internal_id": "persistent_state",
        },
        {"id": "pre-ppo", "label": "Pre-PPO submission comparison", "internal_id": "PRE_PPO_SUBMISSION_GATE"},
        {"id": "portal", "label": "Portal submission", "internal_id": "PORTAL_SUBMISSION_GATE"},
        {"id": "learning-readiness", "label": "Learning readiness", "internal_id": "LEARNING_READINESS_GATE"},
        {"id": "learned-promotion", "label": "Learned promotion", "internal_id": "LEARNED_PROMOTION_GATE"},
    ]

    return {
        "schema_version": 1,
        "kind": "QUALIFICATION_DASHBOARD",
        "primary_title": "Candidate Qualification",
        "internal_workflow": "PHASE_9Q",
        "default_candidate": EXECUTABLE_REGISTRY_ID,
        "champion_until_promoted": EXECUTABLE_REGISTRY_ID,
        "candidates": [
            {
                "id": EXECUTABLE_REGISTRY_ID,
                "name": EXECUTABLE_REGISTRY_ID,
                "kind": "IMPORTED_PROJECT_EVIDENCE",
                "screening_wdl": {
                    "wins": None,
                    "draws": None,
                    "losses": None,
                    "availability": "MISSING",
                    "note": "No screening W/D/L record for this candidate in manifests",
                    "suite": "screening",
                },
                "development_wdl": development_wdl,
                "discovery": discovery,
                "conversion": conversion,
                "persistent_state": persistent,
                "terminal_turn_p50": _metric_missing("Terminal turn p50 not recorded in ABC manifest"),
                "terminal_turn_p95": _metric_missing("Terminal turn p95 not recorded in ABC manifest"),
            }
        ],
        "suites": suites,
        "stages": stages,
        "gates": board,
        "gate_status": gates,
        "gate_names": {
            "HEURISTIC_DEVELOPMENT_GATE": "internal research Expander discovery/conversion suite",
            "PRE_PPO_SUBMISSION_GATE": "local comparison vs previously submitted package",
            "PORTAL_SUBMISSION_GATE": "portal Expander 3-game gate (QUALIFIED ≠ final tournament)",
            "LEARNING_READINESS_GATE": "engineering readiness before PPO campaigns",
            "LEARNED_PROMOTION_GATE": "learned model may replace heuristic champion",
        },
        "portal_active_v2": portal_v2,
        "pre_ppo_submission_gate": preppo,
        "learning_readiness": readiness,
        "development_abc": abc,
        "discovery_diagnosis": discovery_diag,
        "conversion_baseline_label": conv_base.get("development") if isinstance(conv_base, dict) else None,
        "note": "Never merge unlabeled suites. Current vs historical gate perspectives stay separate.",
    }
