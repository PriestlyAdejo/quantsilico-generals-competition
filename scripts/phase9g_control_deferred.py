"""Control track artefacts: deferred action-changing control + passive instrumentation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    freeze_path = REPO / "experiments" / "manifests" / "phase9fs_submission_upload_freeze_gate.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8")) if freeze_path.exists() else {}
    freeze_pass = freeze.get("gate_status") == "PASS"
    baseline_type = (freeze.get("frozen") or {}).get("baseline_type") or "HEURISTIC"

    evidence_gate = {
        "schema_version": 1,
        "kind": "ACTION_CHANGING_CONTROL_EVIDENCE_GATE",
        "created_at": now,
        "gate_status": "NOT_SATISFIED",
        "required": [
            "frozen_baseline_identity",
            "failure_reproduced_local_or_public",
            "affected_metric_defined",
            "baseline_failure_rate_measured",
            "controller_targets_that_failure",
            "paired_evaluation_seeds_frozen",
        ],
        "satisfied": {
            "frozen_baseline_identity": freeze_pass,
            "failure_reproduced_local_or_public": False,
            "affected_metric_defined": False,
            "baseline_failure_rate_measured": False,
            "controller_targets_that_failure": False,
            "paired_evaluation_seeds_frozen": False,
        },
        "allowed_action_changing_modes": [],
        "note": (
            "Do not implement STATIC_RISK / PI / BARRIER / MPC that change actions "
            "until this gate PASSes."
        ),
    }

    allowed_passive = True if freeze_pass else False
    # Per plan: passive may begin after upload freeze; without freeze, record deferred
    control = {
        "schema_version": 1,
        "kind": "PHASE9G_CONTROL_DEFERRED",
        "created_at": now,
        "baseline_type": baseline_type,
        "upload_freeze_status": freeze.get("gate_status", "MISSING"),
        "passive_instrumentation": {
            "allowed": allowed_passive,
            "status": "ARMED_AFTER_FREEZE" if freeze_pass else "DEFERRED_UNTIL_UPLOAD_FREEZE",
            "control_mode": "OFF",
            "noninterference": "REQUIRED",
        },
        "action_changing_control": evidence_gate,
        "baseline_allowlist": {
            "HEURISTIC": ["passive", "heuristic_risk_ablations"],
            "BC_HYBRID": ["passive", "neural_score_control_after_gates"],
            "REPAIRED_PPO": ["passive", "neural_score_control_after_gates"],
            "GRAPH_HYBRID": ["passive", "neural_score_control_after_gates"],
        },
        "failure_motivated_order": [
            "OFF",
            "STATIC_RISK",
            "PI",
            "BARRIER",
            "MPC",
        ],
        "combined_only_if_ge_2_favourable": True,
        "eval_tiers": [4, 8, 16],
    }

    (REPO / "experiments" / "manifests" / "phase9fs_action_changing_control_evidence_gate.json").write_text(
        json.dumps(evidence_gate, indent=2) + "\n", encoding="utf-8"
    )
    (REPO / "experiments" / "manifests" / "phase9g_control_deferred.json").write_text(
        json.dumps(control, indent=2) + "\n", encoding="utf-8"
    )
    (REPO / "experiments" / "reports" / "phase9g_control_deferred.md").write_text(
        "\n".join(
            [
                "# Phase 9G control (deferred)",
                "",
                f"Created: {now}",
                "",
                f"- Upload freeze: `{freeze.get('gate_status', 'MISSING')}`",
                f"- Baseline type: `{baseline_type}`",
                f"- Passive: {'allowed after freeze' if freeze_pass else 'deferred until upload freeze'}",
                f"- ACTION_CHANGING_CONTROL_EVIDENCE_GATE: **{evidence_gate['gate_status']}**",
                "",
                "No action-changing controller implemented in this phase.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passive": control["passive_instrumentation"]["status"],
                "evidence_gate": evidence_gate["gate_status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
