"""Autopilot promotion gate scaffolding (does not upload or promote without evidence)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_promotion(
    *,
    champion: str = "heuristic_v1",
    challenger: str | None = None,
    bridge_decision: str,
    linux_parity: bool = False,
    package_windows: bool = True,
) -> dict[str, Any]:
    reasons: list[str] = []
    if challenger is None:
        reasons.append("no learned challenger nominated")
    if bridge_decision == "FAIL":
        reasons.append("bridge FAIL blocks learned promotion")
    if not linux_parity:
        reasons.append("Linux parity not verified; Windows package cannot be UPLOAD_READY")
    decision = "NO LEARNED CANDIDATE PROMOTED"
    status = "PACKAGED" if package_windows and champion == "heuristic_v1" else "RESEARCH"
    upload_ready = False
    report = {
        "schema_version": 1,
        "champion": champion,
        "challenger": challenger,
        "decision": decision,
        "champion_status": status,
        "UPLOAD_READY": upload_ready,
        "reasons": reasons,
        "bridge_decision": bridge_decision,
        "linux_parity": linux_parity,
    }
    path = Path("experiments/manifests/promotion_decision.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
