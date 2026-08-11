"""Patch emergency programme state with current artefact pointers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = ROOT / "experiments/manifests/emergency_rolling_programme_state.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["status"] = "PHASE_C_TRAINING_RUNNING"
    doc["package_status"] = "EMERGENCY_BASELINE_FALLBACK_PACKAGE_EXISTS"
    doc["baseline_fallback"] = "experiments/manifests/emergency_baseline_fallback.json"
    doc["distill"] = "experiments/manifests/emergency_distill_plumbing.json"
    doc["triage"] = "experiments/manifests/emergency_checkpoint_triage.json"
    doc["controls"] = "experiments/manifests/emergency_controls_decision.json"
    doc["selection_gate"] = "experiments/manifests/final_emergency_package_selection_gate.json"
    doc["emergency_upload_this"] = "submission/EMERGENCY_UPLOAD_THIS.md"
    doc["watchdog"] = "RUNNING"
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    with tmp.open("rb") as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    tmp.replace(p)
    print(doc["status"], doc["package_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
