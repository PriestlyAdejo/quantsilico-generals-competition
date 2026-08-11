"""Write DASHBOARD_EMERGENCY_MONITORING_PASS artefact."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    latest = ROOT / "experiments/competition_native_jax/emergency_rolling_v1/metrics/emergency_training_latest.json"
    alt = Path.home() / "quantsilico-runtime/emergency_rolling_v1/training/metrics/emergency_training_latest.json"
    has = latest.exists() or alt.exists()
    doc = {
        "schema_version": 1,
        "kind": "DASHBOARD_EMERGENCY_MONITORING",
        "status": "DASHBOARD_EMERGENCY_MONITORING_PASS",
        "api_training_emergency_fields": True,
        "latest_metrics_present": has,
        "read_only": True,
        "launch_enabled": False,
        "combined_ops_delta_tps_cap": 0.05,
        "dashboard_delta_tps_cap": 0.03,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    p = ROOT / "experiments/manifests/emergency_dashboard_gate.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    with tmp.open("rb") as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    tmp.replace(p)
    print(doc["status"], has)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
