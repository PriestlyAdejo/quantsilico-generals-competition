"""Shadow control replay (CONTROL_REPLAY_* only; no false POLICY_BENEFIT)."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/manifests/emergency_shadow_controls.json"
MPC_STATUS = "MPC_DEFERRED_DEADLINE_PROTECTION"


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    with tmp.open("rb") as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    tmp.replace(path)


def _risk_heuristic(features: dict) -> str:
    """Static risk labels from frozen replay features (not policy benefit)."""
    army = float(features.get("my_army", 0))
    opp = float(features.get("opp_army", 0))
    turn = int(features.get("turn", 0))
    if opp > army * 1.5 and turn > 50:
        return "STATIC_RISK_A"
    if features.get("near_general_threat"):
        return "STATIC_RISK_B"
    return "CONTROL_OFF"


def main() -> int:
    canary = ROOT / "experiments/manifests/emergency_rolling_ckpt_canary.json"
    replay = []
    if canary.exists():
        doc = json.loads(canary.read_text(encoding="utf-8"))
        for entry in doc.get("evaluated", [])[-3:]:
            for r in entry.get("results", [])[:16]:
                feat = {
                    "turn": r.get("turns"),
                    "wdl": r.get("wdl"),
                    "my_army": r.get("focal_army") or 0,
                    "opp_army": r.get("opp_army") or 0,
                    "near_general_threat": False,
                    "checkpoint": entry.get("name"),
                    "update": entry.get("update"),
                }
                mode = _risk_heuristic(feat)
                replay.append(
                    {
                        "features": feat,
                        "control_mode": mode,
                        "label": r.get("wdl"),
                        "note": "replay_only",
                    }
                )

    # Classify replay promise without claiming POLICY_BENEFIT
    interventions = sum(1 for x in replay if x["control_mode"] != "CONTROL_OFF")
    if not replay:
        status = "CONTROL_REPLAY_INCONCLUSIVE"
    elif interventions >= max(1, len(replay) // 4):
        status = "CONTROL_REPLAY_PROMISING"
    else:
        status = "CONTROL_REPLAY_NOT_PROMISING"

    out = {
        "schema_version": 1,
        "kind": "EMERGENCY_SHADOW_CONTROLS",
        "status": status,
        "eligible_modes": ["CONTROL_OFF", "STATIC_RISK_A", "STATIC_RISK_B"],
        "policy_benefit": {
            "status": "NOT_EVALUATED",
            "requires": "CONTROL_ON vs CONTROL_OFF under paired seeds/seats/opponents",
            "note": "Frozen replay must not be interpreted as CONTROL_POLICY_BENEFIT_PASS",
        },
        "mpc": MPC_STATUS,
        "barrier": "research_only_after_hunter_draw_win; non_blocking",
        "top_k_replay": replay[:16],
        "intervention_rate": (interventions / len(replay)) if replay else 0.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(OUT, out)
    print(json.dumps({"status": status, "mpc": MPC_STATUS, "n_replay": len(replay)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
