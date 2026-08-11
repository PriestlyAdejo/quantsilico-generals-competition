"""Read-only emergency sprint dashboard snapshot (≤3% ops budget intent)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / "quantsilico-runtime" / "emergency_rolling_v1"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    prog = _load(ROOT / "experiments/manifests/emergency_rolling_programme_state.json") or {}
    deadlines = _load(ROOT / "experiments/manifests/emergency_deadlines.json") or {}
    gate = _load(ROOT / "experiments/manifests/emergency_deadline_resolution_gate.json") or {}
    distill = _load(ROOT / "experiments/manifests/emergency_distill_plumbing.json") or {}
    canary = _load(ROOT / "experiments/manifests/emergency_rolling_ckpt_canary.json") or {}
    teacher = _load(ROOT / "experiments/manifests/emergency_teacher_selection.json") or {}
    learned = _load(ROOT / "experiments/manifests/emergency_learned_package_v1.json") or {}
    shadow = _load(ROOT / "experiments/manifests/emergency_shadow_controls.json") or {}
    signal = _load(ROOT / "experiments/manifests/emergency_ppo_learning_signal.json") or {}
    gpu = _load(ROOT / "experiments/manifests/emergency_gpu_owner.json") or {}
    contract = _load(ROOT / "experiments/manifests/emergency_distill_sequence_contract.json") or {}

    dash = {
        "schema_version": 1,
        "kind": "EMERGENCY_READONLY_DASHBOARD",
        "programme": prog.get("status") or prog.get("programme"),
        "ppo": {
            "pid": prog.get("ppo_pid"),
            "updates": prog.get("updates"),
            "transitions": prog.get("transitions"),
            "session_tps": prog.get("session_tps") or prog.get("tps"),
            "learning_signal": signal.get("latest_class"),
        },
        "deadline_gate": gate.get("outcome") or gate.get("status"),
        "sprint_end_at": deadlines.get("sprint_end_at"),
        "experiments_stop_at": deadlines.get("experiments_stop_at"),
        "gpu_owner": gpu.get("gpu_owner"),
        "distill": {
            "route": distill.get("status"),
            "sequence_contract": bool(contract),
            "teacher": teacher.get("selected"),
        },
        "canary": {
            "n_evaluated": len(canary.get("evaluated") or []),
            "pending": canary.get("pending"),
            "latest": (canary.get("evaluated") or [{}])[-1].get("name") if canary.get("evaluated") else None,
            "latest_WDL": {
                "W": (canary.get("evaluated") or [{}])[-1].get("W"),
                "D": (canary.get("evaluated") or [{}])[-1].get("D"),
                "L": (canary.get("evaluated") or [{}])[-1].get("L"),
            }
            if canary.get("evaluated")
            else None,
        },
        "learned_package": {
            "exists": bool(learned),
            "path": learned.get("package_path"),
            "sha256": learned.get("sha256"),
            "technical": learned.get("technical"),
            "class": learned.get("status_class"),
        },
        "shadow_controls": {
            "status": shadow.get("status"),
            "mpc": shadow.get("mpc"),
            "policy_benefit": (shadow.get("policy_benefit") or {}).get("status"),
        },
        "upload_pointer": "submission/EMERGENCY_UPLOAD_THIS.md",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    out = ROOT / "experiments/manifests/emergency_dashboard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dash, indent=2) + "\n", encoding="utf-8")
    md = ROOT / "experiments/reports/emergency_dashboard.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        "# Emergency dashboard (read-only)\n\n"
        f"- programme: `{dash['programme']}`\n"
        f"- PPO updates/tps: `{dash['ppo'].get('updates')}` / `{dash['ppo'].get('session_tps')}`\n"
        f"- distill route: `{dash['distill'].get('route')}`\n"
        f"- canary latest: `{dash['canary'].get('latest')}` {dash['canary'].get('latest_WDL')}\n"
        f"- learned: `{dash['learned_package'].get('class')}` tech={dash['learned_package'].get('technical')}\n"
        f"- shadow: `{dash['shadow_controls'].get('status')}` mpc={dash['shadow_controls'].get('mpc')}\n"
        f"- hard end: `{dash['sprint_end_at']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(dash, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
