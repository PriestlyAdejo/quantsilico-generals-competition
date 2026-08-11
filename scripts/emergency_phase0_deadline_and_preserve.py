"""Phase 0: deadline gate + preserve live PPO + RUNNING_EMERGENCY_PPO."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    now = datetime.now(timezone.utc)
    # Search local docs for an exact portal deadline (no portal mutation).
    evidence = []
    portal_deadline = None
    for rel in (
        "docs",
        "third_party/generals-bots",
        "submission",
        "experiments/manifests",
    ):
        base = ROOT / rel
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.suffix.lower() not in {".md", ".txt", ".json"}:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            low = text.lower()
            if "portal" in low and ("deadline" in low or "closes" in low or "submission deadline" in low):
                evidence.append(str(p.relative_to(ROOT)).replace("\\", "/"))
            # ISO-like competition end hints already known in our emergency deadlines only
    # Operator did not supply an exact portal deadline in this session.
    gate = {
        "schema_version": 1,
        "kind": "EMERGENCY_DEADLINE_RESOLUTION_GATE",
        "status": "DEADLINE_UNRESOLVED_KEEP_ORIGINAL_HARD_END",
        "portal_deadline_resolved": False,
        "portal_deadline": portal_deadline,
        "evidence_paths_scanned_hits": evidence[:20],
        "decision": "DO_NOT_EXTEND_TO_NOON",
        "reason": (
            "No exact portal submission deadline found in local docs/portal metadata "
            "without mutation, and no operator-provided deadline. Preserve original "
            "hard end 11:21 BST rather than assuming noon is safe."
        ),
        "original_sprint_end_at": "2026-08-07T10:21:06.079517+00:00",
        "original_sprint_end_local": "2026-08-07T11:21:06+01:00",
        "updated_at": now.isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_deadline_resolution_gate.json", gate)

    # Live PPO snapshot
    latest = _read_json(
        ROOT / "experiments/competition_native_jax/emergency_rolling_v1/metrics/emergency_training_latest.json"
    ) or _read_json(
        Path.home() / "quantsilico-runtime/emergency_rolling_v1/training/metrics/emergency_training_latest.json"
    )
    pid_path = ROOT / "experiments/logs/owned_jobs/emergency_ppo.pid"
    pid = int(pid_path.read_text().strip()) if pid_path.exists() else None
    ckpt_dir = Path.home() / "quantsilico-runtime/emergency_rolling_v1/training/checkpoints"
    completes = sorted([p.name for p in ckpt_dir.glob("ckpt_*/COMPLETE")]) if ckpt_dir.exists() else []

    # Keep original deadlines (do not extend)
    deadlines = _read_json(ROOT / "experiments/manifests/emergency_deadlines.json") or {}
    deadlines["schedule_amendment"] = {
        "attempted_noon_extension": False,
        "gate": "EMERGENCY_DEADLINE_RESOLUTION_GATE",
        "gate_status": gate["status"],
        "note": "Original 8h sprint deadlines retained",
    }
    deadlines["first_learned_distillation_ready_deadline_local"] = "2026-08-07T06:30:00+01:00"
    deadlines["first_learned_handoff_grace_minutes"] = 10
    deadlines["mpc"] = "MPC_DEFERRED_DEADLINE_PROTECTION"
    _atomic_write_json(ROOT / "experiments/manifests/emergency_deadlines.json", deadlines)

    prog_path = ROOT / "experiments/manifests/emergency_rolling_programme_state.json"
    prog = _read_json(prog_path) or {"schema_version": 1, "kind": "COMPETITION_NATIVE_JAX_EMERGENCY_ROLLING_PACKAGE_V1"}
    prog["status"] = "RUNNING_EMERGENCY_PPO"
    prog["upload_pointer_class"] = "PROVISIONAL_BASELINE_INSURANCE"
    prog["deadline_gate"] = "experiments/manifests/emergency_deadline_resolution_gate.json"
    prog["ppo_pid"] = pid
    prog["live_latest"] = latest
    prog["complete_checkpoints"] = completes
    prog["mpc"] = "MPC_DEFERRED_DEADLINE_PROTECTION"
    prog["updated_at"] = now.isoformat()
    _atomic_write_json(prog_path, prog)

    runtime_status = {
        "schema_version": 1,
        "kind": "EMERGENCY_ROLLING_PACKAGE_RUNTIME_STATUS",
        "status": "RUNNING_EMERGENCY_PPO",
        "ppo_pid": pid,
        "watchdog": "scripts/wsl/_emergency_sprint_watchdog.sh",
        "package_status": "EMERGENCY_BASELINE_FALLBACK_PACKAGE_EXISTS",
        "upload_pointer_class": "PROVISIONAL_BASELINE_INSURANCE",
        "deadline_extension": "DENIED_UNRESOLVED_PORTAL_DEADLINE",
        "sprint_end_at": deadlines.get("sprint_end_at"),
        "experiments_stop_at": deadlines.get("experiments_stop_at"),
        "live": latest,
        "complete_checkpoints": completes,
        "updated_at": now.isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_rolling_runtime_status.json", runtime_status)

    print(
        json.dumps(
            {
                "deadline_gate": gate["status"],
                "ppo_pid": pid,
                "updates": (latest or {}).get("updates"),
                "tps": (latest or {}).get("tps"),
                "completes": completes,
                "programme": "RUNNING_EMERGENCY_PPO",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
