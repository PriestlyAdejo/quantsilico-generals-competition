"""OVERNIGHT_READINESS_GATE after adaptive INITIAL."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
DEFAULT_INITIAL = REPO / "experiments" / "manifests" / "adaptive_initial_campaign.json"
DEFAULT_OUT = REPO / "experiments" / "manifests" / "overnight_readiness_gate.json"


def evaluate_overnight_readiness(initial: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []

    if initial.get("status") != "COMPLETED":
        blockers.append(f"adaptive INITIAL status={initial.get('status')}")

    best = initial.get("best") or {}
    if not best:
        blockers.append("No INITIAL best candidate recorded")
    else:
        val = best.get("best_validation") or {}
        wins = int(val.get("wins") or 0)
        draws = int(val.get("draws") or 0)
        losses = int(val.get("losses") or 0)
        faults = int(val.get("protocol_faults") or 0)
        score = val.get("score_rate")
        if wins <= 0:
            blockers.append(f"best validation wins={wins} (require >0 before overnight)")
        if faults > 0:
            blockers.append(f"best validation protocol_faults={faults} (require 0)")
        if losses > wins:
            blockers.append(f"validation losses={losses} exceed wins={wins}")
        if isinstance(score, (int, float)) and float(score) <= 0.5 and wins == 0:
            blockers.append(
                f"validation score_rate={score} is draw-dominated with zero wins — not overnight-ready"
            )
        ckpt = best.get("best_checkpoint")
        if not ckpt or not Path(str(ckpt)).is_file():
            blockers.append("best checkpoint missing on disk")

    # Unattended laptop guards (pre-overnight)
    try:
        import shutil

        usage = shutil.disk_usage(REPO)
        free_gb = usage.free / (1024**3)
        if free_gb < 5.0:
            blockers.append(f"free disk {free_gb:.1f} GiB < 5 GiB unattended guard")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"disk guard unavailable: {exc}")

    decision = "BLOCKED" if blockers else "READY"
    return {
        "schema_version": 1,
        "kind": "OVERNIGHT_READINESS_GATE",
        "decision": decision,
        "status": decision,
        "blockers": blockers,
        "warnings": warnings,
        "best_candidate": {
            "arm_id": best.get("arm_id"),
            "architecture": best.get("architecture"),
            "stop_reason": best.get("stop_reason"),
            "best_checkpoint": best.get("best_checkpoint"),
            "best_validation": best.get("best_validation"),
        }
        if best
        else None,
        "holdout_unused": True,
        "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
        "note": "Overnight requires real validation wins with zero protocol faults. Holdout remains sealed.",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    initial = json.loads(args.initial.read_text(encoding="utf-8"))
    gate = evaluate_overnight_readiness(initial)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": gate["decision"], "path": str(args.out), "blockers": gate["blockers"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
