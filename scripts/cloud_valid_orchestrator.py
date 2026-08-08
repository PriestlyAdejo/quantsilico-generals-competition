#!/usr/bin/env python3
"""Lightweight detached monitor for the valid-learning cloud trainer."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--stale-seconds", type=float, default=300.0)
    args = parser.parse_args()
    stop_at = datetime.fromisoformat(args.stop_at.replace("Z", "+00:00"))
    latest = args.runtime / "metrics" / "cloud_training_latest.json"
    output = args.runtime / "orchestrator_state.json"
    status = "MONITORING"
    reason = None

    while pid_alive(args.pid):
        now = datetime.now(UTC)
        if now >= stop_at:
            (args.runtime / "STOP_REQUEST").touch(exist_ok=True)
            status = "STOP_REQUESTED"
            reason = "TRAINING_BUDGET_STOP"
        metric = None
        if latest.is_file():
            metric = json.loads(latest.read_text(encoding="utf-8"))
            age = time.time() - latest.stat().st_mtime
            if age > args.stale_seconds:
                (args.runtime / "STOP_REQUEST").touch(exist_ok=True)
                status = "STOP_REQUESTED"
                reason = f"STALE_TRAINING_METRICS:{age:.1f}s"
        atomic_json(
            output,
            {
                "schema_version": 1,
                "kind": "CLOUD_VALID_LEARNING_ORCHESTRATOR",
                "status": status,
                "reason": reason,
                "trainer_pid": args.pid,
                "trainer_alive": True,
                "stop_at": args.stop_at,
                "latest_metric": metric,
                "written_at": utc_now(),
            },
        )
        time.sleep(args.poll_seconds)

    programme = args.runtime / "programme_state.json"
    final_state = json.loads(programme.read_text(encoding="utf-8")) if programme.is_file() else None
    checkpoint = Path(final_state["final_checkpoint"]) if final_state else None
    checkpoint_complete = bool(checkpoint and (checkpoint / "COMPLETE").is_file())
    atomic_json(
        output,
        {
            "schema_version": 1,
            "kind": "CLOUD_VALID_LEARNING_ORCHESTRATOR",
            "status": (
                "TRAINER_EXITED_CHECKPOINT_COMPLETE"
                if checkpoint_complete
                else "TRAINER_EXITED_UNSAFE"
            ),
            "reason": reason,
            "trainer_pid": args.pid,
            "trainer_alive": False,
            "stop_at": args.stop_at,
            "programme_state": final_state,
            "checkpoint_complete": checkpoint_complete,
            "written_at": utc_now(),
        },
    )
    return 0 if checkpoint_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
