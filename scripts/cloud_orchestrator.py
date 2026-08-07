#!/usr/bin/env python3
"""Low-overhead watchdog for the authoritative RunPod PPO process.

The watchdog never terminates or stops infrastructure.  It records liveness and
requests a graceful trainer stop at the fixed budget ceiling; the trainer owns
the final atomic checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def process_command(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def gpu_snapshot() -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        utilization, used, total = [
            int(item.strip()) for item in result.stdout.splitlines()[0].split(",")
        ]
        return {
            "utilization_percent": utilization,
            "memory_used_mib": used,
            "memory_total_mib": total,
        }
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_complete_checkpoint(path: Path) -> tuple[bool, list[str]]:
    problems: list[str] = []
    complete = path / "COMPLETE"
    manifest_path = path / "sha256_manifest.json"
    if not complete.is_file():
        problems.append("missing COMPLETE")
    if not manifest_path.is_file():
        problems.append("missing sha256_manifest.json")
        return False, problems
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            artifact = path / name
            if not artifact.is_file():
                problems.append(f"missing {name}")
            elif artifact.stat().st_size != int(expected["bytes"]):
                problems.append(f"size mismatch {name}")
            elif sha256(artifact) != expected["sha256"]:
                problems.append(f"sha256 mismatch {name}")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        problems.append(f"invalid manifest: {exc}")
    return not problems, problems


def run(args: argparse.Namespace) -> int:
    stop_at = datetime.fromisoformat(args.stop_at.replace("Z", "+00:00"))
    state_path = args.runtime / "orchestrator_state.json"
    metrics_path = args.runtime / "training" / "metrics" / "cloud_training_latest.json"
    stop_request = args.runtime / "training" / "STOP_REQUEST"
    while True:
        command = process_command(args.trainer_pid)
        trainer_alive = (
            command is not None
            and "cloud_gpu_last_push.py" in command
            and " train " in f" {command} "
        )
        now = datetime.now(UTC)
        if now >= stop_at and trainer_alive and not stop_request.exists():
            stop_request.parent.mkdir(parents=True, exist_ok=True)
            stop_request.write_text(
                json.dumps({"reason": "TRAINING_BUDGET_STOP_CEILING", "written_at": utc_now()})
                + "\n",
                encoding="utf-8",
            )
        latest = None
        try:
            latest = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
        state = {
            "schema_version": 1,
            "kind": "CLOUD_ORCHESTRATOR_STATE",
            "status": "MONITORING" if trainer_alive else "TRAINER_EXITED",
            "trainer_pid": args.trainer_pid,
            "trainer_alive": trainer_alive,
            "trainer_command": command,
            "training_budget_stop_at": args.stop_at,
            "stop_request_present": stop_request.exists(),
            "latest_training": latest,
            "gpu": gpu_snapshot() if trainer_alive else None,
            "automatic_terminate_authorized": False,
            "automatic_stop_requires_verified_complete_checkpoint": True,
            "written_at": utc_now(),
        }
        atomic_json(state_path, state)
        if not trainer_alive:
            programme_path = args.runtime / "programme_state.json"
            checkpoint_ok = False
            checkpoint_problems = ["programme_state.json unavailable"]
            try:
                programme = json.loads(programme_path.read_text(encoding="utf-8"))
                final_path = Path(programme["final_checkpoint"])
                checkpoint_ok, checkpoint_problems = verify_complete_checkpoint(final_path)
            except (FileNotFoundError, KeyError, OSError, json.JSONDecodeError):
                pass
            state.update(
                {
                    "status": (
                        "COMPLETE_CHECKPOINT_VERIFIED"
                        if checkpoint_ok
                        else "CHECKPOINT_VERIFICATION_FAILED"
                    ),
                    "complete_checkpoint_verified": checkpoint_ok,
                    "checkpoint_problems": checkpoint_problems,
                    "written_at": utc_now(),
                }
            )
            atomic_json(state_path, state)
            return 0 if checkpoint_ok else 2
        time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--trainer-pid", type=int, required=True)
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
