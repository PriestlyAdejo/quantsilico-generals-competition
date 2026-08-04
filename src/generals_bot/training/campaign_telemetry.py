"""Durable campaign telemetry — atomic replace-safe observer records (amendment C)."""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
TELEMETRY_DIR = REPO / "var" / "dashboard" / "campaigns"
STALE_HEARTBEAT_S = 45.0


def _iso(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or time.time(), tz=timezone.utc).isoformat()


def campaign_path(campaign_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in campaign_id)
    return TELEMETRY_DIR / f"{safe}.json"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def new_campaign_record(
    *,
    campaign_id: str,
    stage: str,
    config_hash: str,
    architecture: str,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    now = time.time()
    return {
        "schema_version": 1,
        "kind": "DURABLE_CAMPAIGN_TELEMETRY",
        "campaign_id": campaign_id,
        "stage": stage,
        "state": "RUNNING",
        "heartbeat_at": _iso(now),
        "heartbeat_unix": now,
        "process": {
            "pid": os.getpid(),
            "hostname": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "NOT RECORDED",
        },
        "config_hash": config_hash,
        "architecture": architecture,
        "env_steps": 0,
        "ppo_updates": 0,
        "elapsed_s": 0.0,
        "current_checkpoint": checkpoint,
        "best_checkpoint": checkpoint,
        "latest_validation": None,
        "best_validation": None,
        "validation_game_count": 0,
        "plateau_patience_used": 0,
        "plateau_patience_remaining": None,
        "stop_conditions": [],
        "metrics": {},
        "reward_summary": {},
        "discovery": None,
        "conversion": None,
        "own_general_loss": None,
        "throughput": {},
        "hardware": {},
        "log_tail": [],
        "last_error": None,
        "final_stop_reason": None,
        "updated_at": _iso(now),
        "started_at": _iso(now),
    }


def persist_campaign(record: dict[str, Any]) -> Path:
    path = campaign_path(str(record["campaign_id"]))
    record = dict(record)
    now = time.time()
    record["heartbeat_at"] = _iso(now)
    record["heartbeat_unix"] = now
    record["updated_at"] = _iso(now)
    atomic_write_json(path, record)
    return path


def load_campaign(campaign_id: str) -> dict[str, Any] | None:
    path = campaign_path(campaign_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_campaigns(*, include_stale: bool = True) -> list[dict[str, Any]]:
    if not TELEMETRY_DIR.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    now = time.time()
    for path in sorted(TELEMETRY_DIR.glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        hb = float(rec.get("heartbeat_unix") or 0.0)
        stale = (now - hb) > STALE_HEARTBEAT_S if hb else True
        if rec.get("state") == "RUNNING" and stale:
            rec = dict(rec)
            rec["state"] = "INTERRUPTED"
            rec["observer_note"] = f"Stale heartbeat > {STALE_HEARTBEAT_S}s"
        if not include_stale and rec.get("state") == "INTERRUPTED":
            continue
        rows.append(rec)
    return rows


def append_log(record: dict[str, Any], line: str, *, limit: int = 40) -> None:
    tail = list(record.get("log_tail") or [])
    tail.append(f"{_iso()} {line}")
    record["log_tail"] = tail[-limit:]


def sample_hardware() -> dict[str, Any]:
    out: dict[str, Any] = {"cpu_percent": None, "ram_used_gb": None, "gpu": None}
    try:
        import psutil  # type: ignore

        out["cpu_percent"] = psutil.cpu_percent(interval=0.0)
        out["ram_used_gb"] = round(psutil.virtual_memory().used / (1024**3), 2)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            mem = torch.cuda.memory_allocated(idx) / (1024**2)
            out["gpu"] = {
                "name": props.name,
                "vram_allocated_mb": round(mem, 1),
                "total_vram_mb": round(props.total_memory / (1024**2), 1),
            }
    except Exception:
        pass
    return out
