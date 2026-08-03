"""Replay summary helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from generals_bot.schemas import SCHEMA_VERSION


def write_replay_summary(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": SCHEMA_VERSION, **payload}
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return path


def load_replay_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
