"""Filesystem experiment and model manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from generals_bot.schemas import ExperimentManifest, ModelManifest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = REPO_ROOT / "experiments" / "manifests"
MODELS = REPO_ROOT / "models" / "registry"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_file_hash(seed_path: Path) -> str:
    return _sha256_text(seed_path.read_text(encoding="utf-8"))


def create_experiment_manifest(**kwargs: Any) -> ExperimentManifest:
    manifest = ExperimentManifest(
        created_at=datetime.now(UTC).isoformat(),
        **kwargs,
    )
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENTS / f"{manifest.experiment_id}.json"
    write_json(path, manifest.to_dict())
    return manifest


def create_model_manifest(**kwargs: Any) -> ModelManifest:
    manifest = ModelManifest(**kwargs)
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"{manifest.model_id}.json"
    write_json(path, manifest.to_dict())
    return manifest
