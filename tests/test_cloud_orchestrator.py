from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.cloud_orchestrator import verify_complete_checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_complete_checkpoint_accepts_matching_atomic_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "raw.npz"
    artifact.write_bytes(b"checkpoint")
    (tmp_path / "COMPLETE").write_text('{"ok": true}\n', encoding="utf-8")
    (tmp_path / "sha256_manifest.json").write_text(
        json.dumps(
            {"files": {"raw.npz": {"bytes": artifact.stat().st_size, "sha256": _sha256(artifact)}}}
        ),
        encoding="utf-8",
    )

    assert verify_complete_checkpoint(tmp_path) == (True, [])


def test_verify_complete_checkpoint_rejects_corrupt_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "raw.npz"
    artifact.write_bytes(b"checkpoint")
    original_hash = _sha256(artifact)
    (tmp_path / "COMPLETE").write_text('{"ok": true}\n', encoding="utf-8")
    (tmp_path / "sha256_manifest.json").write_text(
        json.dumps(
            {"files": {"raw.npz": {"bytes": artifact.stat().st_size, "sha256": original_hash}}}
        ),
        encoding="utf-8",
    )
    artifact.write_bytes(b"corruption")

    ok, problems = verify_complete_checkpoint(tmp_path)

    assert not ok
    assert problems == ["sha256 mismatch raw.npz"]
