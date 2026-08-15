"""Tests for the deterministic submission outbox allocator (Stage 4B)."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.submission import outbox as outbox_mod  # noqa: E402
from generals_bot.submission.outbox import (  # noqa: E402
    OutboxCollisionError,
    allocate_outbox,
    next_version,
)


def _make_zip(path: Path, payload: bytes = b"run.sh payload") -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("run.sh", payload.decode("utf-8"))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_allocates_v001_with_sidecars(tmp_path: Path) -> None:
    source = _make_zip(tmp_path / "source.zip")
    out = allocate_outbox("QS-TEST-V1", source, date="2026-08-15", root=tmp_path / "outbox")
    assert out.version == 1
    assert out.zip_path.name == "qs-QS-TEST-V1-v001-2026-08-15.zip"
    assert out.zip_path.is_file()
    assert out.sha256 == _sha256(out.zip_path) == _sha256(source)
    manifest = json.loads(out.manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidate_id"] == "QS-TEST-V1"
    assert manifest["sha256"] == out.sha256
    assert manifest["upload_policy"].startswith("MANUAL_UPLOAD_ONLY")
    assert out.sidecar_path.read_text(encoding="utf-8") == f"{out.sha256}  {out.zip_path.name}\n"
    # no staging debris
    assert not [p for p in out.zip_path.parent.iterdir() if p.name.startswith(".")]


def test_collision_refused_and_no_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_zip(tmp_path / "source.zip")
    root = tmp_path / "outbox"
    # Simulate a concurrent allocator that already occupied the computed slot.
    root.mkdir()
    squatter = root / "qs-QS-TEST-V1-v001-2026-08-15.zip"
    squatter.write_bytes(b"pre-existing immutable bytes")
    monkeypatch.setattr(outbox_mod, "next_version", lambda *a, **k: 1)
    with pytest.raises(OutboxCollisionError):
        allocate_outbox("QS-TEST-V1", source, date="2026-08-15", root=root)
    # original bytes untouched, no partial staging files
    files = sorted(p.name for p in root.iterdir())
    assert files == ["qs-QS-TEST-V1-v001-2026-08-15.zip"]
    assert squatter.read_bytes() == b"pre-existing immutable bytes"


def test_version_increments_across_dates(tmp_path: Path) -> None:
    source = _make_zip(tmp_path / "source.zip")
    root = tmp_path / "outbox"
    first = allocate_outbox("QS-TEST-V1", source, date="2026-08-15", root=root)
    second = allocate_outbox("QS-TEST-V1", source, date="2026-08-16", root=root)
    assert (first.version, second.version) == (1, 2)
    assert second.zip_path.name == "qs-QS-TEST-V1-v002-2026-08-16.zip"
    assert next_version("QS-TEST-V1", root=root) == 3
    # unrelated candidate unaffected
    assert next_version("QS-OTHER", root=root) == 1


def test_rejects_malformed_inputs(tmp_path: Path) -> None:
    source = _make_zip(tmp_path / "source.zip")
    root = tmp_path / "outbox"
    with pytest.raises(ValueError):
        allocate_outbox("../escape", source, date="2026-08-15", root=root)
    with pytest.raises(ValueError):
        allocate_outbox("QS-TEST-V1", source, date="15-08-2026", root=root)
    with pytest.raises(FileNotFoundError):
        allocate_outbox("QS-TEST-V1", tmp_path / "missing.zip", date="2026-08-15", root=root)
    assert not root.exists() or not list(root.iterdir())
