"""Unit tests for emergency final package selection priority."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import emergency_final_selection_gate as gate  # noqa: E402


def test_selection_prefers_normal_over_learned(tmp_path: Path, monkeypatch):
    root = tmp_path
    (root / "submission/roles").mkdir(parents=True)
    (root / "experiments/manifests").mkdir(parents=True)
    (root / "submission/roles/recommended.json").write_text(
        json.dumps(
            {
                "status": "RECOMMENDED",
                "package_zip": "submission/packages/normal.zip",
                "package_sha256": "aa" * 32,
            }
        )
    )
    (root / "experiments/manifests/emergency_baseline_fallback.json").write_text(
        json.dumps(
            {
                "status": "EMERGENCY_BASELINE_FALLBACK_PACKAGE_EXISTS",
                "package_zip_repo": "submission/packages/base.zip",
                "package_sha256": "bb" * 32,
            }
        )
    )
    monkeypatch.setattr(gate, "ROOT", root)
    assert gate.main() == 0
    sel = json.loads((root / "experiments/manifests/final_emergency_package_selection_gate.json").read_text())
    assert sel["final_status"] == "NORMAL_UPLOAD_READY_CANDIDATE_EXISTS"
    assert "aa" * 32 == sel["selected"]["package_sha256"]


def test_selection_keeps_established_fallback_when_inconclusive(tmp_path: Path, monkeypatch):
    root = tmp_path
    (root / "submission/roles").mkdir(parents=True)
    (root / "experiments/manifests").mkdir(parents=True)
    (root / "submission").mkdir(parents=True, exist_ok=True)
    (root / "submission/roles/recommended.json").write_text(
        json.dumps({"status": "NO_CANDIDATE_CURRENTLY_RECOMMENDED", "package_zip": None})
    )
    (root / "experiments/manifests/emergency_baseline_fallback.json").write_text(
        json.dumps(
            {
                "status": "EMERGENCY_BASELINE_FALLBACK_PACKAGE_EXISTS",
                "package_zip_repo": "submission/packages/base.zip",
                "package_sha256": "cc" * 32,
                "already_submitted": True,
            }
        )
    )
    monkeypatch.setattr(gate, "ROOT", root)
    assert gate.main() == 0
    sel = json.loads((root / "experiments/manifests/final_emergency_package_selection_gate.json").read_text())
    assert sel["final_status"] == "EMERGENCY_BASELINE_FALLBACK_PACKAGE_EXISTS"
    md = (root / "submission/EMERGENCY_UPLOAD_THIS.md").read_text()
    assert "cc" * 32 in md
