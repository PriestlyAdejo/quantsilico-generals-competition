"""Slow bridge benchmark marker (excluded from default fast suite unless selected)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.slow
def test_bridge_benchmark_report_reproducible_schema() -> None:
    path = Path("experiments/summaries/jax_pytorch_bridge_benchmark.json")
    assert path.is_file(), "run bridge benchmark before this slow check"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["schema_version"] >= 2
    assert report["decision"] in {"PASS", "PARTIAL", "FAIL"}
    assert "by_batch" in report
    assert "1" in report["by_batch"]
    stages = report["by_batch"]["1"]["stages"]
    assert "feature_channel_ms" in stages
    assert "mlp_forward_ms" in stages
    # Vectorised encode must not dominate after Phase 5 optimisation.
    assert stages["feature_channel_ms"]["pct_of_total"] < 25.0
