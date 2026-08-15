"""Contract tests for the read-only marathon registry dashboard route (4B)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "dashboard" / "backend"))

from dashboard.backend.app.readers.marathon_registry import (  # noqa: E402
    RECOGNISED_KINDS,
    marathon_registry_dto,
)


def test_real_registry_is_typed_and_complete() -> None:
    dto = marathon_registry_dto(REPO)
    assert dto["authority"] == "STAGE_3_CANONICAL_REGISTRY"
    for kind in RECOGNISED_KINDS:
        assert kind in dto["counts"]
        assert dto["counts"][kind] == len(dto["records"][kind])
    # Every tracked registry record must be recognised (no silent drops).
    records_dir = REPO / "experiments/marathon/registry/records"
    total = len(list(records_dir.glob("*.json")))
    assert sum(dto["counts"].values()) + len(dto["malformed"]) == total
    assert dto["malformed"] == [], f"unrecognised registry kinds: {dto['malformed']}"
    # experiments carry semantics + lineage summaries
    for record in dto["records"]["experiment"]:
        assert record["PPO_SEMANTICS"] in {"UNCHANGED", "PRE_SAMPLING_MASK", "OFF_POLICY_AUXILIARY", "EVAL_ONLY"}


def test_missing_registry_dir_is_empty_not_error(tmp_path: Path) -> None:
    dto = marathon_registry_dto(tmp_path)
    assert all(count == 0 for count in dto["counts"].values())
    assert dto["malformed"] == []


def test_malformed_json_is_surfaced(tmp_path: Path) -> None:
    records = tmp_path / "experiments/marathon/registry/records"
    records.mkdir(parents=True)
    (records / "broken__x__000000000000.json").write_text("{not json", encoding="utf-8")
    good = {"KIND": "run", "ID": "run#t#000000000000", "NAME": "t"}
    (records / "run__t__000000000000.json").write_text(json.dumps(good), encoding="utf-8")
    dto = marathon_registry_dto(tmp_path)
    assert dto["counts"]["run"] == 1
    assert dto["malformed"] == ["broken__x__000000000000.json"]
