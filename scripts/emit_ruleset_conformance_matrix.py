"""Emit Phase 9E ruleset conformance matrix from critical parity tests."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/manifests/ruleset_conformance_matrix.json"
REGISTRY = REPO / "experiments/manifests/ruleset_source_registry.json"

ROWS = [
    {
        "id": "map_geometry_padding",
        "critical": True,
        "tests": [
            "tests/rules_conformance/test_critical_parity.py::test_local_constants_match_competition_preset",
            "tests/rules_conformance/test_critical_parity.py::test_competition_reset_pads_to_21",
            "tests/rules_conformance/test_critical_parity.py::test_padding_mask_for_all_supported_active_sizes",
        ],
    },
    {
        "id": "castle_economy",
        "critical": True,
        "tests": [
            "tests/rules_conformance/test_castle_prices.py::test_prices_match_official",
            "tests/rules_conformance/test_critical_parity.py::test_valid_build_resolves_before_moves_and_deducts_cost",
            "tests/rules_conformance/test_critical_parity.py::test_invalid_build_is_consumed_as_pass",
        ],
    },
    {
        "id": "simultaneous_resolution",
        "critical": True,
        "tests": [
            "tests/rules_conformance/test_critical_parity.py::test_move_order_bigger_army_holds_contested_neutral",
        ],
    },
    {
        "id": "fog_visibility",
        "critical": True,
        "tests": [
            "tests/rules_conformance/test_critical_parity.py::test_visibility_is_local_moore_neighbourhood",
        ],
    },
    {
        "id": "logistics_growth",
        "critical": True,
        "tests": [
            "tests/rules_conformance/test_critical_parity.py::test_land_growth_every_fifty_turns",
        ],
    },
    {
        "id": "deathtouch_terminal",
        "critical": True,
        "tests": [
            "tests/rules_conformance/test_critical_parity.py::test_deathtouch_threshold_aligned",
        ],
    },
]


def _run_tests(nodeids: list[str]) -> tuple[str, str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=line",
        *nodeids,
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    status = "PASS" if proc.returncode == 0 else "FAIL"
    detail = (proc.stdout + "\n" + proc.stderr).strip()
    return status, detail[-4000:]


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.is_file() else {}
    results = []
    critical_fail = False
    for row in ROWS:
        status, detail = _run_tests(row["tests"])
        if row["critical"] and status == "FAIL":
            critical_fail = True
        results.append(
            {
                "id": row["id"],
                "critical": row["critical"],
                "status": status,
                "tests": row["tests"],
                "detail_tail": detail,
            }
        )

    gate = "FAIL" if critical_fail else "PASS"
    report = {
        "schema_version": 1,
        "kind": "RULESET_CONFORMANCE_MATRIX",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine_pin": registry.get("engine_pin"),
        "source_registry": str(REGISTRY.relative_to(REPO)) if REGISTRY.is_file() else None,
        "rows": results,
        "RESEARCH_RULESET_INTEGRITY_GATE": gate,
        "schema_repairs_required": False,
        "checkpoint_migration_required": False,
        "notes": [
            "Core critical rows must PASS before neural Phase 9E training.",
            "No schema-incompatible repairs applied in this matrix emission.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"RESEARCH_RULESET_INTEGRITY_GATE": gate, "out": str(OUT)}, indent=2))
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
