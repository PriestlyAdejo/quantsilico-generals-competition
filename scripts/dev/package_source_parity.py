"""Source vs unpacked-package action parity for deterministic heuristics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import TraceLevel
from generals_bot.selector import create_policy

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sample_obs(turn: int = 100) -> Observation:
    return Observation(
        height=5,
        width=5,
        turn=turn,
        my_land=6,
        my_army=40,
        opp_land=4,
        opp_army=20,
        type_grid=(
            (4, 1, 1, 0, 1),
            (1, 1, 1, 1, 1),
            (1, 1, 1, 1, 1),
            (1, 1, 1, 2, 1),
            (0, 1, 1, 1, 1),
        ),
        owner_grid=(
            (1, 1, 1, 0, 0),
            (1, 1, 0, 0, 0),
            (1, 0, 0, 0, 0),
            (0, 0, 0, 2, 2),
            (0, 0, 0, 2, 0),
        ),
        army_grid=(
            (15, 4, 3, 0, 0),
            (5, 2, 0, 0, 0),
            (3, 0, 0, 0, 0),
            (0, 0, 0, 8, 4),
            (0, 0, 0, 6, 0),
        ),
    )


def _load_packaged_policy(package: Path, candidate: str):
    staging = Path(tempfile.mkdtemp(prefix="pkg_parity_"))
    with zipfile.ZipFile(package, "r") as zf:
        zf.extractall(staging)
    # Package layout: generals_bot/ + main.py at root
    sys.path.insert(0, str(staging))
    try:
        if candidate == "heuristic_v1":
            from generals_bot.policies.heuristic_v1 import HeuristicV1Policy

            return HeuristicV1Policy(), staging
        if candidate == "heuristic_v2_qualifier":
            from generals_bot.policies.heuristic_v2_qualifier import HeuristicV2QualifierPolicy

            return HeuristicV2QualifierPolicy(), staging
        if candidate == "heuristic_v0":
            from generals_bot.policies.heuristic_v0 import HeuristicV0Policy

            return HeuristicV0Policy(), staging
        raise KeyError(candidate)
    finally:
        # Keep path for subsequent acts in same process; caller removes staging.
        pass


def compare_source_package_parity(
    *,
    candidate: str,
    package: Path,
    turns: list[int] | None = None,
) -> dict[str, Any]:
    turns = turns or [10, 100, 400, 800, 1050, 1150]
    source = create_policy(candidate)
    packaged, staging = _load_packaged_policy(package, candidate)
    mismatches: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    # Inspect package contains policy module
    pkg_files = {p.as_posix() for p in staging.rglob("*") if p.is_file()}
    has_castle = any("castle_cost.py" in f for f in pkg_files)
    has_policy = any(candidate.replace("qualifier", "").split("_")[0] in f or candidate in f for f in pkg_files)
    # More precise: check main.py content
    main_py = (staging / "main.py").read_text(encoding="utf-8")
    config_ok = candidate.replace("-", "_") in main_py or candidate.split("_")[1] in main_py
    if candidate == "heuristic_v2_qualifier":
        config_ok = "HeuristicV2QualifierPolicy" in main_py
    elif candidate == "heuristic_v1":
        config_ok = "HeuristicV1Policy" in main_py

    for turn in turns:
        obs = _sample_obs(turn)
        s_state = source.initial_state(GameContext(0, 5, 5))
        p_state = packaged.initial_state(GameContext(0, 5, 5))
        s_dec = source.act(obs, s_state, deterministic=True, trace=TraceLevel.DECISION, deadline=None)
        p_dec = packaged.act(obs, p_state, deterministic=True, trace=TraceLevel.DECISION, deadline=None)
        row = {
            "turn": turn,
            "source_action": s_dec.action.as_tuple(),
            "package_action": p_dec.action.as_tuple(),
            "source_option": s_dec.strategic_option,
            "package_option": p_dec.strategic_option,
            "match": s_dec.action == p_dec.action,
        }
        comparisons.append(row)
        if not row["match"]:
            mismatches.append(row)

    report = {
        "schema_version": 1,
        "kind": "SOURCE_PACKAGE_PARITY",
        "candidate": candidate,
        "package": str(package),
        "package_sha256": hashlib.sha256(package.read_bytes()).hexdigest() if package.exists() else None,
        "main_py_config_ok": config_ok,
        "castle_module_present": has_castle,
        "comparisons": comparisons,
        "mismatches": mismatches,
        "exact_action_equivalence": len(mismatches) == 0,
        "status": "PASS" if len(mismatches) == 0 and config_ok and has_castle else "FAIL",
    }
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="heuristic_v1")
    parser.add_argument(
        "--package",
        default=str(REPO_ROOT / "submission" / "packages" / "heuristic_v1_packaged.zip"),
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = compare_source_package_parity(candidate=args.candidate, package=Path(args.package))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
