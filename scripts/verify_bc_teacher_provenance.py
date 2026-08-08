#!/usr/bin/env python3
"""Verify the BC teacher against the immutable submitted fallback package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEACHER_ID = "heuristic_v2f_plus_planner_terminal_form"
TEACHER_CANONICAL_ID = "heuristic_v2f_plus_planner_terminal_fix"
FALLBACK_ID = "QS-PUBLIC-V001"
EXPECTED_FALLBACK_SHA = "e1237f77dee469935fc3a60811b9a34522b83dd37bf4d76fa2555e6107a8edfa"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def aggregate_hash(entries: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(entries):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entries[name])
        digest.update(b"\0")
    return digest.hexdigest()


BEHAVIOUR_PROBE = r"""
import json
from generals_bot.observation import GameContext, Observation
from generals_bot.policies.base import TraceLevel
from generals_bot.policies.heuristic_v2_ablations import create_ablation

policy = create_ablation("heuristic_v2f_plus_planner_terminal_fix")
rows = []
for turn in (10, 100, 400, 800, 1050, 1150):
    obs = Observation(
        height=5, width=5, turn=turn, my_land=6, my_army=40,
        opp_land=4, opp_army=20,
        type_grid=((4,1,1,0,1),(1,1,1,1,1),(1,1,1,1,1),(1,1,1,2,1),(0,1,1,1,1)),
        owner_grid=((1,1,1,0,0),(1,1,0,0,0),(1,0,0,0,0),(0,0,0,2,2),(0,0,0,2,0)),
        army_grid=((15,4,3,0,0),(5,2,0,0,0),(3,0,0,0,0),(0,0,0,8,4),(0,0,0,6,0)),
    )
    state = policy.initial_state(GameContext(0, 5, 5))
    decision = policy.act(obs, state, deterministic=True, trace=TraceLevel.DECISION, deadline=None)
    rows.append({
        "turn": turn,
        "action": decision.action.as_tuple(),
        "option": decision.strategic_option,
    })
print(json.dumps(rows, sort_keys=True))
"""


def run_behaviour_probe(pythonpath: str, cwd: Path) -> list[dict]:
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath
    output = subprocess.check_output(
        [sys.executable, "-c", BEHAVIOUR_PROBE],
        cwd=cwd,
        env=env,
        text=True,
        timeout=60,
    )
    return json.loads(output)


def verify(package: Path) -> dict:
    package_bytes = package.read_bytes()
    package_sha = sha256_bytes(package_bytes)
    with zipfile.ZipFile(package) as archive:
        package_entries = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("generals_bot/")
            and name.endswith(".py")
            and not name.startswith("generals_bot/submission/")
        }
        main_py = archive.read("main.py").decode("utf-8")

    source_entries: dict[str, bytes] = {}
    mismatches: list[str] = []
    for package_name, package_data in package_entries.items():
        relative = Path(package_name).relative_to("generals_bot")
        source_path = ROOT / "src" / "generals_bot" / relative
        if not source_path.is_file():
            mismatches.append(f"missing:{relative.as_posix()}")
            continue
        source_data = source_path.read_bytes()
        source_entries[package_name] = source_data
        if source_data != package_data:
            mismatches.append(f"bytes:{relative.as_posix()}")

    source_behaviour = run_behaviour_probe(str(ROOT / "src"), ROOT)
    package_behaviour = run_behaviour_probe(str(package), ROOT)
    behaviour_equal = source_behaviour == package_behaviour
    source_hash = aggregate_hash(source_entries)
    package_source_hash = aggregate_hash(package_entries)
    main_selects_teacher = (
        'create_ablation("heuristic_v2f_plus_planner_terminal_fix")' in main_py
    )
    equal = bool(
        package_sha == EXPECTED_FALLBACK_SHA
        and not mismatches
        and source_hash == package_source_hash
        and behaviour_equal
        and main_selects_teacher
    )
    return {
        "schema_version": 1,
        "kind": "HEURISTIC_BASELINE_PROVENANCE",
        "status": "PASS" if equal else "RECORDED_DIFFERENCE",
        "written_at": datetime.now(UTC).isoformat(),
        "BC_TEACHER_ID": TEACHER_ID,
        "BC_TEACHER_CANONICAL_ID": TEACHER_CANONICAL_ID,
        "BC_TEACHER_SOURCE_HASH": source_hash,
        "FALLBACK_PACKAGE_ID": FALLBACK_ID,
        "FALLBACK_PACKAGE_SHA": package_sha,
        "FALLBACK_PACKAGE_EXPECTED_SHA": EXPECTED_FALLBACK_SHA,
        "BC_TEACHER_EQUALS_FALLBACK": equal,
        "evidence": {
            "package_path": str(package.resolve()),
            "package_runtime_source_hash": package_source_hash,
            "runtime_source_file_count": len(package_entries),
            "runtime_source_mismatches": mismatches,
            "main_selects_canonical_teacher": main_selects_teacher,
            "behaviour_probe_equal": behaviour_equal,
            "source_behaviour": source_behaviour,
            "package_behaviour": package_behaviour,
            "fallback_immutability": "READ_ONLY_HASHED_NOT_MODIFIED",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "manifests" / "bc_teacher_fallback_provenance.json",
    )
    args = parser.parse_args()
    report = verify(args.package)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
