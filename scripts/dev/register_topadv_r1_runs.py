"""Register TOPADV-R1 run + terminal checkpoint records.

Adapted from register_curriculum_r1_runs.py. Run AFTER adjudication
(EV-0043): registers all six arms and their terminal checkpoints from
local artefacts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

REGISTRY_ROOT = REPO / "experiments/marathon/registry"
CAPSULE = REPO / "experiments/marathon/baseline_capsule_v0.json"
RUN_ROOT = REPO / "experiments/marathon/topadv_round_1_runs/screening_runs_topadv_r1"
EXPERIMENT_ID = "experiment#top-advantage-fractions-r1#5166e1a5d9db"
POD_ID = "zu2pvculot0yhp"
SOURCE_COMMIT = "70d4669"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    registry = Registry(REGISTRY_ROOT)
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    hashes = capsule["source_identity"]["lineage_hashes"]
    lineage = {
        "NAME": "competition_native_jax_v1",
        "IMPLEMENTATION_FINGERPRINT": hashes["learner_implementation_hash"][:16],
        "LINEAGE_HASHES": hashes,
    }
    hardware = {
        "provider": "runpod",
        "pod_id": POD_ID,
        "gpu": "NVIDIA A40",
        "rate_usd_per_hr": 0.44,
        "cloud_type": "SECURE",
        "data_center": "EU-SE-1",
    }
    added = 0
    for arm_dir in sorted(RUN_ROOT.iterdir()):
        summary_path = arm_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        name = arm_dir.name
        material = hashlib.sha256(name.encode()).hexdigest()[:12]
        frac = summary["top_advantage_fraction"]
        run_record = {
            "KIND": "run",
            "ID": canonical_id("run", name.lower(), material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": name,
            "EXPERIMENT_ID": EXPERIMENT_ID,
            "PPO_SEMANTICS": "UNCHANGED",
            "LINEAGE": lineage,
            "GEOMETRY": summary["geometry"],
            "SEEDS": [summary["seed"]],
            "TOP_ADVANTAGE_FRACTION": frac,
            "MIN_GENERALS_DISTANCE": summary.get("min_generals_distance", 17),
            "BUDGET": f"{summary['budget_transitions']} transitions / {summary['updates']} updates",
            "ACTUAL_TRANSITIONS": summary["actual_transitions"],
            "COMMAND": (
                "remote_topadv_r1_orchestrator.sh -> run_sh_r1_arm.py --arm-id "
                f"{name} --budget-transitions 1966080 --seed {summary['seed']} "
                f"--top-advantage-fraction {frac} (pool-fixed runner, self-stop wrapper)"
            ),
            "STOP_REASON": summary["stop_reason"],
            "ENVIRONMENT": {"python": "3.12.14", "jax": "0.11.0+cuda12"},
            "HARDWARE": hardware,
            "SOURCE_COMMIT": SOURCE_COMMIT,
            "ARTEFACT_LOCATIONS": [
                (RUN_ROOT / name / "summary.json").as_posix(),
                (RUN_ROOT / name / "telemetry.jsonl").as_posix(),
            ],
            "ARTEFACT_HASHES": {
                "summary.json": sha256_file(summary_path),
                "telemetry.jsonl": sha256_file(arm_dir / "telemetry.jsonl"),
            },
            "RESULT": (
                f"SURVIVED VLOSS_REDUCTION_{summary['metrics']['VLOSS_REDUCTION_OVER_ROUND']:.5f} "
                f"VALID_SHARE_{summary['metrics']['VALID_LEARNING_SHARE']:.3f} "
                f"STEADY_E2E_TPS_{summary['throughput']['end_to_end_tps']:.0f} EV-0043"
            ),
        }
        if not registry.exists(run_record["ID"]):
            registry.add(run_record)
            added += 1
            print(f"ADDED  {run_record['ID']}")
        raw = arm_dir / "raw.npz"
        if raw.exists():
            ckpt = {
                "KIND": "checkpoint",
                "ID": canonical_id(
                    "checkpoint", name.lower() + "-terminal", sha256_file(raw)[:12]
                ),
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "NAME": name + "-TERMINAL",
                "RUN_ID": run_record["ID"],
                "PPO_SEMANTICS": "UNCHANGED",
                "LINEAGE": lineage,
                "TRANSITIONS": summary["actual_transitions"],
                "ARTEFACT_HASHES": {"raw.npz": sha256_file(raw)},
                "ARTEFACT_LOCATIONS": [str(raw.as_posix()) + " (npz gitignored)"],
            }
            if not registry.exists(ckpt["ID"]):
                registry.add(ckpt)
                added += 1
                print(f"ADDED  {ckpt['ID']}")
    print(f"TOTAL_NEW={added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
