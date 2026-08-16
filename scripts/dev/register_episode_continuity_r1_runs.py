"""Register EPISODE_CONTINUITY_R1 run/checkpoint records (EV-0049).

Regime question answered under predeclared rules: PERSISTENT_EPISODE_REGIME_V1
ADOPTED as the canonical screening regime (both seeds survived integrity;
first-ever terminal competition signal in screening: 2 decisive ticks in
PERSIST-S2). All reset-regime history preserved as VALID_IN_RESET_REGIME.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

REGISTRY_ROOT = REPO / "experiments/marathon/registry"
CAPSULE = REPO / "experiments/marathon/baseline_capsule_v0.json"
RUN_ROOT = REPO / "experiments/marathon/screening_runs_episode_continuity_r1"
PLAN = REPO / "experiments/marathon/episode_continuity_round_1_plan.yaml"
POD_ID = "yanhy83zocslwp"
SOURCE_COMMIT = "bd2cf36"

VERDICTS = {
    "EPC1-A0-RESET-S1": (
        "CONTROL_SURVIVED EARLY_WINDOW_RESET_REGIME_V1 baseline reproduced on-pod; "
        "decisive_share 0.0 confirms zero terminal experience in the reset regime"
    ),
    "EPC1-A0-RESET-S2": (
        "CONTROL_SURVIVED EARLY_WINDOW_RESET_REGIME_V1 baseline reproduced on-pod; "
        "decisive_share 0.0 confirms zero terminal experience in the reset regime"
    ),
    "EPC1-A1-PERSIST-S1": (
        "REGIME_ADOPTED PERSISTENT_EPISODE_REGIME_V1 survives integrity (valid share 1.0, "
        "finite, no entropy collapse); entropy_last 5.91 vs reset 2.11 records the "
        "state-distribution change"
    ),
    "EPC1-A1-PERSIST-S2": (
        "REGIME_ADOPTED PERSISTENT_EPISODE_REGIME_V1 survives integrity; FIRST-EVER "
        "terminal competition signal in screening history: 2 decisive ticks "
        "(decisive_share max 1.22e-4)"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    registry = Registry(REGISTRY_ROOT)
    capsule = json_load(CAPSULE)
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
    plan_digest = hashlib.sha256(PLAN.read_text(encoding="utf-8").encode()).hexdigest()[:12]
    experiment_id = canonical_id("experiment", "episode-continuity-r1", plan_digest)
    if not registry.exists(experiment_id):
        print(f"ERROR: experiment record missing: {experiment_id}")
        return 1
    added = 0
    for arm_dir in sorted(RUN_ROOT.iterdir()):
        summary_path = arm_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json_load(summary_path)
        name = arm_dir.name
        regime = (
            "PERSISTENT_EPISODE_REGIME_V1"
            if summary["episode_carry"] == "persistent"
            else "EARLY_WINDOW_RESET_REGIME_V1"
        )
        material = hashlib.sha256(name.encode()).hexdigest()[:12]
        run_record = {
            "KIND": "run",
            "ID": canonical_id("run", name.lower(), material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": name,
            "EXPERIMENT_ID": experiment_id,
            "PPO_SEMANTICS": "UNCHANGED",
            "TRAINING_REGIME": regime,
            "LINEAGE": lineage,
            "GEOMETRY": summary["geometry"],
            "SEEDS": [summary["seed"]],
            "EPISODE_CARRY": summary["episode_carry"],
            "BUDGET": f"{summary['budget_transitions']} transitions / {summary['updates']} updates",
            "ACTUAL_TRANSITIONS": summary["actual_transitions"],
            "COMMAND": (
                "remote_episode_continuity_r1_orchestrator.sh -> run_sh_r1_arm.py --arm-id "
                f"{name} --budget-transitions 1966080 --seed {summary['seed']} "
                f"--episode-carry {summary['episode_carry']} (self-stop wrapper)"
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
                f"{VERDICTS[name]}; "
                f"VLOSS_REDUCTION_{summary['metrics']['VLOSS_REDUCTION_OVER_ROUND']:.5f} "
                f"VALID_SHARE_{summary['metrics']['VALID_LEARNING_SHARE']:.3f} "
                f"ENTROPY_LAST_{summary['metrics']['ENTROPY_LAST']:.4f}"
            ),
        }
        if not registry.exists(run_record["ID"]):
            registry.add(run_record)
            added += 1
            print(f"ADDED  {run_record['ID']}")
        else:
            print(f"ALREADY_REGISTERED {run_record['ID']}")
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
                "TRAINING_REGIME": regime,
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
    print(f"RECORDED_AT_UTC={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
