"""Register SH-R1 GPU ladder + horizon arm runs and their checkpoints.

Run AFTER retrieval of remote evidence (EV-0027). Registers:
  run#sh-r1-a40-ladder#* (screening throughput ladder, EVAL_ONLY telemetry)
  run#sh-r1-a1-horizon-64-a40#*, run#sh-r1-a2-horizon-32-a40#*
  checkpoint records for each arm's terminal raw weights
All PPO_SEMANTICS UNCHANGED; lineage from the baseline capsule (EV-0016).
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
EVIDENCE_ROOT = REPO / "experiments/marathon/remote_screening_runs_gpu"
EXPERIMENT_ID = "experiment#sh-r1-cpu-pilot#796c4d71603c"
POD_ID = "tmmov7t54z5mbu"

ARMS = {
    "SH-R1-A1-HORIZON-64": {"num_envs": 128, "rollout_len": 64},
    "SH-R1-A2-HORIZON-32": {"num_envs": 512, "rollout_len": 16},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    lineage = {
        "NAME": "competition_native_jax_v1",
        "IMPLEMENTATION_FINGERPRINT": capsule["source_identity"]["lineage_hashes"][
            "learner_implementation_hash"
        ][:16],
        "LINEAGE_HASHES": capsule["source_identity"]["lineage_hashes"],
    }
    registry = Registry(REGISTRY_ROOT)
    added = []

    ladder_log = EVIDENCE_ROOT / "ladder_run.log"
    ladder_record = {
        "KIND": "run",
        "ID": canonical_id("run", "sh-r1-a40-ladder", hashlib.sha256(
            ladder_log.read_bytes()).hexdigest()[:12]),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "NAME": "SH-R1-A40-ENV-LADDER",
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "PPO_SEMANTICS": "UNCHANGED",
        "LINEAGE": lineage,
        "HARDWARE": {"provider": "runpod", "pod_id": POD_ID, "gpu": "NVIDIA A40",
                     "rate_usd_per_hr": 0.44, "cloud_type": "SECURE",
                     "data_center": "EU-SE-1"},
        "SOURCE_COMMIT": "ce99e5d",
        "SEEDS": [20260815],
        "COMMAND": "PYTHONPATH=repo:src:third_party/generals-bots .venv312/bin/python "
                   "scripts/cloud_a100_env_ladder.py (CLOUD_LADDER_UPDATES=4)",
        "BUDGET": "7 geometries x 4 timed updates (throughput probe only)",
        "STOP_REASON": "LADDER_COMPLETE",
        "ENVIRONMENT": {"python": "3.12.14", "jax": "0.11.0+cuda12", "image":
                        "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"},
        "ARTEFACT_LOCATIONS": ["experiments/marathon/remote_screening_runs_gpu/ladder_run.log"],
        "ARTEFACT_HASHES": {"ladder_run.log": sha256_file(ladder_log)},
        "RESULT": "BEST_GEOMETRY_256x32_VALID_LEARNING_TPS_7611_GPU_UTIL_100 "
                  "E512_OOM EV-0026 EV-0027",
    }
    if not registry.exists(ladder_record["ID"]):
        registry.add(ladder_record)
        added.append(ladder_record["ID"])

    for arm_id, geometry in ARMS.items():
        arm_dir = EVIDENCE_ROOT / arm_id
        summary = json.loads((arm_dir / "summary.json").read_text(encoding="utf-8"))
        raw = arm_dir / "raw.npz"
        material = hashlib.sha256(arm_id.encode()).hexdigest()[:12]
        run_record = {
            "KIND": "run",
            "ID": canonical_id("run", arm_id.lower() + "-a40", material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": arm_id + "-A40",
            "EXPERIMENT_ID": EXPERIMENT_ID,
            "PPO_SEMANTICS": "UNCHANGED",
            "LINEAGE": lineage,
            "GEOMETRY": geometry,
            "SEEDS": [summary["seed"]],
            "BUDGET_TRANSITIONS": summary["budget_transitions"],
            "ACTUAL_TRANSITIONS": summary["actual_transitions"],
            "UPDATES": summary["updates"],
            "HARDWARE": {"provider": "runpod", "pod_id": POD_ID, "gpu": "NVIDIA A40",
                         "rate_usd_per_hr": 0.44, "cloud_type": "SECURE",
                         "data_center": "EU-SE-1"},
            "SOURCE_COMMIT": "ce99e5d",
            "COMMAND": f"PYTHONPATH=repo:src:third_party/generals-bots .venv312/bin/python "
                       f"scripts/training/run_sh_r1_arm.py --arm-id {arm_id} "
                       f"--num-envs {geometry['num_envs']} --rollout-len {geometry['rollout_len']} "
                       f"--checkpoint /workspace/ckpt_baseline_v0",
            "BUDGET": f"{summary['budget_transitions']} transitions",
            "STOP_REASON": summary["stop_reason"],
            "ENVIRONMENT": {"python": "3.12.14", "jax": "0.11.0+cuda12", "image":
                            "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"},
            "ARTEFACT_LOCATIONS": [
                f"experiments/marathon/remote_screening_runs_gpu/{arm_id}/summary.json",
                f"experiments/marathon/remote_screening_runs_gpu/{arm_id}/telemetry.jsonl",
            ],
            "ARTEFACT_HASHES": {
                "summary.json": sha256_file(arm_dir / "summary.json"),
                "telemetry.jsonl": sha256_file(arm_dir / "telemetry.jsonl"),
            },
            "RESULT": ("ELIMINATED_NO_VLOSS_REDUCTION_BUDGET_LIMITED "
                       f"VALID_SHARE_{summary['metrics']['VALID_LEARNING_SHARE']} "
                       "EV-0027"),
        }
        if not registry.exists(run_record["ID"]):
            registry.add(run_record)
            added.append(run_record["ID"])

        ckpt_record = {
            "KIND": "checkpoint",
            "ID": canonical_id("checkpoint", arm_id.lower() + "-a40-terminal",
                               sha256_file(raw)[:12]),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": arm_id + "-A40-TERMINAL",
            "RUN_ID": run_record["ID"],
            "PPO_SEMANTICS": "UNCHANGED",
            "LINEAGE": lineage,
            "TRANSITIONS": summary["actual_transitions"],
            "ARTEFACT_HASHES": {"raw.npz": sha256_file(raw)},
            "ARTEFACT_LOCATIONS": [
                f"experiments/marathon/remote_screening_runs_gpu/{arm_id}/raw.npz"
                " (npz gitignored; hash recorded)"
            ],
        }
        if not registry.exists(ckpt_record["ID"]):
            registry.add(ckpt_record)
            added.append(ckpt_record["ID"])

    for record_id in added:
        print(f"ADDED  {record_id}")
    print(f"TOTAL_NEW={len(added)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
