"""Register REWARD-SHAPING-R2 run/checkpoint records (EV-0047 adjudication).

R2 executed 2026-08-16 on yanhy83zocslwp; all four arms exit=0. Adjudication
under the predeclared plan: control survived both seeds; LAND-POTENTIAL
advances TELEMETRY-GRADE ONLY (gameplay arbiter deferred by the
LEARNING-PATH-INTEGRITY audit amendment until the training-regime correction).
Control reproduction vs preserved R1 arms: update-0 metrics identical to
~1e-7, slow divergence afterwards - recorded integrity flag attributed to
cross-pod GPU nondeterminism (different pod/hardware instance than R1), not
code/config drift.
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
RUN_ROOT = REPO / "experiments/marathon/screening_runs_reward_shaping_r2"
R2_PLAN = REPO / "experiments/marathon/reward_shaping_round_2_plan.yaml"
POD_ID = "yanhy83zocslwp"
SOURCE_COMMIT = "c7233f8"

REGIME = "EARLY_WINDOW_RESET_REGIME_V1"

VERDICTS = {
    "RSH2-A0-CONTROL-S1": (
        "CONTROL_SURVIVED; reproduces R1 control to ~1e-7 at update 0 with slow "
        "cross-pod GPU-nondeterminism drift (integrity flag recorded, EV-0047)"
    ),
    "RSH2-A0-CONTROL-S2": (
        "CONTROL_SURVIVED; reproduces R1 control to ~1e-7 at update 0 with slow "
        "cross-pod GPU-nondeterminism drift (integrity flag recorded, EV-0047)"
    ),
    "RSH2-A1-LAND-S1": (
        "ADVANCES_TELEMETRY_GRADE_ONLY under " + REGIME + "; gameplay arbiter "
        "deferred pending LEARNING-PATH-INTEGRITY regime correction"
    ),
    "RSH2-A1-LAND-S2": (
        "ADVANCES_TELEMETRY_GRADE_ONLY under " + REGIME + "; gameplay arbiter "
        "deferred pending LEARNING-PATH-INTEGRITY regime correction"
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
    plan_digest = hashlib.sha256(R2_PLAN.read_text(encoding="utf-8").encode()).hexdigest()[:12]
    experiment_id = canonical_id("experiment", "reward-shaping-r2", plan_digest)
    if not registry.exists(experiment_id):
        print(f"ERROR: R2 experiment record missing: {experiment_id}")
        return 1
    added = 0
    for arm_dir in sorted(RUN_ROOT.iterdir()):
        summary_path = arm_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json_load(summary_path)
        name = arm_dir.name
        material = hashlib.sha256(name.encode()).hexdigest()[:12]
        run_record = {
            "KIND": "run",
            "ID": canonical_id("run", name.lower(), material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": name,
            "EXPERIMENT_ID": experiment_id,
            "PPO_SEMANTICS": "UNCHANGED",
            "TRAINING_REGIME": REGIME,
            "LINEAGE": lineage,
            "GEOMETRY": summary["geometry"],
            "SEEDS": [summary["seed"]],
            "REWARD_SHAPE": summary["reward_shape"],
            "REWARD_SHAPE_BETA": summary["reward_shape_beta"],
            "BUDGET": f"{summary['budget_transitions']} transitions / {summary['updates']} updates",
            "ACTUAL_TRANSITIONS": summary["actual_transitions"],
            "COMMAND": (
                "remote_reward_shaping_r2_orchestrator.sh -> run_sh_r1_arm.py --arm-id "
                f"{name} --budget-transitions 1966080 --seed {summary['seed']} "
                f"--reward-shape {summary['reward_shape']} "
                f"--reward-shape-beta {summary['reward_shape_beta']} (self-stop wrapper)"
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
                f"DECISIVE_SHARE_LAST_{summary['metrics'].get('DECISIVE_SHARE_LAST', 0.0)}"
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
                "TRAINING_REGIME": REGIME,
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
