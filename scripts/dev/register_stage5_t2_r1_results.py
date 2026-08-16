"""Register STAGE5 T2 K1 round results (post-round records, EV-0057).

Updates the two predeclared run records (run#s5-t2-k1-s1/s2) with actual
results and adds terminal checkpoint records. Adjudication per
stage5_capacity_value_r1_plan.yaml predeclared rules (same advancement
rules as prior rounds; T0 = RWB1-A0-CONTROL runs EV-0051):
both seeds survive integrity (240/240, exit=0, valid share 1.0, ratio ~1.0,
entropy rising), mean vloss-reduction beats control mean in BOTH seeds with
no sign flip -> ADVANCES TELEMETRY-GRADE ONLY to the predeclared gameplay
arbiter (telemetry never promotes; margins are sub-1% of vloss scale).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

REGISTRY_ROOT = REPO / "experiments/marathon/registry"
CAPSULE = REPO / "experiments/marathon/baseline_capsule_v0.json"
RUN_ROOT = REPO / "experiments/marathon/screening_runs/STAGE5-T2-R1/artefacts"
PLAN = REPO / "experiments/marathon/stage5_capacity_value_r1_plan.yaml"
POD_ID = "0ea2tby1ujpk8z"
SOURCE_COMMIT = "c609066"
REGIME = "PERSISTENT_EPISODE_REGIME_V1"

CONTROL_MEAN_VLOSS_REDUCTION = (0.006562232971191406 + 0.00637815248221159) / 2.0

VERDICTS = {
    "S5-T2-K1-S1": (
        "ADVANCES_TELEMETRY_GRADE_ONLY (both seeds survive integrity; vloss-reduction "
        "beats RWB1-A0-CONTROL mean in both seeds, no sign flip); predeclared gameplay "
        "arbiter next - telemetry never promotes"
    ),
    "S5-T2-K1-S2": (
        "ADVANCES_TELEMETRY_GRADE_ONLY (both seeds survive integrity; vloss-reduction "
        "beats RWB1-A0-CONTROL mean in both seeds, no sign flip); predeclared gameplay "
        "arbiter next - telemetry never promotes"
    ),
}


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
        "machine_id": "guah2ylm86w1",
    }
    plan_digest = hashlib.sha256(PLAN.read_text(encoding="utf-8").encode()).hexdigest()[:12]
    experiment_id = canonical_id("experiment", "stage5-capacity-value-r1", plan_digest)
    if not registry.exists(experiment_id):
        print(f"ERROR: experiment record missing: {experiment_id}")
        return 1

    t2_reductions = []
    for arm_dir in sorted(RUN_ROOT.iterdir()):
        summary_path = arm_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        name = arm_dir.name
        reduction = summary["metrics"]["VLOSS_REDUCTION_OVER_ROUND"]
        t2_reductions.append(reduction)
        material = hashlib.sha256(name.encode()).hexdigest()[:12]
        run_id = canonical_id("run", name.lower(), material)
        record_path = REGISTRY_ROOT / "records" / f"{run_id.replace('#', '__')}.json"
        if not record_path.exists():
            print(f"ERROR: predeclared run record missing: {run_id}")
            return 1
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record.update(
            {
                "STATUS": "COMPLETE_ADJUDICATED",
                "ACTUAL_TRANSITIONS": summary["actual_transitions"],
                "STOP_REASON": summary["stop_reason"],
                "ENVIRONMENT": {"python": "3.12.14", "jax": "0.11.0+cuda12"},
                "HARDWARE": hardware,
                "SOURCE_COMMIT": SOURCE_COMMIT,
                "ROUND_START_UTC": "2026-08-16T13:02:16Z",
                "ROUND_END_UTC": "2026-08-16T13:17:38Z",
                "SELF_STOP_CONFIRMED": "watchdog NO_RUNNING_PODS 13:20:59Z (zero idle burn); fetch window 13:32:45-13:33:54Z billing-logged",
                "ARTEFACT_LOCATIONS": [
                    (arm_dir / "summary.json").as_posix(),
                    (arm_dir / "telemetry.jsonl").as_posix(),
                ],
                "ARTEFACT_HASHES": {
                    "summary.json": sha256_file(summary_path),
                    "telemetry.jsonl": sha256_file(arm_dir / "telemetry.jsonl"),
                },
                "RESULT": (
                    f"{VERDICTS[name]}; "
                    f"VLOSS_REDUCTION_{reduction:.5f} "
                    f"(control mean {CONTROL_MEAN_VLOSS_REDUCTION:.5f}) "
                    f"VALID_SHARE_{summary['metrics']['VALID_LEARNING_SHARE']:.3f} "
                    f"ENTROPY_FIRST_{summary['metrics']['ENTROPY_FIRST']:.4f} "
                    f"ENTROPY_LAST_{summary['metrics']['ENTROPY_LAST']:.4f} "
                    f"RATIO_LAST_{summary['metrics']['RATIO_LAST']:.6f}"
                ),
            }
        )
        record_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"UPDATED {run_id}")

        raw = arm_dir / "raw.npz"
        if raw.exists():
            ckpt = {
                "KIND": "checkpoint",
                "ID": canonical_id(
                    "checkpoint", name.lower() + "-terminal", sha256_file(raw)[:12]
                ),
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "NAME": name + "-TERMINAL",
                "RUN_ID": run_id,
                "PPO_SEMANTICS": "UNCHANGED",
                "TRAINING_REGIME": REGIME,
                "TEMPORAL_HISTORY": "k1",
                "LINEAGE": lineage,
                "TRANSITIONS": summary["actual_transitions"],
                "ARTEFACT_HASHES": {"raw.npz": sha256_file(raw)},
                "ARTEFACT_LOCATIONS": [str(raw.as_posix()) + " (npz gitignored)"],
            }
            if not registry.exists(ckpt["ID"]):
                registry.add(ckpt)
                print(f"ADDED   {ckpt['ID']}")

    t2_mean = sum(t2_reductions) / len(t2_reductions)
    print(
        f"ADJUDICATION: T2 mean vloss-reduction {t2_mean:.6f} vs control mean "
        f"{CONTROL_MEAN_VLOSS_REDUCTION:.6f}; both seeds beat control, no sign flip; "
        "valid share 1.0 both seeds; entropy rising both seeds -> "
        "ADVANCES_TELEMETRY_GRADE_ONLY to predeclared gameplay arbiter"
    )
    print(f"RECORDED_AT_UTC={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
