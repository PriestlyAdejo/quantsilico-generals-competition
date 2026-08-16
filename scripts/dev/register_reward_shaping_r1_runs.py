"""Register REWARD-SHAPING-R1 run/checkpoint records (VOID round, EV-0046)
and the REWARD-SHAPING-R2 successor experiment (pre-launch).

Failed/void experiments stay discoverable: R1 arms are registered exactly as
run, with the VOID verdict in the RESULT field.
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
RUN_ROOT = (
    REPO / "experiments/marathon/reward_shaping_round_1_runs"
    / "screening_runs_reward_shaping_r1"
)
R1_EXPERIMENT_ID = "experiment#reward-shaping-r1#39ce1360218e"
R2_PLAN = REPO / "experiments/marathon/reward_shaping_round_2_plan.yaml"
POD_ID = "kkmqdgqbw3wgph"
SOURCE_COMMIT = "49e5976"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    registry = Registry(REGISTRY_ROOT)
    capsule = json_load_capsule()
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
        summary = json_load(summary_path)
        name = arm_dir.name
        material = hashlib.sha256(name.encode()).hexdigest()[:12]
        run_record = {
            "KIND": "run",
            "ID": canonical_id("run", name.lower(), material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": name,
            "EXPERIMENT_ID": R1_EXPERIMENT_ID,
            "PPO_SEMANTICS": "UNCHANGED",
            "LINEAGE": lineage,
            "GEOMETRY": summary["geometry"],
            "SEEDS": [summary["seed"]],
            "REWARD_SHAPE": summary["reward_shape"],
            "REWARD_SHAPE_BETA": summary["reward_shape_beta"],
            "BUDGET": f"{summary['budget_transitions']} transitions / {summary['updates']} updates",
            "ACTUAL_TRANSITIONS": summary["actual_transitions"],
            "COMMAND": (
                "remote_reward_shaping_r1_orchestrator.sh -> run_sh_r1_arm.py --arm-id "
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
                "VOID_SIGNAL_VACUOUS_EV-0046 bit-exact control-identical; "
                f"VLOSS_REDUCTION_{summary['metrics']['VLOSS_REDUCTION_OVER_ROUND']:.5f} "
                f"VALID_SHARE_{summary['metrics']['VALID_LEARNING_SHARE']:.3f} preserved"
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
    # Successor experiment registration (pre-launch contract)
    plan_digest = hashlib.sha256(R2_PLAN.read_text(encoding="utf-8").encode()).hexdigest()[:12]
    r2_record = {
        "KIND": "experiment",
        "ID": canonical_id("experiment", "reward-shaping-r2", plan_digest),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "NAME": "REWARD-SHAPING-R2",
        "PPO_SEMANTICS": "UNCHANGED",
        "SEEDS": [20260825, 20260827],
        "BUDGET_PER_ARM_UPDATES": 240,
        "BUDGET_PER_ARM_TRANSITIONS": 1966080,
        "ARMS": [
            "RSH2-A0-CONTROL-S1",
            "RSH2-A0-CONTROL-S2",
            "RSH2-A1-LAND-S1",
            "RSH2-A1-LAND-S2",
        ],
        "CONFIG_IDENTITY": {
            "compute_class": "GPU_RUNPOD_A40",
            "plan": "experiments/marathon/reward_shaping_round_2_plan.yaml",
            "plan_kind": "MARATHON_SCREENING_ROUND_PLAN",
            "status": "PREDECLARED",
            "knob": (
                "train/competition_native_jax/reward_shaping_jax.py "
                "(land_potential; return-invariant; additive)"
            ),
        },
        "LINEAGE": lineage,
        "EVIDENCE_LINKS": [
            "EV-0013", "EV-0015", "EV-0035", "EV-0036", "EV-0038",
            "EV-0039", "EV-0043", "EV-0044", "EV-0046",
        ],
        "RECORDED_AT_UTC": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not registry.exists(r2_record["ID"]):
        registry.add(r2_record)
        added += 1
        print(f"ADDED  {r2_record['ID']}")
    else:
        print(f"ALREADY_REGISTERED {r2_record['ID']}")
    print(f"TOTAL_NEW={added}")
    return 0


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def json_load_capsule() -> dict:
    import json

    return json.loads(CAPSULE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
