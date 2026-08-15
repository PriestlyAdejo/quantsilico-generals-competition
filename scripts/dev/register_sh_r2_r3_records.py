"""Register SH-R3 experiment pre-launch and SH-R2/CPU-control run records.

Run BEFORE any SH-R3 training launch (EXECUTION_PLAN 8 registry contract).
Also back-registers the completed SH-R2 GPU arms, the SH-R1 CPU control,
and their terminal checkpoints from local evidence (EV-0028).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

REGISTRY_ROOT = REPO / "experiments/marathon/registry"
CAPSULE = REPO / "experiments/marathon/baseline_capsule_v0.json"
R3_PLAN = REPO / "experiments/marathon/screening_round_3_plan.yaml"
R2_ROOT = REPO / "experiments/marathon/screening_round_2_runs"
CPU_ROOT = REPO / "experiments/marathon/screening_runs/SH-R1-A0-CONTROL"
R2_EXPERIMENT_ID = "experiment#sh-r2-updates-matched#653b291f0ef0"
R1_EXPERIMENT_ID = "experiment#sh-r1-cpu-pilot#796c4d71603c"
POD_ID = "ba86a74lq28t3f"
SOURCE_COMMIT = "6753c57"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lineage_of(capsule: dict) -> dict:
    hashes = capsule["source_identity"]["lineage_hashes"]
    return {
        "NAME": "competition_native_jax_v1",
        "IMPLEMENTATION_FINGERPRINT": hashes["learner_implementation_hash"][:16],
        "LINEAGE_HASHES": hashes,
    }


def add_run_and_checkpoint(
    registry: Registry,
    *,
    name: str,
    experiment_id: str,
    arm_dir: Path,
    geometry: dict,
    hardware: dict,
    command: str,
    lineage: dict,
    added: list[str],
) -> None:
    summary = json.loads((arm_dir / "summary.json").read_text(encoding="utf-8"))
    material = hashlib.sha256(name.encode()).hexdigest()[:12]
    run_record = {
        "KIND": "run",
        "ID": canonical_id("run", name.lower(), material),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "NAME": name,
        "EXPERIMENT_ID": experiment_id,
        "PPO_SEMANTICS": "UNCHANGED",
        "LINEAGE": lineage,
        "GEOMETRY": geometry,
        "SEEDS": [summary["seed"]],
        "BUDGET": f"{summary['budget_transitions']} transitions / {summary['updates']} updates",
        "ACTUAL_TRANSITIONS": summary["actual_transitions"],
        "COMMAND": command,
        "STOP_REASON": summary["stop_reason"],
        "ENVIRONMENT": {"python": "3.12.14", "jax": "0.11.0+cuda12"},
        "HARDWARE": hardware,
        "SOURCE_COMMIT": SOURCE_COMMIT,
        "ARTEFACT_LOCATIONS": [
            str((arm_dir / "summary.json").as_posix()),
            str((arm_dir / "telemetry.jsonl").as_posix()),
        ],
        "ARTEFACT_HASHES": {
            "summary.json": sha256_file(arm_dir / "summary.json"),
            "telemetry.jsonl": sha256_file(arm_dir / "telemetry.jsonl"),
        },
        "RESULT": (
            "SURVIVED " if not summary["eliminated"] else "ELIMINATED "
        )
        + f"VLOSS_REDUCTION_{summary['metrics']['VLOSS_REDUCTION_OVER_ROUND']:.5f} "
        + f"VALID_SHARE_{summary['metrics']['VALID_LEARNING_SHARE']:.3f} EV-0028",
    }
    if not registry.exists(run_record["ID"]):
        registry.add(run_record)
        added.append(run_record["ID"])

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
            added.append(ckpt["ID"])


def main() -> int:
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    lineage = lineage_of(capsule)
    registry = Registry(REGISTRY_ROOT)
    added: list[str] = []

    plan = yaml.safe_load(R3_PLAN.read_text(encoding="utf-8"))
    semantics = {arm["ppo_semantics"] for arm in plan["arms"]}
    if len(semantics) != 1:
        print(f"mixed semantics {semantics}; refusing", file=sys.stderr)
        return 1
    r3 = {
        "KIND": "experiment",
        "ID": canonical_id("experiment", plan["round_id"].lower(), "v1"),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "NAME": plan["round_id"],
        "PPO_SEMANTICS": semantics.pop(),
        "LINEAGE": lineage,
        "CONFIG_IDENTITY": {
            "plan": "experiments/marathon/screening_round_3_plan.yaml",
            "plan_kind": plan["kind"],
            "compute_class": plan["compute_class"],
        },
        "SEEDS": sorted({s for arm in plan["arms"] for s in arm["seeds"]}),
        "EVIDENCE_LINKS": plan["evidence_links"],
        "ARMS": [arm["arm_id"] for arm in plan["arms"]],
        "BUDGET_PER_ARM_UPDATES": plan["arms"][0]["budget_updates"],
        "BUDGET_PER_ARM_TRANSITIONS": plan["arms"][0]["budget_transitions"],
    }
    if not registry.exists(r3["ID"]):
        registry.add(r3)
        added.append(r3["ID"])

    gpu_hardware = {
        "provider": "runpod",
        "pod_id": POD_ID,
        "gpu": "NVIDIA A40",
        "rate_usd_per_hr": 0.44,
        "cloud_type": "SECURE",
        "data_center": "EU-SE-1",
    }
    for arm in ["SH-R2-A0-CONTROL", "SH-R2-A1-HORIZON-64", "SH-R2-A2-HORIZON-128"]:
        geometry = json.loads(
            (R2_ROOT / arm / "summary.json").read_text(encoding="utf-8")
        )["geometry"]
        add_run_and_checkpoint(
            registry,
            name=arm + "-R2",
            experiment_id=R2_EXPERIMENT_ID,
            arm_dir=R2_ROOT / arm,
            geometry=geometry,
            hardware=gpu_hardware,
            command=(
                "bash remote_sh_r2_orchestrator.sh -> run_sh_r1_arm.py --arm-id "
                f"{arm} --budget-transitions 491520"
            ),
            lineage=lineage,
            added=added,
        )

    add_run_and_checkpoint(
        registry,
        name="SH-R1-A0-CONTROL-CPU",
        experiment_id=R1_EXPERIMENT_ID,
        arm_dir=CPU_ROOT,
        geometry={"num_envs": 8, "rollout_len": 16},
        hardware={"provider": "local", "cpu": "laptop CPU (JAX CPU backend)"},
        command="run_sh_r1_arm.py --arm-id SH-R1-A0-CONTROL --num-envs 8 --rollout-len 16",
        lineage=lineage,
        added=added,
    )

    for record_id in added:
        print(f"ADDED  {record_id}")
    print(f"TOTAL_NEW={len(added)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
