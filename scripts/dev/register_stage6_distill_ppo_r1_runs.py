"""Register STAGE6_DISTILL_PPO_R1 experiment + runs BEFORE launch (registry contract).

Predeclared plan: experiments/marathon/stage6_distill_ppo_r1_plan.yaml.
Successor axis after TEACHER-R2 closure (EV-0065 branch (b)): expert
initialization continuation - distill the TEACHER-R2 teacher dataset into
the canonical transformer, then continue PPO on-policy. Consumes the BC
consumption gate's predeclared warm-start path.
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
PLAN = REPO / "experiments/marathon/stage6_distill_ppo_r1_plan.yaml"
REGIME = "PERSISTENT_EPISODE_REGIME_V1"
GEOMETRY = {"num_envs": 256, "rollout_len": 32}

ARMS = {
    "DISTILL-PPO-S1": (20260915, 8388608),
    "DISTILL-PPO-S2": (20260917, 8388608),
}


def main() -> int:
    registry = Registry(REGISTRY_ROOT)
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    hashes = capsule["source_identity"]["lineage_hashes"]
    lineage = {
        "NAME": "competition_native_jax_v1",
        "IMPLEMENTATION_FINGERPRINT": hashes["learner_implementation_hash"][:16],
        "LINEAGE_HASHES": hashes,
    }
    plan_digest = hashlib.sha256(PLAN.read_text(encoding="utf-8").encode()).hexdigest()[:12]
    experiment_id = canonical_id("experiment", "stage6-distill-ppo-r1", plan_digest)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not registry.exists(experiment_id):
        registry.add(
            {
                "KIND": "experiment",
                "ID": experiment_id,
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "NAME": "STAGE6_DISTILL_PPO_R1",
                "SEEDS": [20260915, 20260917],
                "PPO_SEMANTICS": "UNCHANGED",
                "TRAINING_REGIME": REGIME,
                "LINEAGE": lineage,
                "CONFIG_IDENTITY": {
                    "plan": "experiments/marathon/stage6_distill_ppo_r1_plan.yaml",
                    "plan_kind": "MARATHON_STAGE6_ROUND_PLAN",
                    "plan_sha256_12": plan_digest,
                    "status": "PREDECLARED",
                },
                "EVIDENCE_LINKS": ["EV-0034", "EV-0049", "EV-0053", "EV-0060",
                                    "EV-0061", "EV-0063", "EV-0065"],
                "RECORDED_AT_UTC": stamp,
            }
        )
        print("ADDED", experiment_id)

    d0_material = hashlib.sha256(b"DISTILL-S0-TRANSFORMER-BC").hexdigest()[:12]
    d0 = {
        "KIND": "run",
        "ID": canonical_id("run", "distill-s0-transformer-bc", d0_material),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "NAME": "DISTILL-S0-TRANSFORMER-BC",
        "EXPERIMENT_ID": experiment_id,
        "PPO_SEMANTICS": "OFF_POLICY_AUXILIARY",
        "TRAINING_REGIME": REGIME,
        "LINEAGE": lineage,
        "SEEDS": [20260915],
        "STATUS": "REGISTERED_PRE_LAUNCH",
        "RESULT": "REGISTERED_PRE_LAUNCH",
        "BUDGET": "40 epochs over TEACHER-R2 dataset (train n=14806), batch 64, adam 3e-4",
        "COMMAND": (
            "transformer BC distillation on TEACHER-R2 both-seat labels "
            "(transcript sha ed6344232c1e; splits identical to TEACHER-R2); "
            "screening gate EV-0060-identical; bounded A40 round, self-stop"
        ),
        "ARTEFACT_LOCATIONS": ["experiments/marathon/distill_ppo_r1/d0_distill/ (planned)"],
        "ENVIRONMENT": {"jax": "A40 (predeclared bounded round)", "python": "3.12"},
        "STOP_REASON": "NOT_STARTED_PRE_LAUNCH_REGISTRATION",
        "EVIDENCE_LINKS": ["EV-0060", "EV-0065"],
        "RECORDED_AT_UTC": stamp,
    }
    if not registry.exists(d0["ID"]):
        registry.add(d0)
        print("ADDED", d0["ID"])

    for name, (seed, budget) in ARMS.items():
        material = hashlib.sha256(name.encode()).hexdigest()[:12]
        updates = budget // (256 * 32)
        record = {
            "KIND": "run",
            "ID": canonical_id("run", name.lower(), material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": name,
            "EXPERIMENT_ID": experiment_id,
            "PPO_SEMANTICS": "UNCHANGED",
            "TRAINING_REGIME": REGIME,
            "LINEAGE": lineage,
            "GEOMETRY": GEOMETRY,
            "SEEDS": [seed],
            "REWARD_SHAPE": "none",
            "EPISODE_CARRY": "persistent",
            "STATUS": "REGISTERED_PRE_LAUNCH",
            "RESULT": "REGISTERED_PRE_LAUNCH",
            "BUDGET": f"{budget} transitions / ~{updates} updates",
            "COMMAND": (
                "run_sh_r1_arm.py --arm-id "
                f"{name} --num-envs 256 --rollout-len 32 --budget-transitions {budget} "
                f"--seed {seed} --episode-carry persistent --reward-shape none "
                "(init: warm-started from DISTILL-S0 transformer BC checkpoint)"
            ),
            "ARTEFACT_LOCATIONS": [
                f"experiments/marathon/distill_ppo_r1/d1_continuation/{name}/ (planned)"
            ],
            "ENVIRONMENT": {"jax": "A40 (predeclared bounded round)", "python": "3.12"},
            "STOP_REASON": "NOT_STARTED_PRE_LAUNCH_REGISTRATION",
            "EVIDENCE_LINKS": ["EV-0049", "EV-0061", "EV-0063", "EV-0065"],
            "RECORDED_AT_UTC": stamp,
        }
        if not registry.exists(record["ID"]):
            registry.add(record)
            print("ADDED", record["ID"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
