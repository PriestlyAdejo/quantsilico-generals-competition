"""Register OBS_V2_R1 experiment + runs BEFORE launch (registry contract).

Predeclared plan: experiments/marathon/obs_v2_r1_plan.yaml. Highest-priority
learner successor per EV-0071 (OBSERVATION_GAP_CONFIRMED_MAJOR). Executes
after RC-R1 adjudication; RC-R1 arms are the matched controls.
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
PLAN = REPO / "experiments/marathon/obs_v2_r1_plan.yaml"
REGIME = "PERSISTENT_EPISODE_REGIME_V1"

ARMS = {
    "OBS-V2-R1-S1": 20260923,
    "OBS-V2-R1-S2": 20260925,
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
    experiment_id = canonical_id("experiment", "obs-v2-r1", plan_digest)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not registry.exists(experiment_id):
        registry.add(
            {
                "KIND": "experiment",
                "ID": experiment_id,
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "NAME": "OBS_V2_R1",
                "SEEDS": [20260923, 20260925],
                "PPO_SEMANTICS": "UNCHANGED",
                "TRAINING_REGIME": REGIME,
                "LINEAGE": lineage,
                "CONFIG_IDENTITY": {
                    "plan": "experiments/marathon/obs_v2_r1_plan.yaml",
                    "plan_kind": "MARATHON_OBS_ROUND_PLAN",
                    "plan_sha256_12": plan_digest,
                    "status": "PREDECLARED",
                },
                "EVIDENCE_LINKS": [
                    "EV-0071", "EV-0069",
                    "KNOWN_WORKING_RECIPE_GAP_AUDIT_V1",
                    "KNOWN_WORKING_RECIPE_COMPATIBILITY_GATE_V1",
                ],
                "RECORDED_AT_UTC": stamp,
            }
        )
        print("ADDED", experiment_id)

    for name, seed in ARMS.items():
        material = hashlib.sha256(name.encode()).hexdigest()[:12]
        record = {
            "KIND": "run",
            "ID": canonical_id("run", name.lower(), material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": name,
            "EXPERIMENT_ID": experiment_id,
            "PPO_SEMANTICS": "UNCHANGED",
            "TRAINING_REGIME": REGIME,
            "LINEAGE": lineage,
            "GEOMETRY": {"num_envs": 256, "rollout_len": 128},
            "SEEDS": [seed],
            "REWARD_SHAPE": "none",
            "EPISODE_CARRY": "persistent",
            "OBS_VERSION": "v2",
            "TOP_ADVANTAGE_FRACTION": 0.25,
            "CURRICULUM": "competence-spawn [8,17] (RC-R1 rule)",
            "SCHEDULES": "rc1",
            "STATUS": "REGISTERED_PRE_LAUNCH",
            "RESULT": "REGISTERED_PRE_LAUNCH",
            "BUDGET": "FROZEN 18612224 transitions (568 updates x 32768) = largest whole-update budget within 90-min cap at measured min RC-R1 TPS 3446.76 (est 5399.9s); 95-min wall cap backstop",
            "COMMAND": (
                "run_sh_r1_arm.py --arm-id "
                f"{name} --num-envs 256 --rollout-len 128 --seed {seed} "
                "--episode-carry persistent --reward-shape none "
                "--top-advantage-fraction 0.25 --curriculum competence-spawn "
                "--schedules rc1 --accumulate-minibatches 8 --obs-version v2 "
                "(init: warm-start MARATHON_BASELINE_V0 with declared shape surgery)"
            ),
            "ARTEFACT_LOCATIONS": [
                f"experiments/marathon/obs_v2_r1/{name}/ (planned)"
            ],
            "ENVIRONMENT": {"jax": "A40 (predeclared bounded round)", "python": "3.12"},
            "STOP_REASON": "NOT_STARTED_PRE_LAUNCH_REGISTRATION",
            "EVIDENCE_LINKS": ["EV-0071", "EV-0049"],
            "RECORDED_AT_UTC": stamp,
        }
        if not registry.exists(record["ID"]):
            registry.add(record)
            print("ADDED", record["ID"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
