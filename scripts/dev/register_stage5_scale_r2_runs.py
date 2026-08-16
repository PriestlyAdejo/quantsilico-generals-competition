"""Register STAGE5_SCALE_R2 experiment + runs BEFORE launch (registry contract).

Predeclared plan: experiments/marathon/stage5_scale_r2_plan.yaml. Successor
to SCALE-R1 (EV-0061): replication of 8M win emergence (2 new seeds) + one
16M depth probe. Gameplay arbiter is the ONLY gate.
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
PLAN = REPO / "experiments/marathon/stage5_scale_r2_plan.yaml"
REGIME = "PERSISTENT_EPISODE_REGIME_V1"
GEOMETRY = {"num_envs": 256, "rollout_len": 32}

ARMS = {
    "SCALE-B0-8M-S3": (20260915, 8388608),
    "SCALE-B0-8M-S4": (20260917, 8388608),
    "SCALE-B1-16M-S1": (20260915, 16777216),
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
    experiment_id = canonical_id("experiment", "stage5-scale-r2", plan_digest)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not registry.exists(experiment_id):
        registry.add(
            {
                "KIND": "experiment",
                "ID": experiment_id,
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "NAME": "STAGE5_SCALE_R2",
                "SEEDS": [20260915, 20260917, 20260915],
                "PPO_SEMANTICS": "UNCHANGED",
                "TRAINING_REGIME": REGIME,
                "LINEAGE": lineage,
                "CONFIG_IDENTITY": {
                    "plan": "experiments/marathon/stage5_scale_r2_plan.yaml",
                    "plan_kind": "MARATHON_STAGE5_ROUND_PLAN",
                    "plan_sha256_12": plan_digest,
                    "status": "PREDECLARED",
                },
                "EVIDENCE_LINKS": ["EV-0054", "EV-0058", "EV-0060", "EV-0061"],
                "RECORDED_AT_UTC": stamp,
            }
        )
        print("ADDED", experiment_id)
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
                "(init: warm-started from MARATHON_BASELINE_V0)"
            ),
            "ARTEFACT_LOCATIONS": [
                f"experiments/marathon/screening_runs/STAGE5-SCALE-R2/{name}/ (planned)"
            ],
            "ENVIRONMENT": {
                "jax": "A40 (predeclared bounded round)",
                "python": "3.12",
            },
            "STOP_REASON": "NOT_STARTED_PRE_LAUNCH_REGISTRATION",
            "EVIDENCE_LINKS": ["EV-0049", "EV-0054", "EV-0061"],
            "RECORDED_AT_UTC": stamp,
        }
        if not registry.exists(record["ID"]):
            registry.add(record)
            print("ADDED", record["ID"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
