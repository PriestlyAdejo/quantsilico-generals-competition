"""Register STAGE5 R1 T2 screening RUNS BEFORE launch (registry contract).

Two T2-K1 arms (seeds 20260905/20260907) + T0 reference (pre-satisfied by the
RWB1-A0-CONTROL runs, recorded as a pointer, no recompute). Run/checkpoint
records with verdicts are added AFTER completion by a follow-up script; this
file registers the pre-launch run identities only.
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
PLAN = REPO / "experiments/marathon/stage5_capacity_value_r1_plan.yaml"
REGIME = "PERSISTENT_EPISODE_REGIME_V1"
GEOMETRY = {"num_envs": 256, "rollout_len": 32}
BUDGET = 1966080  # 240 updates x 256 x 32, matched to prior screening rounds

ARMS = {
    "S5-T2-K1-S1": 20260905,
    "S5-T2-K1-S2": 20260907,
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
    experiment_id = canonical_id("experiment", "stage5-capacity-value-r1", plan_digest)
    if not registry.exists(experiment_id):
        print(f"ERROR: experiment record missing: {experiment_id}")
        return 1
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
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
            "GEOMETRY": GEOMETRY,
            "SEEDS": [seed],
            "TEMPORAL_HISTORY": "k1",
            "REWARD_SHAPE": "none",
            "EPISODE_CARRY": "persistent",
            "STATUS": "REGISTERED_PRE_LAUNCH",
            "BUDGET": f"{BUDGET} transitions / 240 updates",
            "COMMAND": (
                "run_sh_r1_arm.py --arm-id "
                f"{name} --num-envs 256 --rollout-len 32 --budget-transitions {BUDGET} "
                f"--seed {seed} --temporal-history k1 --episode-carry persistent "
                "--reward-shape none (init: shared layers warm-started from "
                "MARATHON_BASELINE_V0, patch_proj shape-forced fresh per plan)"
            ),
            "STOP_REASON": "NOT_STARTED_PRE_LAUNCH_REGISTRATION",
            "ENVIRONMENT": {
                "python": "3.12.10",
                "jax": "cpu (local) or A40 (predeclared fallback)",
            },
            "ARTEFACT_LOCATIONS": [f"experiments/marathon/screening_runs/{name}/ (planned)"],
            "RESULT": "REGISTERED_PRE_LAUNCH",
            "EVIDENCE_LINKS": ["EV-0049", "EV-0054"],
            "RECORDED_AT_UTC": stamp,
        }
        if registry.exists(record["ID"]):
            print(f"ALREADY_REGISTERED {record['ID']}")
            continue
        registry.add(record)
        print(f"ADDED {record['ID']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
