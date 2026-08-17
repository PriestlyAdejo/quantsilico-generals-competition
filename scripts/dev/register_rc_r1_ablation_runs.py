"""Register RC_R1_ABLATION_R1 experiment + runs BEFORE launch (registry
contract). Predeclared successor of RC_R1_BRIDGE per its adjudication branch
(b) (EV-0072): ONE bounded ablation round isolating D1 (128-fragment) and D2
(competence curriculum) from the insufficient combination. Launch only after
OBS-V2-R1 adjudication or a dependency-safe slot.
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
PLAN = REPO / "experiments/marathon/rc_r1_ablation_r1_plan.yaml"
REGIME = "PERSISTENT_EPISODE_REGIME_V1"

ARMS = {
    # name: (seed, budget, rollout_len)
    "RC-AB1-FRAG-S1": (20260927, 8388608, 128),
    "RC-AB2-CURR-S1": (20260927, 8388608, 32),
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
    experiment_id = canonical_id("experiment", "rc-r1-ablation-r1", plan_digest)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not registry.exists(experiment_id):
        registry.add(
            {
                "KIND": "experiment",
                "ID": experiment_id,
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "NAME": "RC_R1_ABLATION_R1",
                "SEEDS": [20260927],
                "PPO_SEMANTICS": "UNCHANGED",
                "TRAINING_REGIME": REGIME,
                "LINEAGE": lineage,
                "CONFIG_IDENTITY": {
                    "plan": "experiments/marathon/rc_r1_ablation_r1_plan.yaml",
                    "plan_kind": "MARATHON_RC_ABLATION_PLAN",
                    "plan_sha256_12": plan_digest,
                    "status": "PREDECLARED",
                },
                "EVIDENCE_LINKS": ["EV-0072", "EV-0069", "EV-0061", "EV-0049"],
                "RECORDED_AT_UTC": stamp,
            }
        )
        print("ADDED", experiment_id)

    for name, (seed, budget, rollout_len) in ARMS.items():
        material = hashlib.sha256(name.encode()).hexdigest()[:12]
        updates = budget // (256 * rollout_len)
        if name == "RC-AB1-FRAG-S1":
            cmd = (
                "run_sh_r1_arm.py --arm-id "
                f"{name} --num-envs 256 --rollout-len 128 --accumulate-minibatches 8 "
                f"--budget-transitions {budget} --seed {seed} --episode-carry persistent "
                "--reward-shape none (init: warm-start MARATHON_BASELINE_V0 raw/ema/opt)"
            )
            curriculum = "none (default spawn-distance)"
            geometry = {"num_envs": 256, "rollout_len": 128}
        else:
            cmd = (
                "run_sh_r1_arm.py --arm-id "
                f"{name} --num-envs 256 --rollout-len 32 --budget-transitions {budget} "
                f"--seed {seed} --episode-carry persistent --reward-shape none "
                "--curriculum competence-spawn (init: warm-start MARATHON_BASELINE_V0 raw/ema/opt)"
            )
            curriculum = (
                "competence spawn-distance stages [8, 17]; advance at "
                "greedy-vs-legal_random >= 0.6 / 64 games every 32 updates"
            )
            geometry = {"num_envs": 256, "rollout_len": 32}
        record = {
            "KIND": "run",
            "ID": canonical_id("run", name.lower(), material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": name,
            "EXPERIMENT_ID": experiment_id,
            "PPO_SEMANTICS": "UNCHANGED",
            "TRAINING_REGIME": REGIME,
            "LINEAGE": lineage,
            "GEOMETRY": geometry,
            "SEEDS": [seed],
            "REWARD_SHAPE": "none",
            "EPISODE_CARRY": "persistent",
            "CURRICULUM": curriculum,
            "STATUS": "REGISTERED_PRE_LAUNCH",
            "RESULT": "REGISTERED_PRE_LAUNCH",
            "BUDGET": f"{budget} transitions / ~{updates} updates",
            "COMMAND": cmd,
            "ARTEFACT_LOCATIONS": [
                f"experiments/marathon/rc_r1_ablation/{name}/ (planned)"
            ],
            "ENVIRONMENT": {"jax": "A40 (predeclared bounded round after OBS-V2-R1)", "python": "3.12"},
            "STOP_REASON": "NOT_STARTED_PRE_LAUNCH_REGISTRATION",
            "EVIDENCE_LINKS": ["EV-0072", "EV-0069", "EV-0061", "EV-0049"],
            "RECORDED_AT_UTC": stamp,
        }
        if not registry.exists(record["ID"]):
            registry.add(record)
            print("ADDED", record["ID"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
