"""Register RC_R1_BRIDGE experiment + runs BEFORE launch (registry contract).

Predeclared plan: experiments/marathon/rc_r1_bridge_plan.yaml.
KNOWN_WORKING_RECIPE_COMPATIBILITY_GATE_V1 verdict MATERIAL_GAPS_REMAIN ->
smallest corrected-regime bridge round: coherent known-working-style recipe
combination (fragment 128 + competence spawn-distance curriculum + topadv
0.25 + entropy/LR schedules) vs existing SCALE matched controls. Additive
amendment lane; does not touch OPPDIST-R1.
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
PLAN = REPO / "experiments/marathon/rc_r1_bridge_plan.yaml"
REGIME = "PERSISTENT_EPISODE_REGIME_V1"
GEOMETRY = {"num_envs": 256, "rollout_len": 128}

ARMS = {
    "RC-R1-BRIDGE-S1": (20260919, 8388608),
    "RC-R1-BRIDGE-S2": (20260921, 8388608),
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
    experiment_id = canonical_id("experiment", "rc-r1-bridge", plan_digest)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not registry.exists(experiment_id):
        registry.add(
            {
                "KIND": "experiment",
                "ID": experiment_id,
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "NAME": "RC_R1_BRIDGE",
                "SEEDS": [20260919, 20260921],
                "PPO_SEMANTICS": "UNCHANGED",
                "TRAINING_REGIME": REGIME,
                "LINEAGE": lineage,
                "CONFIG_IDENTITY": {
                    "plan": "experiments/marathon/rc_r1_bridge_plan.yaml",
                    "plan_kind": "MARATHON_RC_ROUND_PLAN",
                    "plan_sha256_12": plan_digest,
                    "status": "PREDECLARED",
                },
                "EVIDENCE_LINKS": [
                    "EV-0049", "EV-0061", "EV-0063", "EV-0068",
                    "KNOWN_WORKING_RECIPE_GAP_AUDIT_V1",
                    "KNOWN_WORKING_RECIPE_COMPATIBILITY_GATE_V1",
                ],
                "RECORDED_AT_UTC": stamp,
            }
        )
        print("ADDED", experiment_id)

    for name, (seed, budget) in ARMS.items():
        material = hashlib.sha256(name.encode()).hexdigest()[:12]
        updates = budget // (256 * 128)
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
            "TOP_ADVANTAGE_FRACTION": 0.25,
            "CURRICULUM": "competence spawn-distance stages [8, 17]; advance at greedy-vs-legal_random >= 0.6 / 64 games every 32 updates",
            "SCHEDULES": "ent_coef_t=max(0.05/(t+1)^0.2, 0.001); lr_t=clip(4.5e-3/(t+1)^1.1, 5e-6, 1e-4)",
            "STATUS": "REGISTERED_PRE_LAUNCH",
            "RESULT": "REGISTERED_PRE_LAUNCH",
            "BUDGET": f"{budget} transitions / ~{updates} updates",
            "COMMAND": (
                "run_sh_r1_arm.py --arm-id "
                f"{name} --num-envs 256 --rollout-len 128 --budget-transitions {budget} "
                f"--seed {seed} --episode-carry persistent --reward-shape none "
                "--top-advantage-fraction 0.25 --curriculum competence-spawn --schedules rc1 "
                "(init: warm-start MARATHON_BASELINE_V0 raw/ema/opt)"
            ),
            "ARTEFACT_LOCATIONS": [
                f"experiments/marathon/rc_r1_bridge/{name}/ (planned)"
            ],
            "ENVIRONMENT": {"jax": "A40 (predeclared bounded round)", "python": "3.12"},
            "STOP_REASON": "NOT_STARTED_PRE_LAUNCH_REGISTRATION",
            "EVIDENCE_LINKS": ["EV-0049", "EV-0061", "EV-0063",
                               "KNOWN_WORKING_RECIPE_COMPATIBILITY_GATE_V1"],
            "RECORDED_AT_UTC": stamp,
        }
        if not registry.exists(record["ID"]):
            registry.add(record)
            print("ADDED", record["ID"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
