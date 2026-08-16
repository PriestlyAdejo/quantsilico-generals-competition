"""Register the BC-A-PILOT-R1 replay sub-experiment BEFORE launch.

Registry contract (EXECUTION_PLAN 8): experiments are registered pre-launch.
Pilot scope only (charter §7A first step); the full BC warm-start experiment
with PPO continuation is predeclared separately later.
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
PLAN = REPO / "experiments/marathon/bc_a_pilot_round_1_plan.yaml"


def main() -> int:
    import json

    registry = Registry(REGISTRY_ROOT)
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    hashes = capsule["source_identity"]["lineage_hashes"]
    seed_material = PLAN.read_text(encoding="utf-8")
    plan_digest = hashlib.sha256(seed_material.encode()).hexdigest()[:12]
    record = {
        "KIND": "experiment",
        "ID": canonical_id("experiment", "bc-a-pilot-r1", plan_digest),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "NAME": "BC-A-PILOT-R1",
        "PPO_SEMANTICS": "OFF_POLICY_AUXILIARY",
        "SEEDS": [20260831],
        "CONFIG_IDENTITY": {
            "compute_class": "CPU_LAPTOP",
            "plan": "experiments/marathon/bc_a_pilot_round_1_plan.yaml",
            "plan_kind": "MARATHON_REPLAY_SUBEXPERIMENT_PLAN",
            "status": "PREDECLARED",
            "source_dataset": "DATASET-ELITE-2026-08-15-V01",
            "engine_sha": "9e3b9d13cca51caa1bb07db48bb85c9e90ce0462",
        },
        "LINEAGE": {
            "NAME": "competition_native_jax_v1",
            "IMPLEMENTATION_FINGERPRINT": hashes["learner_implementation_hash"][:16],
            "LINEAGE_HASHES": hashes,
        },
        "EVIDENCE_LINKS": ["EV-0037", "EV-0042", "EV-0045"],
        "RECORDED_AT_UTC": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if registry.exists(record["ID"]):
        print(f"ALREADY_REGISTERED {record['ID']}")
        return 0
    registry.add(record)
    print(f"ADDED {record['ID']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
