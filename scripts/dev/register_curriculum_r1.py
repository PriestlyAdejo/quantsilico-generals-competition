"""Register the SPAWN-DISTANCE-CURRICULUM-R1 experiment BEFORE launch.

Registry contract (EXECUTION_PLAN 8): experiments are registered pre-launch.
Arms/checkpoints are back-registered from artefacts after the round runs.
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
PLAN = REPO / "experiments/marathon/curriculum_round_1_plan.yaml"


def main() -> int:
    import json

    registry = Registry(REGISTRY_ROOT)
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    hashes = capsule["source_identity"]["lineage_hashes"]
    seed_material = PLAN.read_text(encoding="utf-8")
    record = {
        "KIND": "experiment",
        "ID": canonical_id(
            "experiment", "spawn-distance-curriculum-r1", hashlib.sha256(seed_material.encode()).hexdigest()[:12]
        ),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "NAME": "SPAWN-DISTANCE-CURRICULUM-R1",
        "PPO_SEMANTICS": "UNCHANGED",
        "SEEDS": [20260816, 20260818],
        "BUDGET_PER_ARM_UPDATES": 240,
        "BUDGET_PER_ARM_TRANSITIONS": 1966080,
        "ARMS": [
            "CURR1-A0-CONTROL-S1",
            "CURR1-A0-CONTROL-S2",
            "CURR1-A1-CLOSE-8-S1",
            "CURR1-A1-CLOSE-8-S2",
            "CURR1-A2-FAR-21-S1",
            "CURR1-A2-FAR-21-S2",
        ],
        "CONFIG_IDENTITY": {
            "compute_class": "GPU_RUNPOD_A40",
            "plan": "experiments/marathon/curriculum_round_1_plan.yaml",
            "plan_kind": "MARATHON_SCREENING_ROUND_PLAN",
            "status": "PREDECLARED_CONDITIONAL",
        },
        "LINEAGE": {
            "NAME": "competition_native_jax_v1",
            "IMPLEMENTATION_FINGERPRINT": hashes["learner_implementation_hash"][:16],
            "LINEAGE_HASHES": hashes,
        },
        "EVIDENCE_LINKS": ["EV-0017", "EV-0021", "EV-0029", "EV-0031", "EV-0032", "EV-0034"],
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
