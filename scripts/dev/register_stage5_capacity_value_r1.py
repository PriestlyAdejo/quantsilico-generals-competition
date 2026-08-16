"""Register STAGE5_CAPACITY_VALUE_R1 experiment BEFORE implementation/launch.

Triggered by the predeclared CROSS_REGIME_BRIDGE escalation rule (EV-0054:
draw wall persists after B1+B2 gameplay arbiter). EXECUTION_PLAN §11 isolation
ladder; R1 covers T0 (pre-existing RWB1 controls) + T1 (patch transformer).
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


def main() -> int:
    registry = Registry(REGISTRY_ROOT)
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    hashes = capsule["source_identity"]["lineage_hashes"]
    plan_digest = hashlib.sha256(PLAN.read_text(encoding="utf-8").encode()).hexdigest()[:12]
    record = {
        "KIND": "experiment",
        "ID": canonical_id("experiment", "stage5-capacity-value-r1", plan_digest),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "NAME": "STAGE5-CAPACITY-VALUE-R1",
        "PPO_SEMANTICS": "UNCHANGED",
        "SEEDS": [20260905, 20260907],
        "CONFIG_IDENTITY": {
            "stage": "STAGE_5",
            "plan": "experiments/marathon/stage5_capacity_value_r1_plan.yaml",
            "plan_id": "STAGE5_CAPACITY_VALUE_R1",
            "status": "PREDECLARED",
            "arms": ["T0_CONTROL_EXISTING (pre-satisfied by RWB1-A0-CONTROL-S1/S2)",
                     "T1_PATCH_TRANSFORMER (pre-satisfied: lineage already patch transformer)",
                     "T2_LEGAL_TEMPORAL_HISTORY (experimental arm)"],
            "training_regime": "PERSISTENT_EPISODE_REGIME_V1",
        },
        "LINEAGE": {
            "NAME": "competition_native_jax_v1",
            "IMPLEMENTATION_FINGERPRINT": hashes["learner_implementation_hash"][:16],
            "LINEAGE_HASHES": hashes,
        },
        "EVIDENCE_LINKS": ["EV-0049", "EV-0051", "EV-0053", "EV-0054"],
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
