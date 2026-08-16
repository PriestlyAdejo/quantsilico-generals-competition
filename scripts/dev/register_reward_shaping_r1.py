"""Register the REWARD-SHAPING-R1 experiment BEFORE launch.

Registry contract (EXECUTION_PLAN 8): experiments are registered pre-launch.
Arms/checkpoints are back-registered from artefacts after the round runs.
The control arms (mode "none") are byte-identical to the control training
path: shaping is only applied when mode != "none" and beta > 0 (additive).
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
PLAN = REPO / "experiments/marathon/reward_shaping_round_1_plan.yaml"


def main() -> int:
    import json

    registry = Registry(REGISTRY_ROOT)
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    hashes = capsule["source_identity"]["lineage_hashes"]
    seed_material = PLAN.read_text(encoding="utf-8")
    plan_digest = hashlib.sha256(seed_material.encode()).hexdigest()[:12]
    record = {
        "KIND": "experiment",
        "ID": canonical_id("experiment", "reward-shaping-r1", plan_digest),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "NAME": "REWARD-SHAPING-R1",
        "PPO_SEMANTICS": "UNCHANGED",
        "SEEDS": [20260825, 20260827],
        "BUDGET_PER_ARM_UPDATES": 240,
        "BUDGET_PER_ARM_TRANSITIONS": 1966080,
        "ARMS": [
            "RSH-A0-CONTROL-S1",
            "RSH-A0-CONTROL-S2",
            "RSH-A1-KILL-DELTA-S1",
            "RSH-A1-KILL-DELTA-S2",
            "RSH-A2-POTENTIAL-S1",
            "RSH-A2-POTENTIAL-S2",
        ],
        "CONFIG_IDENTITY": {
            "compute_class": "GPU_RUNPOD_A40",
            "plan": "experiments/marathon/reward_shaping_round_1_plan.yaml",
            "plan_kind": "MARATHON_SCREENING_ROUND_PLAN",
            "status": "PREDECLARED",
            "knob": (
                "train/competition_native_jax/reward_shaping_jax.py "
                "(additive; identity at mode none)"
            ),
        },
        "LINEAGE": {
            "NAME": "competition_native_jax_v1",
            "IMPLEMENTATION_FINGERPRINT": hashes["learner_implementation_hash"][:16],
            "LINEAGE_HASHES": hashes,
        },
        "EVIDENCE_LINKS": [
            "EV-0013", "EV-0015", "EV-0035", "EV-0036", "EV-0038", "EV-0039", "EV-0043",
        ],
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
