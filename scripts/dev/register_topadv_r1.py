"""Register the TOPADV-R1 experiment BEFORE launch.

Registry contract (EXECUTION_PLAN 8): experiments are registered pre-launch.
Arms/checkpoints are back-registered from artefacts after the round runs.
The control arms (fraction 1.0) are byte-identical to the CURR1 training
path: the mask is only applied when fraction < 1.0 (additive knob).
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
PLAN = REPO / "experiments/marathon/topadv_round_1_plan.yaml"


def main() -> int:
    import json

    registry = Registry(REGISTRY_ROOT)
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    hashes = capsule["source_identity"]["lineage_hashes"]
    seed_material = PLAN.read_text(encoding="utf-8")
    plan_digest = hashlib.sha256(seed_material.encode()).hexdigest()[:12]
    record = {
        "KIND": "experiment",
        "ID": canonical_id("experiment", "top-advantage-fractions-r1", plan_digest),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "NAME": "TOP-ADVANTAGE-FRACTIONS-R1",
        "PPO_SEMANTICS": "UNCHANGED",
        "SEEDS": [20260821, 20260823],
        "BUDGET_PER_ARM_UPDATES": 240,
        "BUDGET_PER_ARM_TRANSITIONS": 1966080,
        "ARMS": [
            "TOPADV-A0-CONTROL-S1",
            "TOPADV-A0-CONTROL-S2",
            "TOPADV-A1-TOP50-S1",
            "TOPADV-A1-TOP50-S2",
            "TOPADV-A2-TOP25-S1",
            "TOPADV-A2-TOP25-S2",
        ],
        "CONFIG_IDENTITY": {
            "compute_class": "GPU_RUNPOD_A40",
            "plan": "experiments/marathon/topadv_round_1_plan.yaml",
            "plan_kind": "MARATHON_SCREENING_ROUND_PLAN",
            "status": "PREDECLARED",
            "knob": "train/competition_native_jax/top_advantage_jax.py (additive; identity at 1.0)",
        },
        "LINEAGE": {
            "NAME": "competition_native_jax_v1",
            "IMPLEMENTATION_FINGERPRINT": hashes["learner_implementation_hash"][:16],
            "LINEAGE_HASHES": hashes,
        },
        "EVIDENCE_LINKS": ["EV-0013", "EV-0015", "EV-0029", "EV-0036", "EV-0038", "EV-0039"],
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
