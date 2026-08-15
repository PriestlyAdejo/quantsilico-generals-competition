"""Register the SH-R1 screening round as a canonical experiment record.

Run BEFORE any training launch: the registry contract requires experiments
to exist with declared PPO_SEMANTICS before training (EXECUTION_PLAN 8).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

REGISTRY_ROOT = REPO / "experiments/marathon/registry"
PLAN = REPO / "experiments/marathon/screening_round_1_plan.yaml"
CAPSULE = REPO / "experiments/marathon/baseline_capsule_v0.json"


def main() -> int:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    registry = Registry(REGISTRY_ROOT)

    lineage = {
        "NAME": "competition_native_jax_v1",
        "IMPLEMENTATION_FINGERPRINT": capsule["source_identity"]["lineage_hashes"][
            "learner_implementation_hash"
        ][:16],
        "LINEAGE_HASHES": capsule["source_identity"]["lineage_hashes"],
    }
    seeds = sorted(
        {seed for arm in plan["arms"] for seed in arm["seeds"]}
    )
    semantics = {arm["ppo_semantics"] for arm in plan["arms"]}
    if len(semantics) != 1:
        print(f"arms declare mixed semantics {semantics}; refusing", file=sys.stderr)
        return 1
    record = {
        "KIND": "experiment",
        "ID": canonical_id("experiment", plan["round_id"].lower(), "v1"),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "NAME": plan["round_id"],
        "PPO_SEMANTICS": semantics.pop(),
        "LINEAGE": lineage,
        "CONFIG_IDENTITY": {
            "plan": "experiments/marathon/screening_round_1_plan.yaml",
            "plan_kind": plan["kind"],
            "compute_class": plan["compute_class"],
        },
        "SEEDS": seeds,
        "EVIDENCE_LINKS": plan["evidence_links"],
        "ARMS": [arm["arm_id"] for arm in plan["arms"]],
        "BUDGET_PER_ARM_TRANSITIONS": plan["arms"][0]["budget_transitions"],
    }
    if registry.exists(record["ID"]):
        print(f"EXISTS {record['ID']}")
        return 0
    registry.add(record)
    print(f"ADDED  {record['ID']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
