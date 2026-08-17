"""Register STAGE5_TEACHER_R1 experiment + runs BEFORE launch (registry contract).

Predeclared plan: experiments/marathon/stage5_teacher_r1_plan.yaml. Successor
axis after SCALE closure (EV-0063 branch (b)): teacher data-generation lane
using the pinned-engine HunterAgent (EV-0053), MPC/expert-lane invariant
honoured (data generation before any policy integration).
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
PLAN = REPO / "experiments/marathon/stage5_teacher_r1_plan.yaml"

RUNS = {
    "TEACHER-R1-STEP1-GENERATION": {
        "BUDGET": "20 pinned-engine games (HunterAgent vs heuristic_v0), 1200-turn cap",
        "COMMAND": (
            "pinned-engine match runner (src/generals_bot/evaluation/match.py): "
            "official_hunter vs heuristic_v0, map seeds 20260901..20260920, "
            "competition rules; hard gate: hunter >= 16/20 wins zero faults"
        ),
        "STOP_REASON": "NOT_STARTED_PRE_LAUNCH_REGISTRATION",
    },
    "TEACHER-R1-STEP2-LABELS": {
        "BUDGET": "teacher-side labels from STEP1 games passing the win-rate gate",
        "COMMAND": (
            "engine-verified action derivation (EV-0045 OWNERS_ONLY) + "
            "fog-of-war legal-POV gate (replay_legal_pov semantics) + "
            "silent-pass exclusion + provenance seal (pinned-engine SHA)"
        ),
        "STOP_REASON": "NOT_STARTED_PRE_LAUNCH_REGISTRATION",
    },
    "TEACHER-R1-STEP3-BC": {
        "BUDGET": "BC-A-FULL recipe (small CNN, CPU), single arm, seed 20260915",
        "COMMAND": (
            "scripts/training/bc_a_train_full.py family on STEP2 dataset "
            "(game-disjoint split 70/15/15); screening gate: held-out top-1 "
            "strictly above BOTH legal-uniform AND majority-pass on >= 1 split"
        ),
        "STOP_REASON": "NOT_STARTED_PRE_LAUNCH_REGISTRATION",
    },
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
    experiment_id = canonical_id("experiment", "stage5-teacher-r1", plan_digest)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not registry.exists(experiment_id):
        registry.add(
            {
                "KIND": "experiment",
                "ID": experiment_id,
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "NAME": "STAGE5_TEACHER_R1",
                "SEEDS": [20260915],
                "PPO_SEMANTICS": "UNCHANGED",
                "TRAINING_REGIME": "PERSISTENT_EPISODE_REGIME_V1",
                "LINEAGE": lineage,
                "CONFIG_IDENTITY": {
                    "plan": "experiments/marathon/stage5_teacher_r1_plan.yaml",
                    "plan_kind": "MARATHON_STAGE5_ROUND_PLAN",
                    "plan_sha256_12": plan_digest,
                    "status": "PREDECLARED",
                },
                "EVIDENCE_LINKS": ["EV-0034", "EV-0042", "EV-0045", "EV-0049",
                                    "EV-0053", "EV-0054", "EV-0060", "EV-0063"],
                "RECORDED_AT_UTC": stamp,
            }
        )
        print("ADDED", experiment_id)
    for name, spec in RUNS.items():
        material = hashlib.sha256(name.encode()).hexdigest()[:12]
        record = {
            "KIND": "run",
            "ID": canonical_id("run", name.lower(), material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": name,
            "EXPERIMENT_ID": experiment_id,
            "PPO_SEMANTICS": "UNCHANGED",
            "TRAINING_REGIME": "PERSISTENT_EPISODE_REGIME_V1",
            "LINEAGE": lineage,
            "SEEDS": [20260915],
            "STATUS": "REGISTERED_PRE_LAUNCH",
            "RESULT": "REGISTERED_PRE_LAUNCH",
            "BUDGET": spec["BUDGET"],
            "COMMAND": spec["COMMAND"],
            "ARTEFACT_LOCATIONS": [
                f"experiments/marathon/teacher_r1/{name.lower()}/ (planned)"
            ],
            "ENVIRONMENT": {
                "jax": "LOCAL_CPU_ONLY (zero RunPod spend predeclared)",
                "python": "3.12",
            },
            "STOP_REASON": spec["STOP_REASON"],
            "EVIDENCE_LINKS": ["EV-0042", "EV-0045", "EV-0053", "EV-0060", "EV-0063"],
            "RECORDED_AT_UTC": stamp,
        }
        if not registry.exists(record["ID"]):
            registry.add(record)
            print("ADDED", record["ID"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
