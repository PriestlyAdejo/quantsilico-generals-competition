"""Register STAGE5_TEACHER_R2 experiment + runs BEFORE launch (registry contract).

Predeclared plan: experiments/marathon/stage5_teacher_r2_plan.yaml. TEACHER-R1
successor (EV-0064 fallback): hunter-vs-hunter SELF-PLAY trajectory lane - a
teacher distribution that exercises win-conversion against an equally
win-converting opponent. MPC/expert-lane invariant honoured (data-generation
lane only, no policy integration).
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
PLAN = REPO / "experiments/marathon/stage5_teacher_r2_plan.yaml"

RUNS = {
    "TEACHER-R2-STEP1-SELFPLAY": {
        "ARTEFACT": "experiments/marathon/teacher_r2/step1_selfplay/ (planned)",
        "BUDGET": "20 pinned-engine self-play games (HunterAgent vs HunterAgent), 1200-turn cap",
        "COMMAND": (
            "scripts/data/teacher_r2_selfplay_generate.py (reuses teacher_r1 "
            "play_recorded + verify_replay): official_hunter vs official_hunter, "
            "map seeds 20260901..20260920, competition rules; hard gate: "
            ">=16/20 DECISIVE games zero engine faults all replays verified"
        ),
        "STOP_REASON": "NOT_STARTED_PRE_LAUNCH_REGISTRATION",
    },
    "TEACHER-R2-STEP2-LABELS": {
        "ARTEFACT": "experiments/marathon/teacher_r2/step2_labels/ (planned)",
        "BUDGET": "both-seat labels from decisive STEP1 games only",
        "COMMAND": (
            "canonical legal observation path (observe_one_jax + per-seat fog "
            "memory) replayed through the pinned engine; silent-pass labels "
            "excluded (outside legal_mask_one_jax); winner + loser seats both "
            "kept; provenance seal (pinned-engine SHA)"
        ),
        "STOP_REASON": "NOT_STARTED_PRE_LAUNCH_REGISTRATION",
    },
    "TEACHER-R2-STEP3-BC": {
        "ARTEFACT": "experiments/marathon/teacher_r2/step3_bc/ (planned)",
        "BUDGET": "BC-A recipe (small CNN, 40 epochs, adam 3e-3, batch 64, CPU), single arm, seed 20260915",
        "COMMAND": (
            "scripts/training teacher_r2 variant of teacher_r1_train.py on STEP2 "
            "dataset (game-disjoint split 14/3/3, same seed assignment as R1); "
            "screening gate: held-out top-1 strictly above BOTH legal-uniform AND "
            "majority-pass on >= 1 split (EV-0060-identical)"
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
    experiment_id = canonical_id("experiment", "stage5-teacher-r2", plan_digest)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not registry.exists(experiment_id):
        registry.add(
            {
                "KIND": "experiment",
                "ID": experiment_id,
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "NAME": "STAGE5_TEACHER_R2",
                "SEEDS": [20260915],
                "PPO_SEMANTICS": "UNCHANGED",
                "TRAINING_REGIME": "PERSISTENT_EPISODE_REGIME_V1",
                "LINEAGE": lineage,
                "CONFIG_IDENTITY": {
                    "plan": "experiments/marathon/stage5_teacher_r2_plan.yaml",
                    "plan_kind": "MARATHON_STAGE5_ROUND_PLAN",
                    "plan_sha256_12": plan_digest,
                    "status": "PREDECLARED",
                },
                "EVIDENCE_LINKS": ["EV-0034", "EV-0042", "EV-0045", "EV-0049",
                                    "EV-0053", "EV-0054", "EV-0060", "EV-0063", "EV-0064"],
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
            "ARTEFACT_LOCATIONS": [spec["ARTEFACT"]],
            "ENVIRONMENT": {
                "jax": "LOCAL_CPU_ONLY (zero RunPod spend predeclared)",
                "python": "3.12",
            },
            "STOP_REASON": spec["STOP_REASON"],
            "EVIDENCE_LINKS": ["EV-0042", "EV-0045", "EV-0053", "EV-0060", "EV-0064"],
            "RECORDED_AT_UTC": stamp,
        }
        if not registry.exists(record["ID"]):
            registry.add(record)
            print("ADDED", record["ID"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
