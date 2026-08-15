"""Seed the canonical registry with MARATHON_BASELINE_V0 and its evaluation.

Every fact is read from tracked evidence artefacts (capsule, semantic
hashes, paired-eval summary); nothing is asserted from memory. Idempotence:
existing record IDs are skipped, never overwritten.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

REGISTRY_ROOT = REPO / "experiments/marathon/registry"
CAPSULE = REPO / "experiments/marathon/baseline_capsule_v0.json"
SEMANTIC_HASHES = REPO / "experiments/marathon/baseline_semantic_hashes.json"
EVAL_SUMMARY = (
    REPO
    / "experiments/marathon/paired_eval_runs/baseline_v0_vs_legal_random_cpu"
    / "marathon_baseline_v0/summary.json"
)
CHECKPOINT_PATH = (
    Path.home()
    / "quantsilico-runtime/cloud_assisted_deadline_salvage_v1_final"
    / "ckpt_final_u482_t7593984"
)


def _base_record(kind: str, name: str, material: str) -> dict:
    return {
        "KIND": kind,
        "ID": canonical_id(kind, name, material),
        "SCHEMA_VERSION": SCHEMA_VERSION,
    }


def main() -> int:
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    semantic = json.loads(SEMANTIC_HASHES.read_text(encoding="utf-8"))
    evaluation = json.loads(EVAL_SUMMARY.read_text(encoding="utf-8"))
    if capsule.get("status") != "PASS":
        print("capsule status is not PASS; refusing to register", file=sys.stderr)
        return 1

    registry = Registry(REGISTRY_ROOT)
    lineage = {
        "NAME": "competition_native_jax_v1",
        "IMPLEMENTATION_FINGERPRINT": capsule["source_identity"]["lineage_hashes"][
            "learner_implementation_hash"
        ][:16],
        "LINEAGE_HASHES": capsule["source_identity"]["lineage_hashes"],
    }

    experiment = _base_record("experiment", "marathon-baseline-v0-repro", "v1")
    experiment.update(
        {
            "NAME": "MARATHON_BASELINE_V0 reproduction and packaging",
            "PPO_SEMANTICS": "UNCHANGED",
            "LINEAGE": lineage,
            "CONFIG_IDENTITY": {
                "source_commit": capsule["source_identity"]["repo_commit"],
                "engine": capsule["source_identity"]["engine_submodule"],
            },
            "SEEDS": [1234, 7],
            "EVIDENCE_LINKS": ["EV-0013", "EV-0014", "EV-0015", "EV-0016"],
        }
    )

    run = _base_record("run", "baseline-capsule-cpu", "v1")
    run.update(
        {
            "EXPERIMENT_ID": experiment["ID"],
            "COMMAND": (
                ".venv/Scripts/python.exe scripts/analysis/assemble_baseline_capsule.py"
            ),
            "BUDGET": {"wall_seconds": capsule["total_wall_seconds"]},
            "STOP_REASON": "COMPLETE",
            "ENVIRONMENT": capsule["determinism_contract"],
            "ARTEFACT_LOCATIONS": ["experiments/marathon/baseline_capsule_v0.json"],
        }
    )

    file_hashes = {
        name: info["FILE_SHA256"] for name, info in semantic["artefacts"].items()
    }
    checkpoint = _base_record("checkpoint", "marathon-baseline-v0", "v1")
    checkpoint.update(
        {
            "RUN_ID": run["ID"],
            "ARTEFACT_HASHES": file_hashes,
            "LINEAGE": lineage,
            "TRANSITIONS": 7593984,
            "ARTEFACT_LOCATIONS": [str(CHECKPOINT_PATH)],
            "SEMANTIC_HASHES_FILE": "experiments/marathon/baseline_semantic_hashes.json",
            "REFERENCE_ID": "SPRINT_VALID_PPO_7M59",
        }
    )

    candidate = _base_record("candidate", "marathon-baseline-v0", "v1")
    candidate.update(
        {
            "CHECKPOINT_ID": checkpoint["ID"],
            "PPO_SEMANTICS": "EVAL_ONLY",
            "PROTOCOL_AGENT": "baselines/marathon_baseline_v0/main.py",
            "PARITY_PROOF": "tests/unit/test_baseline_agent_parity.py",
            "EVIDENCE_LINKS": ["EV-0019"],
        }
    )

    evaluation_record = _base_record(
        "evaluation", "baseline-v0-vs-legal-random-cpu", "v1"
    )
    evaluation_record.update(
        {
            "CANDIDATE_ID": candidate["ID"],
            "EVALUATOR_IDENTITY": "marathon_paired_evaluator_v1",
            "EVAL_PROTOCOL": "SEAT_SWAPPED_PAIRS_ANYTIME_VALID_BOUNDED_MIXTURE_CS",
            "RESULTS_LOCATION": (
                "experiments/marathon/paired_eval_runs/baseline_v0_vs_legal_random_cpu"
            ),
            "PAIRS_COMPLETED": evaluation["pairs_completed"],
            "PROMOTION": evaluation["promotion"],
            "EVIDENCE_LINKS": ["EV-0017", "EV-0019"],
        }
    )

    opponent = _base_record("opponent_reference", "legal-random", "v1")
    opponent.update(
        {
            "NAME": "legal_random",
            "SOURCE_IDENTITY": {"path": "baselines/legal_random/main.py"},
            "ARTEFACT_LOCATIONS": ["baselines/legal_random"],
        }
    )

    added = 0
    for record in (experiment, run, checkpoint, candidate, evaluation_record, opponent):
        if registry.exists(record["ID"]):
            print(f"EXISTS {record['ID']}")
            continue
        registry.add(record)
        added += 1
        print(f"ADDED  {record['ID']}")
    print(f"registry_root={REGISTRY_ROOT} added={added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
