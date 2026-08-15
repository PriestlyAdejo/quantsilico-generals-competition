"""Register SH-R4 finalist promotion-eval RERUN-V2 records (EV-0035).

The first execution's evaluation records remain in the registry as the
contaminated historical record (EV-0033 -> EV-0034 invalidation). These
records capture the clean re-execution after the EVAL_ONLY serving repair.
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

RUN_ROOT = REPO / "experiments/marathon/paired_eval_runs/sh_r4_finalist_rerun_v2"


def main() -> int:
    reg = Registry(REPO / "experiments/marathon/registry")
    capsule = json.loads(
        (REPO / "experiments/marathon/baseline_capsule_v0.json").read_text(encoding="utf-8")
    )
    hashes = capsule["source_identity"]["lineage_hashes"]
    lineage = {
        "NAME": "competition_native_jax_v1",
        "IMPLEMENTATION_FINGERPRINT": hashes["learner_implementation_hash"][:16],
        "LINEAGE_HASHES": hashes,
    }
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for name, pairs in [("sh-r4-finalist-a0", 36), ("sh-r4-finalist-a1", 24)]:
        candidate_id = canonical_id("candidate", name, "v1")
        path = RUN_ROOT / name / "summary.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        material = doc["finished_at_utc"].replace(":", "").replace("-", "")[:12]
        cs = doc["confidence_sequence"]
        rec = {
            "KIND": "evaluation",
            "ID": canonical_id("evaluation", name + "-promotion-rerun-v2", material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": name + "-PROMOTION-EVAL-RERUN-V2",
            "PPO_SEMANTICS": "EVAL_ONLY",
            "LINEAGE": lineage,
            "CONFIG_IDENTITY": {
                "plan": "experiments/marathon/sh_r4_finalist_promotion_eval_plan.yaml",
                "namespace": doc["eval_namespace"],
                "method": cs["method"],
                "supersedes_contaminated_run": "paired_eval_runs/dev",
            },
            "EVIDENCE_LINKS": ["EV-0017", "EV-0021", "EV-0032", "EV-0034", "EV-0035"],
            "CANDIDATE_ID": candidate_id,
            "EVALUATOR_IDENTITY": {
                "tool": "scripts/evaluation/run_marathon_paired_eval.py",
                "method": cs["method"],
                "confidence": cs["confidence"],
            },
            "EVAL_PROTOCOL": {
                "pairs_per_opponent": 12,
                "seat_swapped": True,
                "mode": "competition",
                "namespace": doc["eval_namespace"],
                "practical_margin": doc["promotion"]["practical_margin"],
            },
            "RESULTS_LOCATION": path.relative_to(REPO).as_posix(),
            "RESULT": (
                f"NO_PROMOTION pairs={pairs} ALL_GAMES_DRAW_AT_TRUNCATION_ZERO_FAULTS "
                f"mean_diff={doc['mean_difference']:.3f} cs_lower={cs['lower']:.4f} "
                f"worst_matchup=0.5 EV-0035"
            ),
            "ARTEFACT_LOCATIONS": [path.relative_to(REPO).as_posix()],
            "RECORDED_AT_UTC": stamp,
        }
        if not reg.exists(rec["ID"]):
            reg.add(rec)
            print("ADDED", rec["ID"])
        else:
            print("ALREADY_REGISTERED", rec["ID"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
