"""Register SH-R4 finalist promotion-eval records (EV-0033)."""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402


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
    finalists = [
        (
            "sh-r4-finalist-a0",
            36,
            "checkpoint#sh-r4-a0-control-b16-terminal#a3967229932f",
        ),
        (
            "sh-r4-finalist-a1",
            24,
            "checkpoint#sh-r4-a1-horizon-64-b16-terminal#cff66887671f",
        ),
    ]
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for name, _pairs, checkpoint_id in finalists:
        candidate = {
            "KIND": "candidate",
            "ID": canonical_id("candidate", name, "v1"),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "CHECKPOINT_ID": checkpoint_id,
            "PPO_SEMANTICS": "EVAL_ONLY",
            "PARITY_PROOF": "tests/unit/test_baseline_agent_parity.py",
            "PROTOCOL_AGENT": f"experiments/marathon/eval_candidates/{name}/main.py",
            "EVIDENCE_LINKS": ["EV-0019", "EV-0032"],
            "RECORDED_AT_UTC": stamp,
        }
        if not reg.exists(candidate["ID"]):
            reg.add(candidate)
            print("ADDED", candidate["ID"])
    for name, pairs in [(n, p) for n, p, _c in finalists]:
        candidate_id = canonical_id("candidate", name, "v1")
        path = REPO / f"experiments/marathon/paired_eval_runs/dev/{name}/summary.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        material = doc["finished_at_utc"].replace(":", "").replace("-", "")[:12]
        cs = doc["confidence_sequence"]
        rec = {
            "KIND": "evaluation",
            "ID": canonical_id("evaluation", name + "-promotion", material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": name + "-PROMOTION-EVAL",
            "PPO_SEMANTICS": "EVAL_ONLY",
            "LINEAGE": lineage,
            "CONFIG_IDENTITY": {
                "plan": "experiments/marathon/sh_r4_finalist_promotion_eval_plan.yaml",
                "namespace": doc["eval_namespace"],
                "method": cs["method"],
            },
            "EVIDENCE_LINKS": ["EV-0017", "EV-0021", "EV-0032", "EV-0033"],
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
                f"NO_PROMOTION pairs={pairs} mean_diff={doc['mean_difference']:.3f} "
                f"cs_lower={cs['lower']:.3f} worst_matchup=0.0 EV-0033"
            ),
            "ARTEFACT_LOCATIONS": [path.relative_to(REPO).as_posix()],
        }
        if not reg.exists(rec["ID"]):
            reg.add(rec)
            print("ADDED", rec["ID"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
