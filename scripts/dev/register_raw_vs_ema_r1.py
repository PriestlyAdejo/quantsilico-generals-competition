"""Register RAW-VS-EMA-R1 evaluation records (EV-0039)."""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

RUN_ROOT = REPO / "experiments/marathon/paired_eval_runs/raw_vs_ema_r1"


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
    name = "curr1-a0-control-s1-ema"
    candidate = {
        "KIND": "candidate",
        "ID": canonical_id("candidate", name, "v1"),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "CHECKPOINT_ID": "checkpoint#curr1-a0-control-s1-terminal#c7da72ebd72a",
        "PPO_SEMANTICS": "EVAL_ONLY",
        "PARITY_PROOF": "tests/unit/test_baseline_agent_parity.py",
        "PROTOCOL_AGENT": f"experiments/marathon/eval_candidates/{name}/main.py",
        "EVIDENCE_LINKS": ["EV-0019", "EV-0034", "EV-0038"],
        "RECORDED_AT_UTC": stamp,
    }
    if not reg.exists(candidate["ID"]):
        reg.add(candidate)
        print("ADDED", candidate["ID"])
    path = RUN_ROOT / name / "summary.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    material = doc["finished_at_utc"].replace(":", "").replace("-", "")[:12]
    cs = doc["confidence_sequence"]
    rec = {
        "KIND": "evaluation",
        "ID": canonical_id("evaluation", name + "-raw-vs-ema-r1", material),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "NAME": name + "-RAW-VS-EMA-R1-GAMEPLAY-EVAL",
        "PPO_SEMANTICS": "EVAL_ONLY",
        "LINEAGE": lineage,
        "CONFIG_IDENTITY": {
            "plan": "experiments/marathon/raw_vs_ema_r1_plan.yaml",
            "namespace": doc["eval_namespace"],
            "method": cs["method"],
        },
        "EVIDENCE_LINKS": ["EV-0017", "EV-0021", "EV-0034", "EV-0038", "EV-0039"],
        "CANDIDATE_ID": candidate["ID"],
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
            "UNRESOLVED_EMA_REMAINS_SERVING_DEFAULT pairs=24 ALL_48_GAMES_DRAW_AT_TRUNCATION "
            f"ZERO_FAULTS cs_lower={cs['lower']:.4f} EV-0039"
        ),
        "ARTEFACT_LOCATIONS": [path.relative_to(REPO).as_posix()],
        "RECORDED_AT_UTC": stamp,
    }
    if not reg.exists(rec["ID"]):
        reg.add(rec)
        print("ADDED", rec["ID"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
