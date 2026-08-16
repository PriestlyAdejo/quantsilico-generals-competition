"""Register RWB1 LAND gameplay arbiter evaluation records (EV-0054).

Four candidates x 18 games each (3 opponents x 3 seat-swapped pairs, 1200-turn
horizon, identical map seeds per pair across candidates). Adjudication:
predeclared rwb1_land_gameplay_eval_plan.yaml only.
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

RUN_ROOT = REPO / "experiments/marathon/paired_eval_runs/rwb1_land_arbiter"

CHECKPOINTS = {
    "rwb1_control_s1": "checkpoint#rwb1-a0-control-s1-terminal#de73c1d79cc9",
    "rwb1_control_s2": "checkpoint#rwb1-a0-control-s2-terminal#8ae58f93ce10",
    "rwb1_land_s1": "checkpoint#rwb1-a1-land-s1-terminal#867678634986",
    "rwb1_land_s2": "checkpoint#rwb1-a1-land-s2-terminal#72a226982ee4",
}

OUTCOMES = {
    "rwb1_control_s1": "18_DRAW_ZERO_FAULTS",
    "rwb1_control_s2": "18_DRAW_ZERO_FAULTS",
    "rwb1_land_s1": "16_DRAW_2_LOSS_vs_heuristic_v1_ZERO_FAULTS",
    "rwb1_land_s2": "16_DRAW_2_LOSS_vs_heuristic_v1_ZERO_FAULTS",
}


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
    for name, checkpoint_id in CHECKPOINTS.items():
        candidate = {
            "KIND": "candidate",
            "ID": canonical_id("candidate", name, "v1"),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "CHECKPOINT_ID": checkpoint_id,
            "PPO_SEMANTICS": "EVAL_ONLY",
            "PARITY_PROOF": "tests/competition_native_jax/test_long_horizon_engine_parity.py",
            "PROTOCOL_AGENT": f"baselines/{name}/main.py",
            "EVIDENCE_LINKS": ["EV-0034", "EV-0049", "EV-0051"],
            "RECORDED_AT_UTC": stamp,
        }
        if not reg.exists(candidate["ID"]):
            reg.add(candidate)
            print("ADDED", candidate["ID"])
        path = RUN_ROOT / name / name / "summary.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        material = doc["finished_at_utc"].replace(":", "").replace("-", "")[:12]
        cs = doc["confidence_sequence"]
        rec = {
            "KIND": "evaluation",
            "ID": canonical_id("evaluation", name + "-rwb1-land-arbiter", material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": name + "-RWB1-LAND-ARBITER-EVAL",
            "PPO_SEMANTICS": "EVAL_ONLY",
            "LINEAGE": lineage,
            "CONFIG_IDENTITY": {
                "plan": "experiments/marathon/rwb1_land_gameplay_eval_plan.yaml",
                "namespace": doc["eval_namespace"],
                "method": cs["method"],
                "weights": "raw",
                "max_turns": 1200,
            },
            "EVIDENCE_LINKS": ["EV-0034", "EV-0049", "EV-0051", "EV-0053", "EV-0054"],
            "CANDIDATE_ID": candidate["ID"],
            "EVALUATOR_IDENTITY": {
                "tool": "scripts/evaluation/run_marathon_paired_eval.py",
                "method": cs["method"],
                "confidence": cs["confidence"],
            },
            "EVAL_PROTOCOL": {
                "pairs_per_opponent": 3,
                "opponents": ["legal_random", "heuristic_v0", "heuristic_v1"],
                "seat_swapped": True,
                "mode": "competition",
                "namespace": doc["eval_namespace"],
                "practical_margin": doc["promotion"]["practical_margin"],
            },
            "RESULTS_LOCATION": path.relative_to(REPO).as_posix(),
            "RESULT": (
                f"NO_PROMOTION {OUTCOMES[name]} pairs=9 games=18 "
                f"matchup_mean={doc['mean_difference']:.3f} "
                f"worst={doc['matchup_metrics']['WORST_MATCHUP_SCORE']:.3f} "
                f"cs_lower={cs['lower']:.4f} EV-0054"
            ),
            "ARTEFACT_LOCATIONS": [path.relative_to(REPO).as_posix()],
            "RECORDED_AT_UTC": stamp,
        }
        if not reg.exists(rec["ID"]):
            reg.add(rec)
            print("ADDED", rec["ID"])
    arbiter = REPO / "experiments/marathon/registry/records" / (
        "experiment__rwb1-land-gameplay-arbiter__a45f4ba5a2d6.json"
    )
    doc = json.loads(arbiter.read_text(encoding="utf-8"))
    doc["RESULT"] = (
        "ADJUDICATED EV-0054: LAND-POTENTIAL FAILS predeclared strength test "
        "(0 wins in 36 games; 2 losses each seed vs heuristic_v1; controls 36/36 "
        "draw; zero faults; identical map seeds per pair). Draw wall persists -> "
        "predeclared Stage 5 escalation triggered; B3/B4 reward-knob rounds VOID."
    )
    arbiter.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("VERDICT recorded on experiment#rwb1-land-gameplay-arbiter#a45f4ba5a2d6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
