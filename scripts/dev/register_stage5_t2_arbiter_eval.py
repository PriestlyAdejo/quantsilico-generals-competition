"""Register STAGE5 T2 K1 gameplay arbiter candidate + evaluation records (EV-0058).

Two candidates x 18 games each (3 opponents x 3 seat-swapped pairs, 1200-turn
horizon, RWB1-identical protocol per stage5_capacity_value_r1_plan.yaml
gameplay_arbiter clause). Adjudication: predeclared rules only - strength
requires statistically above matched control (EV-0054 RWB1-A0-CONTROL
results) AND above zero with the harness CS.
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

RUN_ROOT = REPO / "experiments/marathon/paired_eval_runs/t2_arbiter"

CHECKPOINTS = {
    "s5_t2_k1_s1": "checkpoint#s5-t2-k1-s1-terminal#577f90f241cc",
    "s5_t2_k1_s2": "checkpoint#s5-t2-k1-s2-terminal#41d7647f9023",
}

OUTCOMES = {
    "s5_t2_k1_s1": "18_DRAW_ZERO_FAULTS",
    "s5_t2_k1_s2": "18_DRAW_ZERO_FAULTS",
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
            "SERVING_ADAPTER": "baselines/s5_t2_serving_common.py (JaxTransformerHistoryPolicy k1, 16-plane)",
            "SANITY_PROBE": "scripts/analysis/serving_sanity_probe_t2.py PASS pre-game (EV-0034 precedent)",
            "EVIDENCE_LINKS": ["EV-0034", "EV-0049", "EV-0057"],
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
            "ID": canonical_id("evaluation", name + "-stage5-t2-arbiter", material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": name + "-STAGE5-T2-ARBITER-EVAL",
            "PPO_SEMANTICS": "EVAL_ONLY",
            "LINEAGE": lineage,
            "CONFIG_IDENTITY": {
                "plan": "experiments/marathon/stage5_capacity_value_r1_plan.yaml",
                "namespace": doc["eval_namespace"],
                "method": cs["method"],
                "weights": "raw",
                "max_turns": 1200,
            },
            "EVIDENCE_LINKS": ["EV-0034", "EV-0049", "EV-0054", "EV-0057"],
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
                f"cs_lower={cs['lower']:.4f} EV-0058"
            ),
            "ARTEFACT_LOCATIONS": [path.relative_to(REPO).as_posix()],
            "RECORDED_AT_UTC": stamp,
        }
        if not reg.exists(rec["ID"]):
            reg.add(rec)
            print("ADDED", rec["ID"])

    exp_path = (
        REPO / "experiments/marathon/registry/records"
        / "experiment__stage5-capacity-value-r1__f1dc01ccc86f.json"
    )
    doc = json.loads(exp_path.read_text(encoding="utf-8"))
    doc["RESULT"] = (
        "T2 K1 ADJUDICATED EV-0057/EV-0058: screening advanced both seeds "
        "telemetry-grade; gameplay arbiter FAILS predeclared strength test "
        "(0 wins in 36 T2 games, 36/36 draws, zero faults; matched control "
        "also 36/36 draws EV-0054). T2 LEGAL TEMPORAL HISTORY K=1 does NOT "
        "demonstrate win conversion. REJECT for promotion; preserved as "
        "regime-scoped science. NEXT per plan: later ladder axes (T3/T4) "
        "require their own predeclared round plans building on R1 outcomes."
    )
    exp_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("VERDICT recorded on experiment#stage5-capacity-value-r1#f1dc01ccc86f")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
