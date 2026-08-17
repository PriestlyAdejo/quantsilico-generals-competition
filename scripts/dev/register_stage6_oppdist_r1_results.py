"""Register STAGE6_OPPDIST_R1 gameplay arbiter checkpoint/candidate/evaluation
records and the experiment verdict.

Two candidates x 18 games each (3 opponents x 3 seat-swapped pairs,
1200-turn horizon, protocol identical to every prior arbiter per
stage6_oppdist_r1_plan.yaml). Adjudication: predeclared branch (b) fires -
wins in 0-1 seeds -> programme-level escalation review (dossier recorded:
docs/marathon/PROGRAMME_ESCALATION_REVIEW_V1.yaml), informed by EV-0069
compatibility gate (recipe-scoped interpretation; RC-R1 bridge is the
predeclared high-value successor, already registered).
"""

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

RUN_ROOT = REPO / "experiments/marathon/paired_eval_runs/oppdist_arbiter"
ARM_ROOT = REPO / "experiments/marathon/screening_runs/STAGE6-OPPDIST-R1"

CANDIDATES = {
    "oppdist_s1": ("OPPDIST-B0-TEACHOPP-S1", "run#oppdist-b0-teachopp-s1#b956cd1a9bea",
                   20260915, "16_DRAW_2_LOSS_0_WIN_ZERO_FAULTS"),
    "oppdist_s2": ("OPPDIST-B0-TEACHOPP-S2", "run#oppdist-b0-teachopp-s2#20ac28eb3eb5",
                   20260917, "18_DRAW_0_LOSS_0_WIN_ZERO_FAULTS"),
}
EXPERIMENT_ID = "experiment#stage6-oppdist-r1#c0103074d017"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    for name, (arm, run_id, seed, outcome) in CANDIDATES.items():
        raw = ARM_ROOT / arm / "raw.npz"
        h = sha256_file(raw)[:12]
        ckpt_id = canonical_id("checkpoint", arm.lower() + "-terminal", h)
        if not reg.exists(ckpt_id):
            reg.add({
                "KIND": "checkpoint",
                "ID": ckpt_id,
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "NAME": arm + "-TERMINAL",
                "RUN_ID": run_id,
                "PPO_SEMANTICS": "UNCHANGED",
                "TRAINING_REGIME": "PERSISTENT_EPISODE_REGIME_V1",
                "TRANSITIONS": 8388608,
                "SEEDS": [seed],
                "LINEAGE": lineage,
                "ARTEFACT_HASHES": {"raw.npz": sha256_file(raw)},
                "ARTEFACT_LOCATIONS": [
                    f"experiments/marathon/screening_runs/STAGE6-OPPDIST-R1/{arm}/raw.npz (npz gitignored)"
                ],
                "OPPONENT_HASH_INVARIANT": True,
            })
            print("ADDED", ckpt_id)

        cand_material = hashlib.sha256(name.encode()).hexdigest()[:12]
        cand_id = canonical_id("candidate", name, cand_material)
        if not reg.exists(cand_id):
            reg.add({
                "KIND": "candidate",
                "ID": cand_id,
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "NAME": name,
                "CHECKPOINT_ID": ckpt_id,
                "PPO_SEMANTICS": "EVAL_ONLY",
                "SERVING": "canonical 8-plane JaxTransformerPolicy deterministic (EVAL_ONLY)",
                "SANITY_PROBE": "serving_sanity_probe_oppdist_r1.py PASS (L1 weights matter vs fresh + baseline, distinct 0.465; L2 deterministic)",
                "LINEAGE": lineage,
                "EVIDENCE_LINKS": ["EV-0034"],
            })
            print("ADDED", cand_id)

        summary = json.loads(
            (RUN_ROOT / name / name / "summary.json").read_text(encoding="utf-8")
        )
        eval_material = hashlib.sha256(
            f"{name}-oppdist-arbiter".encode()
        ).hexdigest()[:12]
        eval_id = canonical_id("evaluation", f"{name}-oppdist-arbiter", eval_material)
        cs_lower = summary["confidence_sequence"]["lower"]
        mean = summary["mean_difference"]
        worst = summary["matchup_metrics"]["WORST_MATCHUP_SCORE"]
        if not reg.exists(eval_id):
            reg.add({
                "KIND": "evaluation",
                "ID": eval_id,
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "NAME": f"{name}-oppdist-arbiter",
                "CANDIDATE_ID": cand_id,
                "PPO_SEMANTICS": "EVAL_ONLY",
                "EVALUATOR_IDENTITY": {
                    "confidence": 0.95,
                    "method": "ANYTIME_VALID_BOUNDED_MIXTURE_CONFIDENCE_SEQUENCE",
                    "tool": "scripts/evaluation/run_marathon_paired_eval.py",
                },
                "EVAL_PROTOCOL": {
                    "mode": "competition",
                    "namespace": "marathon-eval-v1",
                    "opponents": ["legal_random", "heuristic_v0", "heuristic_v1"],
                    "pairs_per_opponent": 3,
                    "practical_margin": 0.01,
                    "seat_swapped": True,
                },
                "CONFIG_IDENTITY": {
                    "max_turns": 1200,
                    "method": "ANYTIME_VALID_BOUNDED_MIXTURE_CONFIDENCE_SEQUENCE",
                    "namespace": "marathon-eval-v1",
                    "plan": "experiments/marathon/stage6_oppdist_r1_plan.yaml",
                    "weights": "raw",
                },
                "RESULT": (f"NO_PROMOTION {outcome} pairs=9 games=18 "
                           f"matchup_mean={mean:.3f} worst={worst:.3f} cs_lower={cs_lower:.4f}"),
                "RESULTS_LOCATION": f"experiments/marathon/paired_eval_runs/oppdist_arbiter/{name}/{name}/summary.json",
                "LINEAGE": lineage,
                "EVIDENCE_LINKS": ["EV-0034"],
                "RECORDED_AT_UTC": stamp,
            })
            print("ADDED", eval_id)

    exp_path = (
        REPO / "experiments/marathon/registry/records"
        / "experiment__stage6-oppdist-r1__c0103074d017.json"
    )
    exp = json.loads(exp_path.read_text(encoding="utf-8"))
    if "RESULT" not in exp:
        exp["RESULT"] = (
            "ADJUDICATED: NO_PROMOTION both seeds per predeclared rules (s1 16D+2L, s2 18D, "
            "0 wins, zero faults; CS lower -0.628/-0.573 vs practical margin 0.01). Predeclared "
            "branch (b) fires: programme-level escalation review recorded "
            "(docs/marathon/PROGRAMME_ESCALATION_REVIEW_V1.yaml), informed by EV-0069 "
            "compatibility gate (MATERIAL_GAPS_REMAIN) - recipe-scoped interpretation, NOT a "
            "method-class exhaustion claim. Successor RC_R1_BRIDGE "
            "(experiment#rc-r1-bridge#60729d9ef92f) executes next. Rollout diagnostic: "
            "decisive ticks re-emerged under frozen-teacher pressure but were almost all "
            "teacher-opponent wins; trained seat never won in-rollout."
        )
        exp["RECORDED_AT_UTC"] = stamp
        tmp = exp_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(exp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(exp_path)
        print("UPDATED", EXPERIMENT_ID)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
