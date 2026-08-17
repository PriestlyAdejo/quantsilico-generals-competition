"""Register STAGE5 SCALE-R2 gameplay arbiter checkpoint/candidate/evaluation
records and the experiment verdict (EV-0063).

Three candidates x 18 games each (3 opponents x 3 seat-swapped pairs,
1200-turn horizon, RWB1-identical protocol per stage5_scale_r2_plan.yaml).
Adjudication: predeclared branch (b) fires - scale axis CLOSED after one
bounded successor attempt; all three arms winless with zero faults.
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

RUN_ROOT = REPO / "experiments/marathon/paired_eval_runs/scale_r2_arbiter"

CHECKPOINTS = {
    "scale_b0_8m_s3": ("SCALE-B0-8M-S3", 8388608),
    "scale_b0_8m_s4": ("SCALE-B0-8M-S4", 8388608),
    "scale_b1_16m_s1": ("SCALE-B1-16M-S1", 16777216),
}

OUTCOMES = {
    "scale_b0_8m_s3": "16_DRAW_2_LOSS_ZERO_FAULTS",
    "scale_b0_8m_s4": "17_DRAW_1_LOSS_ZERO_FAULTS",
    "scale_b1_16m_s1": "17_DRAW_1_LOSS_ZERO_FAULTS",
}


def sha256_file(path: Path) -> str:
    import hashlib

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

    ckpt_ids = {}
    for name, (arm, transitions) in CHECKPOINTS.items():
        raw = REPO / f"experiments/marathon/screening_runs/STAGE5-SCALE-R2/{arm}/raw.npz"
        h = sha256_file(raw)[:12]
        ckpt_id = canonical_id("checkpoint", arm.lower() + "-terminal", h)
        ckpt_ids[name] = ckpt_id
        run_material = __import__("hashlib").sha256(arm.encode()).hexdigest()[:12]
        if not reg.exists(ckpt_id):
            reg.add({
                "KIND": "checkpoint",
                "ID": ckpt_id,
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "NAME": arm + "-TERMINAL",
                "RUN_ID": canonical_id("run", arm.lower(), run_material),
                "PPO_SEMANTICS": "UNCHANGED",
                "TRAINING_REGIME": "PERSISTENT_EPISODE_REGIME_V1",
                "TRANSITIONS": transitions,
                "LINEAGE": lineage,
                "ARTEFACT_HASHES": {"raw.npz": sha256_file(raw)},
                "ARTEFACT_LOCATIONS": [
                    f"experiments/marathon/screening_runs/STAGE5-SCALE-R2/{arm}/raw.npz (npz gitignored)"
                ],
            })
            print("ADDED", ckpt_id)

    for name, (arm, _t) in CHECKPOINTS.items():
        candidate = {
            "KIND": "candidate",
            "ID": canonical_id("candidate", name, "v1"),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "CHECKPOINT_ID": ckpt_ids[name],
            "PPO_SEMANTICS": "EVAL_ONLY",
            "PARITY_PROOF": "tests/competition_native_jax/test_long_horizon_engine_parity.py",
            "PROTOCOL_AGENT": f"baselines/{name}/main.py",
            "SERVING_ADAPTER": "baselines/scale_r1_serving_common.py (canonical 8-plane JaxTransformerPolicy; CHECKPOINT_ROOTS resolves STAGE5-SCALE-R2)",
            "SANITY_PROBE": "scripts/analysis/serving_sanity_probe_scale_r2.py PASS pre-game (EV-0034 precedent)",
            "EVIDENCE_LINKS": ["EV-0034", "EV-0049", "EV-0061", "EV-0062", "EV-0063"],
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
            "ID": canonical_id("evaluation", name + "-stage5-scale-r2-arbiter", material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": name + "-STAGE5-SCALE-R2-ARBITER-EVAL",
            "PPO_SEMANTICS": "EVAL_ONLY",
            "LINEAGE": lineage,
            "CONFIG_IDENTITY": {
                "plan": "experiments/marathon/stage5_scale_r2_plan.yaml",
                "namespace": doc["eval_namespace"],
                "method": cs["method"],
                "weights": "raw",
                "max_turns": 1200,
            },
            "EVIDENCE_LINKS": ["EV-0034", "EV-0049", "EV-0054", "EV-0061", "EV-0062", "EV-0063"],
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
                f"cs_lower={cs['lower']:.4f} EV-0063"
            ),
            "ARTEFACT_LOCATIONS": [path.relative_to(REPO).as_posix()],
            "RECORDED_AT_UTC": stamp,
        }
        if not reg.exists(rec["ID"]):
            reg.add(rec)
            print("ADDED", rec["ID"])

    exp_path = (
        REPO / "experiments/marathon/registry/records"
        / "experiment__stage5-scale-r2__7a383bb25e4f.json"
    )
    doc = json.loads(exp_path.read_text(encoding="utf-8"))
    doc["RESULT"] = (
        "ADJUDICATED EV-0063: NO_PROMOTION all three arms per predeclared rules "
        "(cs lower -0.628/-0.601/-0.601 vs practical margin 0.01; zero faults; "
        "all games draw-or-loss). Predeclared branch (b) fires: R1's single win "
        "stands as rare noise; SCALE AXIS CLOSED after one bounded successor "
        "attempt. Aggregate scale evidence: 5 arms / 90 games = 1 WIN, 7 LOSSES, "
        "82 DRAWS. Programme advances to objective/value-structure (teacher-class) "
        "interventions per EXECUTION_PLAN Stage 5."
    )
    exp_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("VERDICT recorded on experiment#stage5-scale-r2#7a383bb25e4f")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
