"""Register STAGE5 SCALE-R1 gameplay arbiter candidate + evaluation records (EV-0061).

Two candidates x 18 games each (3 opponents x 3 seat-swapped pairs, 1200-turn
horizon, RWB1-identical protocol per stage5_scale_r1_plan.yaml). Adjudication:
predeclared rules only - NO_PROMOTION both seeds (harness CS decisive);
first genuine trained-candidate WIN recorded (S1 vs legal_random, zero faults).
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

RUN_ROOT = REPO / "experiments/marathon/paired_eval_runs/scale_arbiter"

CHECKPOINTS = {
    "scale_a0_8m_s1": "SCALE-A0-8M-S1",
    "scale_a0_8m_s2": "SCALE-A0-8M-S2",
}

OUTCOMES = {
    "scale_a0_8m_s1": "16_DRAW_1_WIN_vs_legal_random_1_LOSS_vs_heuristic_v0_ZERO_FAULTS",
    "scale_a0_8m_s2": "17_DRAW_1_LOSS_vs_heuristic_v1_ZERO_FAULTS",
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

    # Checkpoint records for the SCALE terminal checkpoints.
    ckpt_ids = {}
    for name, arm in CHECKPOINTS.items():
        raw = REPO / f"experiments/marathon/screening_runs/STAGE5-SCALE-R1/{arm}/raw.npz"
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
                "TRANSITIONS": 8388608,
                "LINEAGE": lineage,
                "ARTEFACT_HASHES": {"raw.npz": sha256_file(raw)},
                "ARTEFACT_LOCATIONS": [
                    f"experiments/marathon/screening_runs/STAGE5-SCALE-R1/{arm}/raw.npz (npz gitignored)"
                ],
            })
            print("ADDED", ckpt_id)

    for name in CHECKPOINTS:
        candidate = {
            "KIND": "candidate",
            "ID": canonical_id("candidate", name, "v1"),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "CHECKPOINT_ID": ckpt_ids[name],
            "PPO_SEMANTICS": "EVAL_ONLY",
            "PARITY_PROOF": "tests/competition_native_jax/test_long_horizon_engine_parity.py",
            "PROTOCOL_AGENT": f"baselines/{name}/main.py",
            "SERVING_ADAPTER": "baselines/scale_r1_serving_common.py (canonical 8-plane JaxTransformerPolicy)",
            "SANITY_PROBE": "scripts/analysis/serving_sanity_probe_scale_r1.py PASS pre-game (EV-0034 precedent)",
            "EVIDENCE_LINKS": ["EV-0034", "EV-0049", "EV-0061"],
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
            "ID": canonical_id("evaluation", name + "-stage5-scale-arbiter", material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": name + "-STAGE5-SCALE-ARBITER-EVAL",
            "PPO_SEMANTICS": "EVAL_ONLY",
            "LINEAGE": lineage,
            "CONFIG_IDENTITY": {
                "plan": "experiments/marathon/stage5_scale_r1_plan.yaml",
                "namespace": doc["eval_namespace"],
                "method": cs["method"],
                "weights": "raw",
                "max_turns": 1200,
            },
            "EVIDENCE_LINKS": ["EV-0034", "EV-0049", "EV-0054", "EV-0061"],
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
                f"cs_lower={cs['lower']:.4f} EV-0061"
            ),
            "ARTEFACT_LOCATIONS": [path.relative_to(REPO).as_posix()],
            "RECORDED_AT_UTC": stamp,
        }
        if not reg.exists(rec["ID"]):
            reg.add(rec)
            print("ADDED", rec["ID"])

    exp_path = (
        REPO / "experiments/marathon/registry/records"
        / "experiment__stage5-scale-r1__600fdb29e986.json"
    )
    doc = json.loads(exp_path.read_text(encoding="utf-8"))
    doc["RESULT"] = (
        "ADJUDICATED EV-0061: NO_PROMOTION both seeds per predeclared rules "
        "(harness CS lower -0.573/-0.601 vs practical margin 0.01; matched 2M "
        "control EV-0054 also winless). FIRST genuine trained-candidate WIN in "
        "any marathon arbiter (S1 vs legal_random, zero faults) with a paired "
        "loss; S2 17 draws + 1 loss. Draw wall cracked, not broken: scale to "
        "8M alone does not demonstrate win conversion; next axis predeclared "
        "before launch."
    )
    exp_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("VERDICT recorded on experiment#stage5-scale-r1#600fdb29e986")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
