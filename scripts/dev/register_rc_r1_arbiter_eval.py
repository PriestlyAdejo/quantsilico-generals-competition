"""Register RC_R1_BRIDGE gameplay arbiter checkpoint/candidate/evaluation
records and the experiment verdict (EV-0072).

Two candidates x 18 games each (3 opponents x 3 seat-swapped pairs,
1200-turn horizon, fixed arbiter protocol per rc_r1_bridge_plan.yaml).
Adjudication: predeclared branch (b) fires - wins in 0 seeds -> combination
insufficient; successor predeclares ONE bounded ablation round (fragment-only
and curriculum-only arms) before any further interpretation.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

RUN_ROOT = REPO / "experiments/marathon/paired_eval_runs/rc1_arbiter"
ARTEFACT_ROOT = REPO / "experiments/marathon/rc_r1_bridge"

CHECKPOINTS = {
    "rc1_s1": ("RC-R1-BRIDGE-S1", "run#rc-r1-bridge-s1#0374c1939762", 8388608),
    "rc1_s2": ("RC-R1-BRIDGE-S2", "run#rc-r1-bridge-s2#6c705ce0a7bf", 8388608),
}

OUTCOMES = {
    "rc1_s1": "16_DRAW_2_LOSS_ZERO_FAULTS",
    "rc1_s2": "17_DRAW_1_LOSS_ZERO_FAULTS",
}


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

    ckpt_ids = {}
    for name, (arm, run_id, transitions) in CHECKPOINTS.items():
        raw = ARTEFACT_ROOT / arm / "raw.npz"
        h = sha256_file(raw)[:12]
        ckpt_id = canonical_id("checkpoint", arm.lower() + "-terminal", h)
        ckpt_ids[name] = ckpt_id
        if not reg.exists(ckpt_id):
            reg.add({
                "KIND": "checkpoint",
                "ID": ckpt_id,
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "NAME": arm + "-TERMINAL",
                "RUN_ID": run_id,
                "PPO_SEMANTICS": "UNCHANGED",
                "TRAINING_REGIME": "PERSISTENT_EPISODE_REGIME_V1",
                "TRANSITIONS": transitions,
                "LINEAGE": lineage,
                "ARTEFACT_HASHES": {"raw.npz": sha256_file(raw)},
                "ARTEFACT_LOCATIONS": [
                    f"experiments/marathon/rc_r1_bridge/{arm}/raw.npz (npz gitignored)"
                ],
            })
            print("ADDED", ckpt_id)

    for name, (arm, run_id, _t) in CHECKPOINTS.items():
        candidate = {
            "KIND": "candidate",
            "ID": canonical_id("candidate", name, "v1"),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "CHECKPOINT_ID": ckpt_ids[name],
            "PPO_SEMANTICS": "EVAL_ONLY",
            "PARITY_PROOF": "tests/competition_native_jax/test_long_horizon_engine_parity.py",
            "PROTOCOL_AGENT": f"baselines/{name}/main.py",
            "SERVING_ADAPTER": "baselines/rc1_serving_common.py (canonical 8-plane JaxTransformerPolicy; all four frozen RC-R1 deltas are training-time only)",
            "SANITY_PROBE": "scripts/analysis/serving_sanity_probe_rc_r1.py PASS pre-game (EV-0034 precedent)",
            "EVIDENCE_LINKS": ["EV-0034", "EV-0049", "EV-0069", "EV-0072"],
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
            "ID": canonical_id("evaluation", name + "-rc-r1-bridge-arbiter", material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": name + "-RC-R1-BRIDGE-ARBITER-EVAL",
            "PPO_SEMANTICS": "EVAL_ONLY",
            "LINEAGE": lineage,
            "CONFIG_IDENTITY": {
                "plan": "experiments/marathon/rc_r1_bridge_plan.yaml",
                "namespace": doc["eval_namespace"],
                "method": cs["method"],
                "weights": "raw",
                "max_turns": 1200,
            },
            "EVIDENCE_LINKS": ["EV-0034", "EV-0049", "EV-0061", "EV-0069", "EV-0072"],
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
                f"cs_lower={cs['lower']:.4f} EV-0072"
            ),
            "ARTEFACT_LOCATIONS": [path.relative_to(REPO).as_posix()],
            "RECORDED_AT_UTC": stamp,
        }
        if not reg.exists(rec["ID"]):
            reg.add(rec)
            print("ADDED", rec["ID"])

        run_path = (
            REPO / "experiments/marathon/registry/records"
            / f"run__{run_id.replace('run#', '').replace('#', '__')}.json"
        )
        run_doc = json.loads(run_path.read_text(encoding="utf-8"))
        run_doc["RESULT"] = (
            f"COMPLETE_EXIT_0_256_UPDATES_VALID_SHARE_1.0_ZERO_ELIMINATIONS; "
            f"curriculum never advanced (screening S1 2W/57L, S2 0W/42L vs "
            f"legal_random at 8M); arbiter {OUTCOMES[name]} NO_PROMOTION EV-0072"
        )
        run_path.write_text(json.dumps(run_doc, indent=1) + "\n", encoding="utf-8", newline="\n")
        print("RUN_RESULT recorded on", run_id)

    exp_path = (
        REPO / "experiments/marathon/registry/records"
        / "experiment__rc-r1-bridge__60729d9ef92f.json"
    )
    doc = json.loads(exp_path.read_text(encoding="utf-8"))
    doc["RESULT"] = (
        "ADJUDICATED EV-0072: NO_PROMOTION both seeds per predeclared rules "
        "(0 wins in 36 games per seed; s1 16D+2L cs lower -0.628, s2 17D+1L "
        "cs lower -0.601 vs practical margin 0.01; zero faults; genuine "
        "mid-game losses to heuristic_v0/v1). Predeclared branch (b) fires: "
        "the coherent known-working-style combination (D1 128-fragment + D2 "
        "competence curriculum + D3 top-advantage 0.25 + D4 schedules) is "
        "INSUFFICIENT at screening scale; curriculum competence threshold was "
        "never reached (S1 2W/57L, S2 0W/42L greedy vs legal_random at 8M). "
        "SUCCESSOR PREDECLARED per branch (b): ONE bounded ablation round "
        "(fragment-only and curriculum-only arms) before any further "
        "interpretation; no exhaustion claim while gate gaps remain (EV-0069 "
        "MATERIAL_GAPS_REMAIN). Draw wall + passive equilibrium findings "
        "extend to the bridge recipe at this scale."
    )
    exp_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("VERDICT recorded on experiment#rc-r1-bridge#60729d9ef92f")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
