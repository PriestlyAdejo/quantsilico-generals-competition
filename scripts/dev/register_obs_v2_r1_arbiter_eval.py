"""Register OBS_V2_R1 gameplay arbiter checkpoint/candidate/evaluation
records and the experiment verdict (EV-0075).

Two candidates x 240 games each (5 opponents x 24 seat-swapped pairs =
120 pairs, 1200-turn horizon, competition mode; unit-corrected predeclared
sizes per EV-0074). Adjudication: predeclared branch (b) fires - wins in
0 seeds -> OBSERVATION_SCOPED result; no post-hoc relaxation. The harness
summary's un-guarded promoted=True artefact (zero-baseline raw-score CS)
is preserved as evidence and overridden by the NO_INCUMBENT_BASELINE guard
(commit 83b894d) plus the predeclared adjudication rules.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

RUN_ROOT = REPO / "experiments/marathon/paired_eval_runs/obs_v2_arbiter"
ARTEFACT_ROOT = REPO / "experiments/marathon/obs_v2_r1"

CHECKPOINTS = {
    "obs_v2_s1": ("OBS-V2-R1-S1", "run#obs-v2-r1-s1#a7924c01375e", 18612224),
    "obs_v2_s2": ("OBS-V2-R1-S2", "run#obs-v2-r1-s2#b8814d3240c9", 18612224),
}

OUTCOMES = {
    "obs_v2_s1": "0_WIN_114_DRAW_6_PAIRED_LOSS_ZERO_FAULTS",
    "obs_v2_s2": "0_WIN_112_DRAW_8_PAIRED_LOSS_ZERO_FAULTS",
}

VERDICT_TEXT = (
    "OBS_V2_R1 ADJUDICATED per predeclared rules: branch (b) fires - wins "
    "in 0 seeds (s1 0W/114D/6 paired losses, s2 0W/112D/8 paired losses, "
    "120 pairs = 240 games each, zero faults; 24/24 draws vs each matched "
    "rc1 control). Objective-information hypothesis at this recipe/scale is "
    "OBSERVATION_SCOPED: legal scoreboard/cell-type/enemy-memory planes did "
    "not convert to wins vs passive/heuristic opponents, though telemetry "
    "signal persists (42/33 decisive ticks) and the dev ladder swept "
    "official_expander 5/5 both seeds (first trained sweep of a "
    "win-capable scripted opponent). No post-hoc relaxation; harness "
    "promoted=True summaries are NO_INCUMBENT_BASELINE artefacts (EV-0075 "
    "guard 83b894d). Successor options predeclared per plan branch (b): "
    "deeper objective memory/history, curriculum floor, predecessor pool - "
    "ranked after RC_R1_ABLATION_R1 adjudication."
)


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
                "OBSERVATION_LINEAGE": "OBS-V2 (EV-0071 feature classification)",
                "TRANSITIONS": transitions,
                "LINEAGE": lineage,
                "ARTEFACT_HASHES": {"raw.npz": sha256_file(raw)},
                "ARTEFACT_LOCATIONS": [
                    f"experiments/marathon/obs_v2_r1/{arm}/raw.npz (npz gitignored)"
                ],
            })
            print("ADDED", ckpt_id)

    for name, (arm, _run_id, _t) in CHECKPOINTS.items():
        candidate = {
            "KIND": "candidate",
            "ID": canonical_id("candidate", name, "v1"),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "CHECKPOINT_ID": ckpt_ids[name],
            "PPO_SEMANTICS": "EVAL_ONLY",
            "PARITY_PROOF": "frozen fixture tests proving training/serving encoder equivalence (obs_v2_r1_plan parity_mandate_frozen)",
            "PROTOCOL_AGENT": f"baselines/{name}/main.py",
            "SERVING_ADAPTER": "baselines/obs_v2_serving_common.py (OBS-V2 encode_observation path; EMA weights)",
            "SANITY_PROBE": "scripts/analysis/serving_sanity_probe_obs_v2_r1.py PASS pre-game (L1 4.768/4.786, L2 deterministic; EV-0034 precedent)",
            "EVIDENCE_LINKS": ["EV-0034", "EV-0071", "EV-0073", "EV-0074", "EV-0075"],
            "RECORDED_AT_UTC": stamp,
        }
        if not reg.exists(candidate["ID"]):
            reg.add(candidate)
            print("ADDED", candidate["ID"])
        path = RUN_ROOT / name / name / "summary.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        material = doc["finished_at_utc"].replace(":", "").replace("-", "")[:12]
        cs = doc["confidence_sequence"]
        evaluation = {
            "KIND": "evaluation",
            "ID": canonical_id("evaluation", f"{name}-obs-v2-r1-arbiter", material),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "CANDIDATE_ID": candidate["ID"],
            "EVALUATOR_IDENTITY": {
                "tool": "scripts/evaluation/run_marathon_paired_eval.py",
                "method": "ANYTIME_VALID_BOUNDED_MIXTURE_CONFIDENCE_SEQUENCE",
                "confidence": 0.95,
            },
            "EVAL_PROTOCOL": {
                "namespace": "marathon-eval-v1",
                "mode": "competition",
                "opponents": [
                    "rc1_s1",
                    "rc1_s2",
                    "legal_random",
                    "heuristic_v0",
                    "heuristic_v1",
                ],
                "pairs_per_opponent": 24,
                "seat_swapped": True,
                "max_turns": 1200,
                "practical_margin": 0.01,
                "unit_note": "CS n counts seat-swapped pairs (EV-0074); 120 pairs = 240 games",
            },
            "CONFIG_IDENTITY": {
                "plan": "experiments/marathon/obs_v2_r1_plan.yaml (tier_c_arbiter_predeclared + tier_c_addendum_unit_reconciliation_v1)",
                "weights": "raw",
                "namespace": "marathon-eval-v1",
                "method": "ANYTIME_VALID_BOUNDED_MIXTURE_CONFIDENCE_SEQUENCE",
                "max_turns": 1200,
            },
            "OUTCOME": OUTCOMES[name],
            "MATCHED_CONTROL_RESULT": "24/24 draws vs rc1_s1 AND 24/24 draws vs rc1_s2 (direct observation attribution)",
            "MEAN_SCORE": doc["mean_difference"],
            "CS_LOWER_RAW_SCORE_BASELINE": cs["lower"],
            "CS_NOTE": (
                "no incumbent: CS stream is raw pair scores vs zero baseline "
                "(absolute score, NOT strength); un-guarded harness verdict "
                "promoted=True is the NO_INCUMBENT_BASELINE artefact repaired "
                "by the EV-0075 guard (commit 83b894d); promotion authority "
                "rests with the predeclared branch rules, which require wins "
                "above controls - 0 wins -> NO_PROMOTION"
            ),
            "PROMOTION": "NO_PROMOTION",
            "FAULTS": "ZERO_CANDIDATE_ZERO_OPPONENT",
            "RESULTS_LOCATION": f"experiments/marathon/paired_eval_runs/obs_v2_arbiter/{name}",
            "ARTEFACT_LOCATIONS": [
                f"experiments/marathon/paired_eval_runs/obs_v2_arbiter/{name}/{name}/summary.json"
            ],
            "LINEAGE": lineage,
            "EVIDENCE_LINKS": ["EV-0034", "EV-0071", "EV-0074", "EV-0075"],
            "RECORDED_AT_UTC": stamp,
        }
        if not reg.exists(evaluation["ID"]):
            reg.add(evaluation)
            print("ADDED", evaluation["ID"])

    exp_id = "experiment#obs-v2-r1#97f257dd096e"
    exp_path = REPO / "experiments/marathon/registry/records" / (
        exp_id.replace("#", "__").replace("experiment", "experiment") + ".json"
    )
    exp = json.loads(exp_path.read_text(encoding="utf-8"))
    exp["STATUS"] = "RESULT"
    exp["VERDICT"] = VERDICT_TEXT
    exp["VERDICT_CLASS"] = "OBSERVATION_SCOPED_NEGATIVE_BRANCH_B"
    exp["UPDATED_AT_UTC"] = stamp
    exp_path.write_text(json.dumps(exp, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("UPDATED", exp_id)

    for name, (arm, run_id, _t) in CHECKPOINTS.items():
        run_path = REPO / "experiments/marathon/registry/records" / (
            run_id.replace("#", "__") + ".json"
        )
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["STATUS"] = "COMPLETE"
        run["RESULT"] = OUTCOMES[name] + " (tier-C arbiter, EV-0075)"
        run["UPDATED_AT_UTC"] = stamp
        run_path.write_text(json.dumps(run, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print("UPDATED", run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
