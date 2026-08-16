"""Register BC-A-FULL-R1 execution + adjudication records (EV-0060).

VALID NEGATIVE per predeclared rules (bc_a_full_round_1_plan.yaml):
held-out-player top-1 strictly above BOTH legal-uniform and majority-pass
baselines on at least one holdout player -> NOT MET (beats legal-uniform
by ~0.50 margin both players; falls 0.16-0.19 short of majority-pass).
No post-hoc relaxation. Elite-replay BC lane NOT VIABLE at current corpus
scale per escalation_note; negative preserved.
PPO_SEMANTICS: OFF_POLICY_AUXILIARY.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.marathon_registry import SCHEMA_VERSION, Registry, canonical_id  # noqa: E402

RESULT = REPO / "var/marathon_takeover/bc_a_full/bc_a_full_result.json"
ADJUDICATION = REPO / "var/marathon_takeover/bc_a_full/adjudication.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    reg = Registry(REPO / "experiments/marathon/registry")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc = json.loads(RESULT.read_text(encoding="utf-8"))
    adj = json.loads(ADJUDICATION.read_text(encoding="utf-8"))

    run = {
        "KIND": "run",
        "ID": canonical_id("run", "bc-a-full-r1-execution", doc["samples_sha256"][:12]),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "NAME": "BC-A-FULL-R1-EXECUTION",
        "EXPERIMENT_ID": "experiment#bc-a-full-r1#abcfcd746ac7",
        "PPO_SEMANTICS": "OFF_POLICY_AUXILIARY",
        "DATASET_ID": doc["shard_dataset_id"],
        "ENGINE_SHA": doc["engine_sha"],
        "SEEDS": [doc["seed"]],
        "EPOCHS": doc["epochs"],
        "BATCH_SIZE": doc["batch_size"],
        "COMMAND": (
            "remote_bc_a_full_r1_orchestrator.sh -> bc_a_train_full.py "
            f"--shard-dir DATASET-BC-DERIVED-UNION-A2 --epochs {doc['epochs']} "
            f"--seed {doc['seed']} (self-stop wrapper)"
        ),
        "BUDGET": f"{doc['epochs']} epochs on 81-game union (feature phase + training); wall {doc['wall_s']}s",
        "STOP_REASON": "EPOCHS_COMPLETE",
        "ENVIRONMENT": {"python": "3.12.14", "jax": "0.11.0+cuda12 (epochs; features CPU)"},
        "HOLDOUT_PLAYERS": doc["holdout_players"],
        "HARDWARE": {
            "provider": "runpod",
            "pod_id": "5pxc0rnfy34nal",
            "gpu": "NVIDIA A40 (epochs; feature phase CPU-bound)",
            "rate_usd_per_hr": 0.44,
            "data_center": "EU-SE-1",
            "machine_id": "5yue4r7xuzpm",
        },
        "SOURCE_COMMIT": "e3a4627",
        "ROUND_START_UTC": "2026-08-16T14:52:45Z",
        "ROUND_END_UTC": "2026-08-16T20:24:14Z",
        "WALL_S": doc["wall_s"],
        "LOCAL_INCIDENT": "EV-0059: first local attempt killed epoch 5/40 by RAM contention; predeclared A40 fallback clause executed (deterministic same-seed same-shard rerun)",
        "ARTEFACT_LOCATIONS": [
            "var/marathon_takeover/bc_a_full/bc_a_full_result.json",
            "var/marathon_takeover/bc_a_full/bc_a_full_round.log",
            "var/marathon_takeover/bc_a_full/adjudication.json",
        ],
        "ARTEFACT_HASHES": {
            "bc_a_full_result.json": sha256_file(RESULT),
            "adjudication.json": sha256_file(ADJUDICATION),
        },
        "RESULT": (
            "VALID_NEGATIVE per predeclared rules: held-out-player top-1 "
            f"ResBot={adj['splits']['ResBot']['top1']} "
            f"nanomena={adj['splits']['nanomena']['top1']} beats legal-uniform "
            "(~0.023) by ~0.50 margin but falls below majority-pass "
            f"({adj['splits']['ResBot']['majority_pass_baseline']}/"
            f"{adj['splits']['nanomena']['majority_pass_baseline']}) by "
            f"{abs(adj['splits']['ResBot']['margin_vs_majority_pass'])}/"
            f"{abs(adj['splits']['nanomena']['margin_vs_majority_pass'])}; "
            "legal share 1.0 all splits; train top-1 "
            f"{doc['train']['top1_accuracy']} (overfitting gap recorded). "
            "Lane NOT VIABLE at current corpus scale (escalation_note); "
            "checkpoint never enters funnel; negative preserved EV-0060"
        ),
        "RECORDED_AT_UTC": stamp,
    }
    if not reg.exists(run["ID"]):
        reg.add(run)
        print("ADDED", run["ID"])

    exp_path = (
        REPO / "experiments/marathon/registry/records"
        / "experiment__bc-a-full-r1__abcfcd746ac7.json"
    )
    exp = json.loads(exp_path.read_text(encoding="utf-8"))
    exp["RESULT"] = (
        "ADJUDICATED EV-0060: VALID NEGATIVE (predeclared rules, no relaxation). "
        "Pipeline validated at scale (81-game/26-player union, leak-guard gates, "
        "engine-verified labels, legal share 1.0); held-out generalisation REAL "
        "but INSUFFICIENT - 52-54% top-1 vs ~2.3% legal-uniform yet below ~70% "
        "majority-pass on both holdout players (ResBot, nanomena); phase profile "
        "0-199: 0.408, 200-399: 0.638, 400-799: 0.579, 800+: 0.386; train 0.970. "
        "Consumption gate: FAIL -> BC lane NOT VIABLE at current corpus scale; "
        "no warm-start enters the funnel; elite replay data plane preserved for "
        "other charter sub-experiments."
    )
    exp["CONFIG_IDENTITY"]["status"] = "ADJUDICATED_VALID_NEGATIVE"
    exp_path.write_text(json.dumps(exp, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("VERDICT recorded on experiment#bc-a-full-r1#abcfcd746ac7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
