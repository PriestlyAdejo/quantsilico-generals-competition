"""Stage 4–5: freeze daytime runtime + eval protocol SHA; record parent class."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    prog = json.loads((ROOT / "experiments/manifests/competition_native_jax_v4_3_programme_state.json").read_text())
    frozen = json.loads((ROOT / "experiments/manifests/competition_native_jax_v4_2_frozen_baseline.json").read_text())
    snap = json.loads((ROOT / "experiments/manifests/competition_native_jax_v4_3_source_snapshot.json").read_text())

    selected = prog.get("selected_runtime") or {"num_envs": 32, "rollout_len": 32, "reset_pool_size": 4096}
    operational_tps = float(frozen["operational_smoke_tps"])
    samples_per_update = int(selected["num_envs"]) * int(selected["rollout_len"])

    # R-E.6: 90 minutes wall, 85% usable
    re6_seconds = 90 * 60
    re6_max_updates = int(math.floor((0.85 * operational_tps * re6_seconds) / samples_per_update))
    re6_transition_budget = re6_max_updates * samples_per_update

    # R-E.7: 4 hours wall, 85% usable (conditional)
    re7_seconds = 4 * 60 * 60
    re7_max_updates = int(math.floor((0.85 * operational_tps * re7_seconds) / samples_per_update))
    re7_transition_budget = re7_max_updates * samples_per_update

    runtime = {
        "schema_version": 1,
        "kind": "DAYTIME_RUNTIME_SELECTED",
        "status": "FROZEN",
        "profile": "V4_2",
        "selected": selected,
        "operational_tps_for_budgets": operational_tps,
        "clean_benchmark_tps": float(frozen["clean_benchmark_tps"]),
        "restore_threshold_tps": float(frozen["restore_threshold_tps"]),
        "source_snapshot": "experiments/manifests/competition_native_jax_v4_3_source_snapshot.json",
        "env_implementation_hash": snap["env_implementation_hash"],
        "learner_implementation_hash": snap["learner_implementation_hash"],
        "env_semantics_hash": snap["env_semantics_hash"],
        "parent_class": "R_E6_PARENT_COMPATIBLE_COLD_RESTART",
        "v4_3a_disposition": "V4_3_NO_MATERIAL_GAIN_USE_V4_2",
        "budgets": {
            "samples_per_update": samples_per_update,
            "r_e6": {
                "max_seconds": re6_seconds,
                "max_complete_updates": re6_max_updates,
                "transition_budget": re6_transition_budget,
                "usable_fraction": 0.85,
            },
            "r_e7": {
                "max_seconds": re7_seconds,
                "max_complete_updates": re7_max_updates,
                "transition_budget": re7_transition_budget,
                "usable_fraction": 0.85,
            },
        },
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "experiments/manifests/daytime_runtime_selected.json").write_text(json.dumps(runtime, indent=2) + "\n")

    proto_path = ROOT / "experiments/manifests/competition_native_jax_daytime_evaluation_protocol_v2.json"
    proto = json.loads(proto_path.read_text())
    # Validate completeness
    required = ["screening", "selection", "confirmation", "opponents", "gates", "scoring", "seat_swaps", "faults"]
    missing = [k for k in required if k not in proto]
    if missing:
        raise SystemExit(f"protocol incomplete: {missing}")
    # Seed overlap check
    s = set(proto["screening"]["seeds"])
    sel = set(proto["selection"]["seeds"])
    conf = set(proto["confirmation"]["seeds"])
    if s & sel or s & conf or sel & conf:
        raise SystemExit("protocol seed overlap")
    proto["status"] = "FROZEN"
    proto["frozen_before_r_e6_launch"] = True
    proto["frozen_at"] = datetime.now(timezone.utc).isoformat()
    # Write then hash (hash of frozen content)
    text = json.dumps(proto, indent=2) + "\n"
    proto_path.write_text(text)
    sha = hashlib.sha256(proto_path.read_bytes()).hexdigest()
    proto["sha256"] = sha
    # Re-write with sha embedded — then recompute final sha of that file for launch manifest
    proto_path.write_text(json.dumps(proto, indent=2) + "\n")
    final_sha = hashlib.sha256(proto_path.read_bytes()).hexdigest()
    proto["sha256"] = final_sha
    proto_path.write_text(json.dumps(proto, indent=2) + "\n")
    final_sha = hashlib.sha256(proto_path.read_bytes()).hexdigest()

    freeze_rec = {
        "schema_version": 1,
        "kind": "DAYTIME_EVAL_PROTOCOL_FREEZE",
        "evaluation_protocol_id": proto["protocol_id"],
        "evaluation_protocol_path": "experiments/manifests/competition_native_jax_daytime_evaluation_protocol_v2.json",
        "evaluation_protocol_sha256": final_sha,
        "immutable_after_r_e6_start": True,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "experiments/manifests/competition_native_jax_daytime_eval_protocol_freeze.json").write_text(
        json.dumps(freeze_rec, indent=2) + "\n"
    )

    prog["status"] = "STAGE_4_RUNTIME_FROZEN"
    prog["current_stage"] = "STAGE_5_CHECKPOINT_ROUNDTRIP"
    prog["daytime_runtime_selected"] = "experiments/manifests/daytime_runtime_selected.json"
    prog["evaluation_protocol_id"] = proto["protocol_id"]
    prog["evaluation_protocol_sha256"] = final_sha
    prog["parent_class"] = "R_E6_PARENT_COMPATIBLE_COLD_RESTART"
    (ROOT / "experiments/manifests/competition_native_jax_v4_3_programme_state.json").write_text(
        json.dumps(prog, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "runtime": selected,
                "re6_updates": re6_max_updates,
                "re6_transitions": re6_transition_budget,
                "protocol_id": proto["protocol_id"],
                "protocol_sha256": final_sha,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
