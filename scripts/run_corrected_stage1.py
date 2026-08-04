"""Corrected Stage 1 Expander validation (Class A frozen INITIAL checkpoints)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from generals_bot.training.adaptive_initial import validate_checkpoint_vs_expander

REPO = Path(__file__).resolve().parents[1]
SEEDS = [1040, 1041, 1042, 1043]
MAX_TURNS = 100
DEVICE = "cpu"
OUT = REPO / "experiments" / "manifests" / "corrected_stage1_expander_validation.json"
DIAG = REPO / "replays" / "private" / "protocol_integrity"


def main() -> None:
    candidates = [
        {
            "arm_id": "cnn_bc_init_seed11",
            "architecture": "recurrent_cnn_v2",
            "checkpoint": REPO
            / "experiments/checkpoints/initial/initial_cnn_cnn_bc_init_seed11/chunk_0/model.json",
        },
        {
            "arm_id": "graph_bc_init_seed7",
            "architecture": "recurrent_graph_belief_v2",
            "checkpoint": REPO
            / "experiments/checkpoints/initial/initial_graph_graph_bc_init_seed7/chunk_0/model.json",
        },
    ]

    results = []
    for cand in candidates:
        ckpt = cand["checkpoint"]
        sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        seats = []
        for seat in (0, 1):
            diag = DIAG / f"stage1_{cand['arm_id']}_seat{seat}_s{SEEDS[0]}.json"
            print(f"STAGE1 {cand['arm_id']} seat={seat}", flush=True)
            res = validate_checkpoint_vs_expander(
                architecture=cand["architecture"],
                checkpoint=ckpt,
                seeds=SEEDS,
                max_turns=MAX_TURNS,
                device=DEVICE,
                learned_seat=seat,
                record_diagnostics=diag,
            )
            print(json.dumps(res), flush=True)
            seats.append(res)
        wins = sum(s["wins"] for s in seats)
        draws = sum(s["draws"] for s in seats)
        losses = sum(s["losses"] for s in seats)
        faults = sum(s["protocol_faults"] for s in seats)
        games = wins + draws + losses
        results.append(
            {
                "arm_id": cand["arm_id"],
                "architecture": cand["architecture"],
                "checkpoint": str(ckpt),
                "checkpoint_sha256": sha,
                "checkpoint_sha256_16": sha[:16],
                "frozen_initial_best": True,
                "by_seat": seats,
                "aggregate": {
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "games": games,
                    "score_rate": (wins + 0.5 * draws) / games if games else None,
                    "protocol_faults": faults,
                    "seeds": SEEDS,
                    "positions": [0, 1],
                    "opponent": "official_expander",
                },
            }
        )

    payload = {
        "schema_version": 1,
        "kind": "CORRECTED_STAGE1_EXPANDER_VALIDATION",
        "gate_name": "CORRECTED_STAGE1_EXPANDER_VALIDATION",
        "research_generation_id": "protocol_dashboard_integrity_2026-08-04",
        "decision": "COMPLETE",
        "reasons": [
            "Class A frozen checkpoints; typed forward contract; 4 seeds x both seats vs Expander"
        ],
        "blockers": [],
        "preserves_historical_initial": True,
        "historical_note": (
            "Original adaptive_initial_campaign.json 0W/2D/0L + 200 faults remains immutable"
        ),
        "candidates": results,
        "engine_sha": "9e3b9d13cca51caa1bb07db48bb85c9e90ce0462",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "supersedes": None,
        "superseded_by": None,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
