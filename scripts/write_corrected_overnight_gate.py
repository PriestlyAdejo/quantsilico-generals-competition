"""Write corrected OVERNIGHT_READINESS_GATE from Stage 1/2 corrected validation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1 = REPO / "experiments" / "manifests" / "corrected_stage1_expander_validation.json"
DEFAULT_STAGE2 = REPO / "experiments" / "manifests" / "corrected_stage2_broader_validation.json"
DEFAULT_RESEARCH = REPO / "experiments" / "manifests" / "research_evidence_integrity_gate.json"
DEFAULT_PORTAL = REPO / "experiments" / "manifests" / "portal_attribution_gate.json"
DEFAULT_OUT = REPO / "experiments" / "manifests" / "overnight_readiness_gate_corrected.json"


def _assess_candidate(agg: dict[str, Any], arm_id: str, architecture: str, ckpt: str) -> dict[str, Any]:
    blockers: list[str] = []
    wins = int(agg.get("wins") or 0)
    draws = int(agg.get("draws") or 0)
    losses = int(agg.get("losses") or 0)
    faults = int(agg.get("protocol_faults") or 0)
    score = agg.get("score_rate")
    games = int(agg.get("games") or 0)
    if faults != 0:
        blockers.append(f"{arm_id}: protocol_faults={faults} (require 0)")
    if wins <= 0:
        blockers.append(f"{arm_id}: wins={wins} (require >0 before overnight)")
    if losses > wins:
        blockers.append(f"{arm_id}: losses={losses} exceed wins={wins}")
    if isinstance(score, (int, float)) and float(score) <= 0.5 and wins == 0:
        blockers.append(
            f"{arm_id}: score_rate={score} is draw-dominated with zero wins — not overnight-ready"
        )
    if not Path(ckpt).is_file():
        blockers.append(f"{arm_id}: checkpoint missing on disk")
    decision = "BLOCKED" if blockers else "READY"
    return {
        "arm_id": arm_id,
        "architecture": architecture,
        "checkpoint": ckpt,
        "decision": decision,
        "blockers": blockers,
        "aggregate": {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "games": games,
            "score_rate": score,
            "protocol_faults": faults,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1", type=Path, default=DEFAULT_STAGE1)
    p.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    p.add_argument("--research", type=Path, default=DEFAULT_RESEARCH)
    p.add_argument("--portal", type=Path, default=DEFAULT_PORTAL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    research = json.loads(args.research.read_text(encoding="utf-8"))
    portal = json.loads(args.portal.read_text(encoding="utf-8"))
    stage1 = json.loads(args.stage1.read_text(encoding="utf-8"))
    stage2 = json.loads(args.stage2.read_text(encoding="utf-8")) if args.stage2.is_file() else None

    global_blockers: list[str] = []
    if research.get("decision") != "PASS":
        global_blockers.append(f"RESEARCH_EVIDENCE_INTEGRITY_GATE={research.get('decision')}")
    if portal.get("decision") != "RESOLVED":
        global_blockers.append(
            f"PORTAL_ATTRIBUTION_GATE={portal.get('decision')} (corrected overnight requires RESOLVED or predeclared alternate)"
        )

    source = stage2 if stage2 is not None else stage1
    assessments = []
    for cand in source.get("candidates") or []:
        assessments.append(
            _assess_candidate(
                cand.get("aggregate") or {},
                str(cand.get("arm_id")),
                str(cand.get("architecture")),
                str(cand.get("checkpoint")),
            )
        )

    ready = [a for a in assessments if a["decision"] == "READY"]
    decision = "READY" if ready and not global_blockers else "BLOCKED"
    selected = None
    if len(ready) == 1:
        selected = ready[0]["arm_id"]
    elif len(ready) > 1:
        # Both READY selection deferred — should not happen with current weak policies
        selected = "BOTH_READY_SELECTION_REQUIRED"

    gate = {
        "schema_version": 1,
        "kind": "OVERNIGHT_READINESS_GATE",
        "gate_name": "OVERNIGHT_READINESS_GATE",
        "research_generation_id": "protocol_dashboard_integrity_2026-08-04",
        "decision": decision,
        "status": decision,
        "reasons": [
            "Evaluated from corrected Stage 1/2 zero-fault validation after Class A protocol repair.",
            "Does not mutate historical overnight_readiness_gate.json from Phase 9 INITIAL.",
        ],
        "blockers": global_blockers
        + [b for a in assessments for b in a["blockers"]],
        "warnings": research.get("warnings") or [],
        "candidates": assessments,
        "selected_overnight_candidate": selected,
        "portal_attribution": portal.get("decision"),
        "canonical_portal_candidate": portal.get("canonical_candidate_id"),
        "holdout_unused": True,
        "phase10_permitted": False,
        "candidate_checkpoint_identity": {
            a["arm_id"]: a["checkpoint"] for a in assessments
        },
        "source_manifest_hashes": {
            "research_evidence_integrity_gate.json": research.get("decision"),
            "portal_attribution_gate.json": portal.get("decision"),
            "corrected_stage1_expander_validation.json": stage1.get("decision"),
            "corrected_stage2_broader_validation.json": (stage2 or {}).get("decision"),
        },
        "engine_sha": "9e3b9d13cca51caa1bb07db48bb85c9e90ce0462",
        "evaluator_sha": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "supersedes": "experiments/manifests/overnight_readiness_gate.json",
        "superseded_by": None,
        "note": "Overnight requires genuine wins with zero protocol faults. Holdout remains sealed.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": gate["decision"], "path": str(args.out), "blockers": gate["blockers"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
