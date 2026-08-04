"""INITIAL_READINESS_GATE — at most one CNN + one graph from DEVELOPMENT audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT = REPO / "experiments" / "manifests" / "development_arm_audit.json"
DEFAULT_OUT = REPO / "experiments" / "manifests" / "initial_readiness_gate.json"


def evaluate_initial_readiness(audit: dict[str, Any]) -> dict[str, Any]:
    best_cnn = audit.get("best_cnn")
    best_graph = audit.get("best_graph")
    graph_allowed = bool(audit.get("graph_training_allowed"))
    latency = audit.get("latency_classification") or {}

    blockers: list[str] = []
    selected: list[dict[str, Any]] = []

    if not best_cnn:
        blockers.append("No engineering-ok CNN DEVELOPMENT arm")
    else:
        if latency.get("recurrent_cnn_v2") not in {None, "PASS", "PARTIAL"}:
            blockers.append(f"CNN latency classification blocked: {latency.get('recurrent_cnn_v2')}")
        else:
            selected.append(
                {
                    "slot": "cnn",
                    "arm_id": best_cnn["arm_id"],
                    "architecture": best_cnn["architecture"],
                    "checkpoint": best_cnn["checkpoint"],
                    "seed": best_cnn.get("seed"),
                    "init": best_cnn.get("init"),
                }
            )

    if best_graph and graph_allowed:
        if latency.get("recurrent_graph_belief_v2") not in {None, "PASS", "PARTIAL"}:
            blockers.append(
                f"Graph latency classification blocked: {latency.get('recurrent_graph_belief_v2')}"
            )
        else:
            selected.append(
                {
                    "slot": "graph",
                    "arm_id": best_graph["arm_id"],
                    "architecture": best_graph["architecture"],
                    "checkpoint": best_graph["checkpoint"],
                    "seed": best_graph.get("seed"),
                    "init": best_graph.get("init"),
                }
            )
    elif best_graph and not graph_allowed:
        blockers.append("Graph arm present but graph_training_allowed=false")

    if not selected:
        decision = "NONE"
        status = "BLOCKED"
    else:
        decision = "READY"
        status = "READY"

    return {
        "schema_version": 1,
        "kind": "INITIAL_READINESS_GATE",
        "decision": decision,
        "status": status,
        "selected_candidates": selected,
        "max_candidates": 2,
        "rule": "At most one CNN + one graph; none → terminal without INITIAL",
        "blockers": blockers,
        "source_audit": audit.get("kind"),
        "hierarchy_note": audit.get("hierarchy_note"),
        "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
        "note": "Holdout seeds remain unused. Durable campaign telemetry is required before adaptive INITIAL launch.",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    if not args.audit.is_file():
        raise SystemExit(f"missing audit: {args.audit}")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    gate = evaluate_initial_readiness(audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": gate["decision"], "path": str(args.out), "selected": [c["arm_id"] for c in gate["selected_candidates"]]}))
    return 0 if gate["decision"] in {"READY", "NONE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
