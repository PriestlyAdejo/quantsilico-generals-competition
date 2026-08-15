"""Read-only Marathon registry reader for the dashboard (Stage 4B lane).

Consumes Stage-3 canonical truth (experiments/marathon/registry/records)
instead of inferring state from filenames. Pure reads: no shell, no path
execution, no Git mutation. Record kinds distinguished: experiment, run,
checkpoint (candidate/package identity remains in package_registry.json).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RECOGNISED_KINDS = (
    "experiment",
    "run",
    "checkpoint",
    "candidate",
    "evaluation",
    "opponent_reference",
)

_SUMMARY_FIELDS = (
    "ID",
    "KIND",
    "NAME",
    "PPO_SEMANTICS",
    "SEEDS",
    "BUDGET",
    "STOP_REASON",
    "RESULT",
    "TRANSITIONS",
    "EVIDENCE_LINKS",
)


def _registry_root(repo_root: Path) -> Path:
    return repo_root / "experiments" / "marathon" / "registry" / "records"


def _summarise(record: dict[str, Any]) -> dict[str, Any]:
    summary = {
        field: record.get(field)
        for field in _SUMMARY_FIELDS
        if record.get(field) is not None
    }
    lineage = record.get("LINEAGE")
    if isinstance(lineage, dict):
        summary["LINEAGE_NAME"] = lineage.get("NAME")
        summary["IMPLEMENTATION_FINGERPRINT"] = lineage.get("IMPLEMENTATION_FINGERPRINT")
    return summary


def marathon_registry_dto(repo_root: Path) -> dict[str, Any]:
    """Typed registry view grouped by KIND with counts and record summaries."""
    root = _registry_root(repo_root)
    by_kind: dict[str, list[dict[str, Any]]] = {kind: [] for kind in RECOGNISED_KINDS}
    malformed: list[str] = []
    if root.is_dir():
        for path in sorted(root.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                malformed.append(path.name)
                continue
            kind = str(record.get("KIND", "")).lower()
            if kind not in by_kind:
                malformed.append(path.name)
                continue
            by_kind[kind].append(_summarise(record))
    return {
        "source": "experiments/marathon/registry/records",
        "authority": "STAGE_3_CANONICAL_REGISTRY",
        "counts": {kind: len(records) for kind, records in by_kind.items()},
        "records": by_kind,
        "malformed": malformed,
    }
