"""Allowlisted console documentation registry (no browser-supplied paths)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dashboard.backend.app.paths import REPO_ROOT

DOCS_ROOT = REPO_ROOT / "docs" / "console"


@dataclass(frozen=True)
class DocEntry:
    id: str
    title: str
    order: int
    tags: tuple[str, ...]
    rel_path: str


# Allowlist only — relative to docs/console/
DOC_REGISTRY: tuple[DocEntry, ...] = (
    DocEntry("overview", "Console overview", 1, ("start",), "01-overview.md"),
    DocEntry("startup", "Starting the dashboard", 2, ("start", "commands"), "02-startup.md"),
    DocEntry("status-stop", "Stopping and checking dashboard status", 3, ("commands",), "03-status-stop.md"),
    DocEntry("gates", "Overview and current gate meanings", 4, ("gates",), "04-gates.md"),
    DocEntry("arena", "Arena configuration and running a match", 5, ("arena",), "05-arena.md"),
    DocEntry("env-official", "Environment Lab official sessions", 6, ("environment",), "06-env-official.md"),
    DocEntry("env-demo", "Environment Lab DEMO mode", 7, ("environment", "demo"), "07-env-demo.md"),
    DocEntry("replay", "Replay Lab", 8, ("replay",), "08-replay.md"),
    DocEntry("qualification", "Qualification workflow", 9, ("qualification", "phase9q"), "09-qualification.md"),
    DocEntry("experiments", "Experiments and comparisons", 10, ("experiments",), "10-experiments.md"),
    DocEntry("training", "Training campaigns and presets", 11, ("training", "ppo"), "11-training.md"),
    DocEntry("ppo-metrics", "Interpreting PPO metrics", 12, ("training", "ppo"), "12-ppo-metrics.md"),
    DocEntry("models", "Model lifecycle and compatibility", 13, ("models",), "13-models.md"),
    DocEntry("population", "Population and PFSP", 14, ("population", "pfsp"), "14-population.md"),
    DocEntry("explainability", "Explainability and faithfulness", 15, ("explainability",), "15-explainability.md"),
    DocEntry("champion", "Champion and learned-promotion status", 16, ("champion",), "16-champion.md"),
    DocEntry("package", "Package validation", 17, ("package",), "17-package.md"),
    DocEntry("manual-upload", "Manual competition upload", 18, ("upload",), "18-manual-upload.md"),
    DocEntry("portal", "Portal observations and attribution", 19, ("portal",), "19-portal.md"),
    DocEntry("repository", "Repository status", 20, ("repository",), "20-repository.md"),
    DocEntry("opera-gx", "Opera GX Force Dark troubleshooting", 21, ("browser",), "21-opera-gx.md"),
    DocEntry("missing-data", "Missing-data meanings", 22, ("data",), "22-missing-data.md"),
    DocEntry("errors", "Backend unavailable and schema mismatch", 23, ("errors",), "23-errors.md"),
    DocEntry("recovery", "Common failures and recovery", 24, ("errors",), "24-recovery.md"),
    DocEntry("glossary", "Glossary", 25, ("glossary", "phase9q", "pfsp", "ppo"), "25-glossary.md"),
)


def documentation_index() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "DOCUMENTATION_INDEX",
        "sections": [
            {
                "id": e.id,
                "title": e.title,
                "order": e.order,
                "tags": list(e.tags),
                "path": f"docs/console/{e.rel_path}",
            }
            for e in DOC_REGISTRY
        ],
    }


def documentation_section(section_id: str) -> dict[str, Any] | None:
    entry = next((e for e in DOC_REGISTRY if e.id == section_id), None)
    if entry is None:
        return None
    path = DOCS_ROOT / entry.rel_path
    if not path.is_file():
        return {
            "schema_version": 1,
            "id": entry.id,
            "title": entry.title,
            "order": entry.order,
            "tags": list(entry.tags),
            "path": f"docs/console/{entry.rel_path}",
            "content": f"# {entry.title}\n\nNOT RECORDED — documentation file missing at `{entry.rel_path}`.\n",
            "updated_at": None,
            "availability": "MISSING",
        }
    text = path.read_text(encoding="utf-8")
    mtime = path.stat().st_mtime
    from datetime import datetime, timezone

    return {
        "schema_version": 1,
        "id": entry.id,
        "title": entry.title,
        "order": entry.order,
        "tags": list(entry.tags),
        "path": f"docs/console/{entry.rel_path}",
        "content": text,
        "updated_at": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
        "availability": "RECORDED",
    }
