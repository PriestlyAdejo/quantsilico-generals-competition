"""Read-only repository cleanup classification inventory (Stage 4B lane).

Produces a DRY-RUN classification report using the canonical cleanup
classes KEEP / MIGRATE / ARCHIVE / REGENERABLE / DELETE_CANDIDATE /
UNKNOWN (AGENTS.md implementation discipline). NEVER deletes or moves
anything; unique evidence is never classified DELETE_CANDIDATE by
heuristic alone — such entries are only SUGGESTIONS requiring architect
sign-off.

Usage:
  python scripts/dev/repo_cleanup_inventory.py [--out path.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Paths holding unique research evidence: heuristics alone must never
# propose deletion here.
EVIDENCE_PROTECTED_PREFIXES = (
    "experiments",
    "docs/marathon",
    "models",
    "replays",
    "submission/packages",
    "submission/public_versions",
    "submission/manifests",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _protected(rel: str) -> bool:
    return any(
        rel == prefix or rel.startswith(prefix + "/")
        for prefix in EVIDENCE_PROTECTED_PREFIXES
    )


def classify_dist() -> list[dict]:
    """dist/ provenance + hash inventory (canonical cleanup target area)."""
    entries = []
    dist = REPO / "dist"
    if not dist.is_dir():
        return entries
    for path in sorted(dist.rglob("*")):
        if not path.is_file():
            continue
        rel = _rel(path)
        has_manifest = (path.parent / f"{path.stem}.manifest.json").exists() or (
            path.with_suffix(path.suffix + ".sha256").exists()
        )
        entries.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "provenance_sidecar": has_manifest,
                "suggested_class": "ARCHIVE" if has_manifest else "UNKNOWN",
                "reason": (
                    "hash + sidecar present; archive under provenance layout"
                    if has_manifest
                    else "no provenance sidecar; needs architect review"
                ),
            }
        )
    return entries


def classify_empty_dirs() -> list[dict]:
    """Empty scaffolds (.gitkeep-only or truly empty) outside evidence paths."""
    entries = []
    for base in ("submission", "training", "tools", "var"):
        root = REPO / base
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_dir():
                continue
            rel = _rel(path)
            if _protected(rel):
                continue
            contents = list(path.iterdir())
            if not contents:
                entries.append(
                    {
                        "path": rel,
                        "suggested_class": "DELETE_CANDIDATE",
                        "reason": "empty directory",
                    }
                )
            elif all(child.name == ".gitkeep" for child in contents):
                entries.append(
                    {
                        "path": rel,
                        "suggested_class": "DELETE_CANDIDATE",
                        "reason": ".gitkeep-only scaffold",
                    }
                )
    return entries


def classify_staging() -> list[dict]:
    """submission/staging leftovers: regenerable by design, never unique."""
    entries = []
    staging = REPO / "submission" / "staging"
    if not staging.is_dir():
        return entries
    for child in sorted(staging.iterdir()):
        rel = _rel(child)
        if child.name == ".gitkeep":
            continue
        entries.append(
            {
                "path": rel,
                "suggested_class": "REGENERABLE",
                "reason": "staging is rebuilt by packaging; verify empty before removal",
                "empty": not any(child.iterdir()) if child.is_dir() else False,
            }
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = {
        "schema_version": 1,
        "kind": "REPO_CLEANUP_INVENTORY_DRY_RUN",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classes": [
            "KEEP",
            "MIGRATE",
            "ARCHIVE",
            "REGENERABLE",
            "DELETE_CANDIDATE",
            "UNKNOWN",
        ],
        "policy": (
            "DRY RUN ONLY. Nothing is deleted or moved. Entries under "
            "evidence-protected prefixes are never suggested for deletion. "
            "DELETE_CANDIDATE entries require architect sign-off."
        ),
        "dist_inventory": classify_dist(),
        "empty_scaffolds": classify_empty_dirs(),
        "staging_leftovers": classify_staging(),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"WROTE {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
