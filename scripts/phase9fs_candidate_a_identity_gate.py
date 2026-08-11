"""CANDIDATE_A_IDENTITY_GATE — resolve portal implementation from evidence."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Search terms only — none preferred a priori (plan Stage 1A).
SEARCH_SPELLINGS = (
    "heuristic_v2f_plus_planner_terminal_fix",
    "heuristic_v2f_plus_planner_terminal_form",
    "heuristic_v2f_plus_planner_terminal_force",
)

TERMINAL_RE = re.compile(r"heuristic_v2f[^\s\"'`]{0,80}?terminal_[a-z0-9_]+")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan() -> dict[str, list[str]]:
    roots = [
        REPO / "experiments" / "manifests",
        REPO / "experiments" / "reports",
        REPO / "dist" / "roles",
        REPO / "submission",
        REPO / "src" / "generals_bot",
        REPO / "scripts",
        REPO / "tests",
        REPO / "docs",
        REPO / "plans",
        REPO / "dashboard",
    ]
    skip_parts = {"datasets", "checkpoints", "__pycache__", ".git", "node_modules"}
    found: dict[str, list[str]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in skip_parts for part in path.parts):
                continue
            if path.suffix.lower() not in {".json", ".md", ".py", ".txt", ".yml", ".yaml"}:
                continue
            if path.stat().st_size > 2_000_000:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            rel = str(path.relative_to(REPO).as_posix())
            for spelling in SEARCH_SPELLINGS:
                if spelling in text:
                    found.setdefault(spelling, [])
                    if rel not in found[spelling]:
                        found[spelling].append(rel)
            for match in TERMINAL_RE.findall(text):
                match = match.rstrip(".,);:]")
                found.setdefault(match, [])
                if rel not in found[match]:
                    found[match].append(rel)
    return found


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    found = _scan()

    from generals_bot.policies.heuristic_v2_ablations import FLAGS, V2F_PLANNER_TERMINAL
    from generals_bot.submission.builder import POLICY_IMPORTS

    pkg_reg = json.loads(
        (REPO / "experiments" / "manifests" / "phase9f_package_registry_v2.json").read_text(
            encoding="utf-8"
        )
    )
    role_reg = json.loads(
        (REPO / "experiments" / "manifests" / "phase9f_role_registry.json").read_text(encoding="utf-8")
    )

    source_canonical = V2F_PLANNER_TERMINAL
    registry_canonical = pkg_reg.get("canonical_portal_id")
    stable_id = pkg_reg.get("stable_portal_id") or "QS-P9F-PORTAL-V0"
    role_portal = role_reg.get("roles", {}).get("portal_current_verified", {}).get("candidate_id")

    evidence = {
        "source_V2F_PLANNER_TERMINAL": source_canonical,
        "source_in_FLAGS": source_canonical in FLAGS,
        "builder_POLICY_IMPORTS_has_source": source_canonical in POLICY_IMPORTS,
        "package_registry_canonical": registry_canonical,
        "role_registry_portal": role_portal,
        "spelling_file_counts": {
            k: len(v) for k, v in sorted(found.items(), key=lambda kv: -len(kv[1]))
        },
        "fix_hits": len(found.get("heuristic_v2f_plus_planner_terminal_fix", [])),
        "form_hits": len(found.get("heuristic_v2f_plus_planner_terminal_form", [])),
        "force_hits": len(found.get("heuristic_v2f_plus_planner_terminal_force", [])),
    }

    agreed = (
        source_canonical == registry_canonical == role_portal
        and source_canonical in FLAGS
        and source_canonical in POLICY_IMPORTS
    )
    if source_canonical == registry_canonical and source_canonical in FLAGS and source_canonical in POLICY_IMPORTS:
        canonical = source_canonical
        gate_status = "PASS" if agreed else "PASS_WITH_ROLE_DRIFT"
    else:
        canonical = "UNKNOWN_UNTIL_IDENTITY_GATE"
        gate_status = "FAIL"

    zip_candidates = [
        REPO / "dist" / "windows_smoke_passed" / "quantsilico_portal_current_verified_packaged.zip",
        REPO / "dist" / "legacy_mislabelled_upload_ready" / "quantsilico_portal_current_verified_packaged.zip",
    ]
    package_evidence = []
    for zp in zip_candidates:
        if zp.exists():
            package_evidence.append(
                {
                    "path": str(zp.as_posix()),
                    "sha256": _sha256(zp),
                    "size": zp.stat().st_size,
                }
            )

    doc = {
        "schema_version": 1,
        "kind": "CANDIDATE_A_IDENTITY_GATE",
        "created_at": now,
        "expected_historical_portal_id": "QS-P9F-PORTAL-V0",
        "stable_candidate_id": stable_id,
        "canonical_implementation_string": canonical,
        "gate_status": gate_status,
        "evidence": evidence,
        "spelling_paths": {k: v[:20] for k, v in found.items()},
        "existing_package_hashes": package_evidence,
        "notes": [
            "Search terms force/form/force are not preferred a priori; hashes and registries decide.",
            "Do not rename or rebuild from a guessed string.",
        ],
    }

    out = REPO / "experiments" / "manifests" / "phase9fs_candidate_a_identity_gate.json"
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    (REPO / "experiments" / "reports" / "phase9fs_candidate_a_identity_gate.md").write_text(
        "\n".join(
            [
                "# CANDIDATE_A_IDENTITY_GATE",
                "",
                f"Created: {now}",
                "",
                f"- Gate status: **{gate_status}**",
                f"- Stable ID: `{stable_id}`",
                f"- Canonical implementation: `{canonical}`",
                f"- Source FLAGS: `{source_canonical}`",
                f"- Package registry: `{registry_canonical}`",
                f"- Role registry portal: `{role_portal}`",
                (
                    f"- force hits: {evidence['fix_hits']}; "
                    f"form hits: {evidence['form_hits']}; "
                    f"force hits: {evidence['force_hits']}"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"gate_status": gate_status, "canonical": canonical, "stable_id": stable_id}, indent=2))
    return 0 if gate_status.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
