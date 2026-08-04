"""Candidate identity mapping for submitted heuristic vs dashboard typos.

Authoritative executable and submission evidence ID:
  heuristic_v2f_plus_planner_terminal_force

The string heuristic_v2f_plus_planner_terminal_form appeared only as a
dashboard allowlist typo on feature/figma-console-integration @ 4be2a55.
It is NOT a distinct registered policy.
"""

from __future__ import annotations

from typing import Any

SUBMITTED_CANDIDATE_ID = "heuristic_v2f_plus_planner_terminal_force"
EXECUTABLE_REGISTRY_ID = "heuristic_v2f_plus_planner_terminal_force"
DISPLAY_LABEL = "heuristic_v2f + planner + terminal force (submitted)"

# Historical / erroneous strings → canonical ID (not separate policies).
ALIASES_TO_CANONICAL: dict[str, str] = {
    "heuristic_v2f_plus_planner_terminal_form": SUBMITTED_CANDIDATE_ID,
}


def canonicalize_candidate_id(candidate_id: str) -> str:
    key = candidate_id.strip()
    return ALIASES_TO_CANONICAL.get(key, key)


def candidate_identity_record() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "CANDIDATE_IDENTITY_MAPPING",
        "submitted_evidence_id": SUBMITTED_CANDIDATE_ID,
        "executable_registry_id": EXECUTABLE_REGISTRY_ID,
        "display_label": DISPLAY_LABEL,
        "aliases": [
            {
                "alias": "heuristic_v2f_plus_planner_terminal_form",
                "canonical": SUBMITTED_CANDIDATE_ID,
                "status": "DASHBOARD_TYPO_NOT_DISTINCT_POLICY",
                "note": "Introduced erroneously in 4be2a55 allowlist; discarded for Arena launch.",
            }
        ],
        "same_as_packaged": True,
    }
