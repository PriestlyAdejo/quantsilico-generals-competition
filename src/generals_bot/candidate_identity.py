"""Candidate identity mapping for the submitted heuristic.

Authoritative executable and submission evidence ID:
  heuristic_v2f_plus_planner_terminal_fix

Do not confuse with lookalike typos:
  …_terminal_form  (dashboard allowlist typo on 4be2a55)
  …_terminal_force (transcription error)
Neither typo is a distinct registered policy.
"""

from __future__ import annotations

from typing import Any

SUBMITTED_CANDIDATE_ID = "heuristic_v2f_plus_planner_terminal_fix"
EXECUTABLE_REGISTRY_ID = "heuristic_v2f_plus_planner_terminal_fix"
DISPLAY_LABEL = "heuristic_v2f + planner + terminal fix (submitted)"

ALIASES_TO_CANONICAL: dict[str, str] = {
    "heuristic_v2f_plus_planner_terminal_form": SUBMITTED_CANDIDATE_ID,
    "heuristic_v2f_plus_planner_terminal_force": SUBMITTED_CANDIDATE_ID,
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
            },
            {
                "alias": "heuristic_v2f_plus_planner_terminal_force",
                "canonical": SUBMITTED_CANDIDATE_ID,
                "status": "TRANSCRIPTION_ERROR_NOT_DISTINCT_POLICY",
            },
        ],
        "same_as_packaged": True,
    }
