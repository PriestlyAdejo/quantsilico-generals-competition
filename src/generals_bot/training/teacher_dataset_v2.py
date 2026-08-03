"""Teacher dataset versioning note for heuristic_v2_qualifier demonstrations.

Do not overwrite previous BC datasets. New late-game demos are additive.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

TEACHER_DIVERSITY = [
    "heuristic_v0",
    "heuristic_v1",
    "heuristic_v2_qualifier",
    "heuristic_aggressive",
    "heuristic_defensive",
    "heuristic_castle",
    "heuristic_deathtouch",
]

LATE_GAME_FOCUS = [
    "enemy_general_search",
    "dominant_position_conversion",
    "late_game_fog_clearing",
    "post_800_deathtouch",
    "post_1050_draw_avoidance",
    "army_concentration",
    "emergency_general_defence",
    "selective_castle_construction",
]


def teacher_dataset_manifest(*, version: str = "bc_teacher_v2_qualifier") -> dict:
    return {
        "schema_version": 1,
        "kind": "TEACHER_DATASET_VERSION",
        "version": version,
        "previous_versions_retained": True,
        "overwrite_forbidden": True,
        "teachers": TEACHER_DIVERSITY,
        "late_game_focus": LATE_GAME_FOCUS,
        "oversample_late_transitions": True,
        "dominate_whole_dataset": False,
        "record_fields": ["teacher_id", "phase"],
        "status": "SPEC_READY_COLLECTION_PENDING",
        "note": "Generate demos after heuristic_v2_qualifier passes smoke; do not overwrite v1 datasets.",
    }


def write_manifest(path: Path | None = None) -> Path:
    path = path or (REPO_ROOT / "experiments" / "manifests" / "teacher_dataset_v2_qualifier.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(teacher_dataset_manifest(), indent=2) + "\n", encoding="utf-8")
    return path
