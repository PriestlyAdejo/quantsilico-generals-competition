"""Portal submission version attribution schemas (manual / public observation only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = 1

AttributionMethod = Literal[
    "EXACT_PORTAL_VERSION_ID",
    "EXACT_PACKAGE_HASH",
    "INFERRED_ACTIVE_UPLOAD_WINDOW",
    "MANUAL_OPERATOR_ASSIGNMENT",
    "UNATTRIBUTED",
]


@dataclass
class PortalSubmissionVersion:
    schema_version: int = SCHEMA_VERSION
    portal_submission_id: str | None = None
    portal_submission_label: str = ""
    candidate_id: str = ""
    package_sha256: str = ""
    config_hash: str = ""
    content_source_commit: str = ""
    embedded_manifest_bot_commit: str | None = None
    repository_completion_commit: str | None = None
    engine_commit: str = ""
    active_from: str | None = None
    active_until: str | None = None
    lifecycle: str = "UNKNOWN"
    portal_verdict: str | None = None
    portal_gate_name: str = "PORTAL_SUBMISSION_GATE"
    attribution_method: AttributionMethod = "UNATTRIBUTED"
    attribution_confidence: str = "NONE"
    source_url_or_reference: str | None = None
    observed_at: str | None = None
    learned_model_included: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def attribution_warning(self) -> str | None:
        if self.attribution_method in {
            "INFERRED_ACTIVE_UPLOAD_WINDOW",
            "UNATTRIBUTED",
        }:
            return (
                f"WARNING: attribution_method={self.attribution_method} "
                "is not exact portal version identity."
            )
        return None


@dataclass
class PortalMatchObservation:
    schema_version: int = SCHEMA_VERSION
    portal_match_id: str | None = None
    portal_submission_id: str | None = None
    portal_submission_label: str | None = None
    package_sha256: str | None = None
    candidate_id: str | None = None
    opponent: str | None = None
    outcome: str | None = None
    turns: int | None = None
    timestamp: str | None = None
    replay_reference: str | None = None
    active_from: str | None = None
    active_until: str | None = None
    attribution_method: AttributionMethod = "UNATTRIBUTED"
    attribution_confidence: str = "NONE"
    source_url_or_reference: str | None = None
    observed_at: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.attribution_method in {
            "INFERRED_ACTIVE_UPLOAD_WINDOW",
            "UNATTRIBUTED",
        }:
            d["attribution_warning"] = (
                f"WARNING: attribution_method={self.attribution_method} "
                "is not exact portal version identity."
            )
        return d


@dataclass
class GateStatusBoard:
    """Distinguish internal research gates from portal gates."""

    schema_version: int = SCHEMA_VERSION
    HEURISTIC_DEVELOPMENT_GATE: str = "UNKNOWN"
    PRE_PPO_SUBMISSION_GATE: str = "UNKNOWN"
    PORTAL_SUBMISSION_GATE: str = "UNKNOWN"
    LEARNING_READINESS_GATE: str = "UNKNOWN"
    LEARNED_PROMOTION_GATE: str = "UNKNOWN"
    final_tournament_qualified: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
