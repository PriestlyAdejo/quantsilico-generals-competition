"""Versioned filesystem schemas for experiments and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from generals_bot.schemas.portal import (
    GateStatusBoard,
    PortalMatchObservation,
    PortalSubmissionVersion,
)

SCHEMA_VERSION = 1


@dataclass
class MatchResultRecord:
    schema_version: int = SCHEMA_VERSION
    experiment_id: str = ""
    seed: int = 0
    paired_position: int = 0
    candidate: str = ""
    opponent: str = ""
    winner: int = -1
    turns: int = 0
    faults0: int = 0
    faults1: int = 0
    crash0: bool = False
    crash1: bool = False
    forfeited0: bool = False
    forfeited1: bool = False
    truncated: bool = False
    elapsed_s: float = 0.0
    peak_memory_mb0: float | None = None
    peak_memory_mb1: float | None = None
    latency_p50_ms0: float | None = None
    latency_p99_ms0: float | None = None
    latency_p50_ms1: float | None = None
    latency_p99_ms1: float | None = None
    illegal_action_count0: int = 0
    illegal_action_count1: int = 0
    protocol_fault_count0: int = 0
    protocol_fault_count1: int = 0
    replay_path: str | None = None
    telemetry_path: str | None = None
    seed_split: str | None = None
    portal_match_id: str | None = None
    portal_submission_id: str | None = None
    portal_submission_label: str | None = None
    package_sha256: str | None = None
    candidate_id: str | None = None
    attribution_method: str | None = None
    attribution_confidence: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentManifest:
    schema_version: int = SCHEMA_VERSION
    experiment_id: str = ""
    created_at: str = ""
    engine_commit: str = ""
    bot_commit: str = ""
    dirty_worktree: bool = False
    candidate: str = ""
    opponent: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    seed_file: str = ""
    seed_hash: str = ""
    seed_split: str = ""
    paired_positions: bool = True
    games: int = 0
    rules_identifier: str = "competition"
    device: str = "cpu"
    python: str = ""
    pytorch: str = ""
    jax: str = ""
    cuda: str | None = None
    gpu: str | None = None
    result_paths: list[str] = field(default_factory=list)
    replay_paths: list[str] = field(default_factory=list)
    status: str = "SCAFFOLDED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelManifest:
    schema_version: int = SCHEMA_VERSION
    model_id: str = ""
    architecture: str = ""
    parameter_count: int = 0
    checkpoint_hash: str = ""
    parent_model: str | None = None
    training_experiment: str | None = None
    lifecycle_status: str = "SCAFFOLDED"
    promotion_status: str | None = None
    package_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "SCHEMA_VERSION",
    "MatchResultRecord",
    "ExperimentManifest",
    "ModelManifest",
    "PortalSubmissionVersion",
    "PortalMatchObservation",
    "GateStatusBoard",
]
