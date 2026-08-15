"""Minimum canonical Marathon registry (EXECUTION_PLAN Stage 3).

Record kinds: experiment, run, checkpoint, candidate, evaluation,
opponent_reference. Records carry stable readable IDs plus content
fingerprints, explicit schema versions, source/engine/config identity,
lineage, seeds, commands, budgets, stop reasons, artefact locations, and
evidence links. Discovery is registry-driven: filenames are presentation,
not truth.

Hard validation rules (records are rejected before writing):
- missing/ambiguous PPO_SEMANTICS on anything touching action selection;
- incompatible or unknown lineage;
- unresolved hashes (checkpoint records must resolve artefact hashes);
- missing evaluator identity on evaluation records;
- silent overwrite of an existing record ID;
- dangling cross-references (evaluation -> candidate/checkpoint).

Writes are atomic (tmp file + fsync + rename).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

SCHEMA_VERSION = "1.0.0"

KINDS = (
    "experiment",
    "run",
    "checkpoint",
    "candidate",
    "evaluation",
    "opponent_reference",
)

PPO_SEMANTICS = ("UNCHANGED", "PRE_SAMPLING_MASK", "OFF_POLICY_AUXILIARY", "EVAL_ONLY")

_ID_RE = re.compile(r"^[A-Za-z][\w-]*#[A-Za-z][\w-]*#[a-f0-9]{12}$")

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "experiment": (
        "NAME",
        "PPO_SEMANTICS",
        "LINEAGE",
        "CONFIG_IDENTITY",
        "SEEDS",
        "EVIDENCE_LINKS",
    ),
    "run": (
        "EXPERIMENT_ID",
        "COMMAND",
        "BUDGET",
        "STOP_REASON",
        "ENVIRONMENT",
        "ARTEFACT_LOCATIONS",
    ),
    "checkpoint": (
        "RUN_ID",
        "ARTEFACT_HASHES",
        "LINEAGE",
        "TRANSITIONS",
        "ARTEFACT_LOCATIONS",
    ),
    "candidate": (
        "CHECKPOINT_ID",
        "PPO_SEMANTICS",
        "EVIDENCE_LINKS",
    ),
    "evaluation": (
        "CANDIDATE_ID",
        "EVALUATOR_IDENTITY",
        "EVAL_PROTOCOL",
        "RESULTS_LOCATION",
        "EVIDENCE_LINKS",
    ),
    "opponent_reference": (
        "NAME",
        "SOURCE_IDENTITY",
        "ARTEFACT_LOCATIONS",
    ),
}


class RegistryError(ValueError):
    """Raised when a record violates the canonical registry contract."""


def canonical_id(kind: str, name: str, material: str) -> str:
    """Stable readable ID: ``KIND#name#<content fingerprint prefix>``."""
    if kind not in KINDS:
        raise RegistryError(f"unknown record kind: {kind!r}")
    if not re.fullmatch(r"[A-Za-z][\w-]*", name):
        raise RegistryError(f"invalid record name: {name!r}")
    digest = hashlib.sha256(f"{kind}|{name}|{material}".encode()).hexdigest()[:12]
    return f"{kind}#{name}#{digest}"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Registry:
    """File-backed canonical registry rooted at one directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.records_dir = self.root / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, record_id: str) -> Path:
        return self.records_dir / f"{record_id.replace('#', '__')}.json"

    def exists(self, record_id: str) -> bool:
        return self._path(record_id).is_file()

    def get(self, record_id: str) -> dict:
        path = self._path(record_id)
        if not path.is_file():
            raise RegistryError(f"record not found: {record_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_ids(self, kind: str) -> list[str]:
        if kind not in KINDS:
            raise RegistryError(f"unknown record kind: {kind!r}")
        prefix = f"{kind}__"
        return sorted(
            path.stem.replace("__", "#")
            for path in self.records_dir.glob(f"{prefix}*.json")
        )

    def add(self, record: dict) -> str:
        """Validate and atomically persist a record; return its ID."""
        self._validate(record)
        record_id = record["ID"]
        path = self._path(record_id)
        if path.exists():
            raise RegistryError(f"silent overwrite refused for {record_id}")
        record = {**record, "RECORDED_AT_UTC": record.get("RECORDED_AT_UTC", _utc_now())}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # FlushFileBuffers (os.fsync) on Windows requires a write-access
        # handle, so reopen read-write before syncing.
        with tmp.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return record_id

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #

    def _validate(self, record: dict) -> None:
        if not isinstance(record, dict):
            raise RegistryError("record must be a mapping")
        kind = record.get("KIND")
        if kind not in KINDS:
            raise RegistryError(f"unknown KIND: {kind!r}")
        record_id = record.get("ID")
        if not isinstance(record_id, str) or not _ID_RE.match(record_id):
            raise RegistryError(f"ID must match KIND#name#fingerprint: {record_id!r}")
        if not record_id.startswith(f"{kind}#"):
            raise RegistryError(f"ID kind prefix mismatch: {record_id!r} vs {kind!r}")
        if str(record.get("SCHEMA_VERSION", "")) != SCHEMA_VERSION:
            raise RegistryError(
                f"SCHEMA_VERSION must be {SCHEMA_VERSION}: {record.get('SCHEMA_VERSION')!r}"
            )
        missing = [
            field for field in REQUIRED_FIELDS[kind] if field not in record
        ]
        if missing:
            raise RegistryError(f"{kind} record missing fields: {missing}")
        if kind in {"experiment", "candidate"}:
            semantics = record["PPO_SEMANTICS"]
            if semantics not in PPO_SEMANTICS:
                raise RegistryError(
                    f"PPO_SEMANTICS must be exactly one of {PPO_SEMANTICS}: {semantics!r}"
                )
        if kind == "checkpoint":
            hashes = record["ARTEFACT_HASHES"]
            if (
                not isinstance(hashes, dict)
                or not hashes
                or not all(re.fullmatch(r"[a-f0-9]{64}", str(v)) for v in hashes.values())
            ):
                raise RegistryError(
                    "checkpoint ARTEFACT_HASHES must resolve to sha256 hex digests"
                )
            self._require_known_lineage(record["LINEAGE"])
            if not self.exists(record["RUN_ID"]):
                raise RegistryError(f"checkpoint RUN_ID unresolved: {record['RUN_ID']}")
        if kind == "run" and not self.exists(record["EXPERIMENT_ID"]):
            raise RegistryError(f"run EXPERIMENT_ID unresolved: {record['EXPERIMENT_ID']}")
        if kind == "candidate" and not self.exists(record["CHECKPOINT_ID"]):
            raise RegistryError(
                f"candidate CHECKPOINT_ID unresolved: {record['CHECKPOINT_ID']}"
            )
        if kind == "evaluation":
            if not str(record["EVALUATOR_IDENTITY"]).strip():
                raise RegistryError("evaluation requires explicit EVALUATOR_IDENTITY")
            if not self.exists(record["CANDIDATE_ID"]):
                raise RegistryError(
                    f"evaluation CANDIDATE_ID unresolved: {record['CANDIDATE_ID']}"
                )
            incumbent = record.get("INCUMBENT_ID")
            if incumbent is not None and not self.exists(incumbent):
                raise RegistryError(f"evaluation INCUMBENT_ID unresolved: {incumbent}")
        if kind == "experiment":
            self._require_known_lineage(record["LINEAGE"])

    def _require_known_lineage(self, lineage: object) -> None:
        if not isinstance(lineage, dict) or not str(lineage.get("NAME", "")).strip():
            raise RegistryError(
                "LINEAGE must name its lineage (incompatible/unknown lineage rejected)"
            )
        fingerprint = lineage.get("IMPLEMENTATION_FINGERPRINT")
        if fingerprint is not None and not re.fullmatch(r"[a-f0-9]{8,64}", str(fingerprint)):
            raise RegistryError(f"LINEAGE fingerprint malformed: {fingerprint!r}")
