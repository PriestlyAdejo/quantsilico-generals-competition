"""Versioned record validation for the local agentic orchestrator."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

SCHEMA_VERSION = "1.1.0"


class SchemaError(ValueError):
    """Raised when an orchestration record violates its durable schema."""


class State(StrEnum):
    IDLE = "IDLE"
    ARCHITECTING = "ARCHITECTING"
    IMPLEMENTING = "IMPLEMENTING"
    REVIEWING = "REVIEWING"
    REPAIRING = "REPAIRING"
    VALIDATING = "VALIDATING"
    ACCEPTED = "ACCEPTED"
    BLOCKED = "BLOCKED"
    PAUSED_USAGE = "PAUSED_USAGE"
    PAUSED_HUMAN_BOUNDARY = "PAUSED_HUMAN_BOUNDARY"
    FAILED = "FAILED"


class ReviewVerdict(StrEnum):
    ACCEPT = "ACCEPT"
    FIX_FIRST = "FIX_FIRST"
    RETHINK = "RETHINK"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    HUMAN_BOUNDARY = "HUMAN_BOUNDARY"


class ImplementationStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"
    REFUSED = "REFUSED"
    FAILED = "FAILED"


class HumanBoundaryAction(StrEnum):
    PAID_RESOURCE_CREATE_OR_EXPAND = "PAID_RESOURCE_CREATE_OR_EXPAND"
    REPOSITORY_VISIBILITY_CHANGE = "REPOSITORY_VISIBILITY_CHANGE"
    COMPETITION_UPLOAD = "COMPETITION_UPLOAD"
    DESTRUCTIVE_UNIQUE_EVIDENCE_REMOVAL = "DESTRUCTIVE_UNIQUE_EVIDENCE_REMOVAL"
    FORCE_PUSH_OR_HISTORY_REWRITE = "FORCE_PUSH_OR_HISTORY_REWRITE"
    PINNED_ENGINE_MODIFICATION = "PINNED_ENGINE_MODIFICATION"
    CREDENTIAL_OPERATION = "CREDENTIAL_OPERATION"


TASK_REQUIRED = {
    "SCHEMA_VERSION",
    "TASK_ID",
    "PLAN_STAGE",
    "GOAL",
    "FILES_OR_AREAS",
    "REPOSITORY_FACTS",
    "IMPLEMENTATION_SPEC",
    "ACCEPTANCE_CRITERIA",
    "TESTS_REQUIRED",
    "FORBIDDEN_ACTIONS",
    "HUMAN_BOUNDARY",
    "EXPECTED_OUTPUT",
}
TASK_OPTIONAL = {"VERIFICATION_ONLY"}

REPORT_REQUIRED = {
    "SCHEMA_VERSION",
    "PLAN_ID",
    "STATUS",
    "TASK_ID",
    "BASE_COMMIT",
    "HEAD_COMMIT",
    "END_COMMIT",
    "FILES_CHANGED",
    "TESTS_RUN",
    "TESTS_PASSED",
    "TESTS_FAILED",
    "EVIDENCE",
    "KNOWN_LIMITATIONS",
    "PLAN_CONFLICTS",
    "PLAN_DEVIATIONS",
    "NEXT_SAFE_ACTION",
}

REVIEW_REQUIRED = {
    "SCHEMA_VERSION",
    "VERDICT",
    "TASK_ID",
    "PROBLEM_ID",
    "FINDINGS",
    "FAILED_CRITERIA",
    "REQUIRED_FIXES",
    "OPTIONAL_IMPROVEMENTS",
    "EVIDENCE",
}


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{name} must be a JSON object")
    return value


def _validate_keys(
    record: dict[str, Any], *, required: set[str], optional: set[str] | None, name: str
) -> None:
    missing = required - record.keys()
    if missing:
        raise SchemaError(f"{name} missing keys: {sorted(missing)}")
    unknown = record.keys() - required - (optional or set())
    if unknown:
        raise SchemaError(f"{name} unknown keys: {sorted(unknown)}")
    if record["SCHEMA_VERSION"] != SCHEMA_VERSION:
        raise SchemaError(
            f"{name} SCHEMA_VERSION must be {SCHEMA_VERSION!r}, got {record['SCHEMA_VERSION']!r}"
        )


def _require_string(
    record: dict[str, Any], key: str, name: str, *, allow_empty: bool = False
) -> None:
    value = record[key]
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise SchemaError(f"{name}.{key} must be a non-empty string")


def _require_string_list(record: dict[str, Any], key: str, name: str) -> None:
    value = record[key]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SchemaError(f"{name}.{key} must be a list of strings")


def validate_task(value: Any) -> dict[str, Any]:
    record = _require_object(value, "task")
    _validate_keys(record, required=TASK_REQUIRED, optional=TASK_OPTIONAL, name="task")
    for key in ("TASK_ID", "PLAN_STAGE", "GOAL", "IMPLEMENTATION_SPEC", "EXPECTED_OUTPUT"):
        _require_string(record, key, "task")
    for key in (
        "FILES_OR_AREAS",
        "REPOSITORY_FACTS",
        "ACCEPTANCE_CRITERIA",
        "TESTS_REQUIRED",
        "FORBIDDEN_ACTIONS",
    ):
        _require_string_list(record, key, "task")
    boundary = _require_object(record["HUMAN_BOUNDARY"], "task.HUMAN_BOUNDARY")
    if set(boundary) != {"REQUIRED", "ACTIONS", "REASON"}:
        raise SchemaError("task.HUMAN_BOUNDARY requires exactly REQUIRED, ACTIONS, REASON")
    if not isinstance(boundary["REQUIRED"], bool):
        raise SchemaError("task.HUMAN_BOUNDARY.REQUIRED must be a boolean")
    if not isinstance(boundary["REASON"], str):
        raise SchemaError("task.HUMAN_BOUNDARY.REASON must be a string")
    if not isinstance(boundary["ACTIONS"], list):
        raise SchemaError("task.HUMAN_BOUNDARY.ACTIONS must be a list")
    try:
        actions = [HumanBoundaryAction(action) for action in boundary["ACTIONS"]]
    except (TypeError, ValueError) as exc:
        raise SchemaError("task.HUMAN_BOUNDARY.ACTIONS contains an invalid action") from exc
    if boundary["REQUIRED"] != bool(actions):
        raise SchemaError("task.HUMAN_BOUNDARY.REQUIRED must equal whether ACTIONS is non-empty")
    if actions and not boundary["REASON"].strip():
        raise SchemaError("task.HUMAN_BOUNDARY.REASON is required when ACTIONS are present")
    if "VERIFICATION_ONLY" in record and not isinstance(record["VERIFICATION_ONLY"], bool):
        raise SchemaError("task.VERIFICATION_ONLY must be a boolean")
    return record


def validate_report(value: Any, *, task: dict[str, Any] | None = None) -> dict[str, Any]:
    record = _require_object(value, "implementation_report")
    _validate_keys(record, required=REPORT_REQUIRED, optional=None, name="implementation_report")
    for key in (
        "PLAN_ID",
        "TASK_ID",
        "BASE_COMMIT",
        "HEAD_COMMIT",
        "END_COMMIT",
        "NEXT_SAFE_ACTION",
    ):
        _require_string(record, key, "implementation_report")
    if record["PLAN_ID"] != "MARATHON_REDESIGN_LOCKED_V1":
        raise SchemaError("implementation report PLAN_ID mismatch")
    if record["END_COMMIT"] != record["HEAD_COMMIT"]:
        raise SchemaError("implementation report END_COMMIT must equal HEAD_COMMIT")
    try:
        ImplementationStatus(record["STATUS"])
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"invalid implementation STATUS: {record['STATUS']!r}") from exc
    for key in (
        "FILES_CHANGED",
        "TESTS_RUN",
        "TESTS_PASSED",
        "TESTS_FAILED",
        "EVIDENCE",
        "KNOWN_LIMITATIONS",
        "PLAN_CONFLICTS",
        "PLAN_DEVIATIONS",
    ):
        _require_string_list(record, key, "implementation_report")
    if task is not None:
        if record["TASK_ID"] != task["TASK_ID"]:
            raise SchemaError("implementation report TASK_ID does not match current task")
        verification_only = bool(task.get("VERIFICATION_ONLY", False))
        if record["STATUS"] == ImplementationStatus.COMPLETE and not record["FILES_CHANGED"]:
            if not verification_only:
                raise SchemaError(
                    "COMPLETE implementation has no diff and is not verification-only"
                )
        if record["STATUS"] == ImplementationStatus.COMPLETE and not record["TESTS_RUN"]:
            raise SchemaError("COMPLETE implementation has no tests")
    return record


def validate_review(value: Any, *, task: dict[str, Any] | None = None) -> dict[str, Any]:
    record = _require_object(value, "review")
    _validate_keys(record, required=REVIEW_REQUIRED, optional=None, name="review")
    _require_string(record, "TASK_ID", "review")
    _require_string(record, "PROBLEM_ID", "review")
    try:
        ReviewVerdict(record["VERDICT"])
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"invalid review VERDICT: {record['VERDICT']!r}") from exc
    for key in (
        "FINDINGS",
        "FAILED_CRITERIA",
        "REQUIRED_FIXES",
        "OPTIONAL_IMPROVEMENTS",
        "EVIDENCE",
    ):
        _require_string_list(record, key, "review")
    if task is not None and record["TASK_ID"] != task["TASK_ID"]:
        raise SchemaError("review TASK_ID does not match current task")
    if record["VERDICT"] == ReviewVerdict.FIX_FIRST and not record["REQUIRED_FIXES"]:
        raise SchemaError("FIX_FIRST review must contain REQUIRED_FIXES")
    if record["VERDICT"] == ReviewVerdict.FIX_FIRST and record["PROBLEM_ID"] == "NONE":
        raise SchemaError("FIX_FIRST review must contain a stable PROBLEM_ID")
    return record


def task_json_schema() -> dict[str, Any]:
    return _json_schema(
        required=TASK_REQUIRED,
        properties={
            "SCHEMA_VERSION": {"const": SCHEMA_VERSION},
            "TASK_ID": {"type": "string", "minLength": 1},
            "PLAN_STAGE": {"type": "string", "minLength": 1},
            "GOAL": {"type": "string", "minLength": 1},
            "FILES_OR_AREAS": _string_array(),
            "REPOSITORY_FACTS": _string_array(),
            "IMPLEMENTATION_SPEC": {"type": "string", "minLength": 1},
            "ACCEPTANCE_CRITERIA": _string_array(),
            "TESTS_REQUIRED": _string_array(),
            "FORBIDDEN_ACTIONS": _string_array(),
            "HUMAN_BOUNDARY": {
                "type": "object",
                "additionalProperties": False,
                "required": ["REQUIRED", "ACTIONS", "REASON"],
                "properties": {
                    "REQUIRED": {"type": "boolean"},
                    "ACTIONS": {
                        "type": "array",
                        "items": {"enum": [item.value for item in HumanBoundaryAction]},
                    },
                    "REASON": {"type": "string"},
                },
            },
            "EXPECTED_OUTPUT": {"type": "string", "minLength": 1},
            "VERIFICATION_ONLY": {"type": "boolean"},
        },
    )


def report_json_schema() -> dict[str, Any]:
    return _json_schema(
        required=REPORT_REQUIRED,
        properties={
            "SCHEMA_VERSION": {"const": SCHEMA_VERSION},
            "PLAN_ID": {"const": "MARATHON_REDESIGN_LOCKED_V1"},
            "STATUS": {"enum": [item.value for item in ImplementationStatus]},
            "TASK_ID": {"type": "string", "minLength": 1},
            "BASE_COMMIT": {"type": "string", "minLength": 1},
            "HEAD_COMMIT": {"type": "string", "minLength": 1},
            "END_COMMIT": {"type": "string", "minLength": 1},
            "FILES_CHANGED": _string_array(),
            "TESTS_RUN": _string_array(),
            "TESTS_PASSED": _string_array(),
            "TESTS_FAILED": _string_array(),
            "EVIDENCE": _string_array(),
            "KNOWN_LIMITATIONS": _string_array(),
            "PLAN_CONFLICTS": _string_array(),
            "PLAN_DEVIATIONS": _string_array(),
            "NEXT_SAFE_ACTION": {"type": "string", "minLength": 1},
        },
    )


def review_json_schema() -> dict[str, Any]:
    return _json_schema(
        required=REVIEW_REQUIRED,
        properties={
            "SCHEMA_VERSION": {"const": SCHEMA_VERSION},
            "VERDICT": {"enum": [item.value for item in ReviewVerdict]},
            "TASK_ID": {"type": "string", "minLength": 1},
            "PROBLEM_ID": {"type": "string", "minLength": 1},
            "FINDINGS": _string_array(),
            "FAILED_CRITERIA": _string_array(),
            "REQUIRED_FIXES": _string_array(),
            "OPTIONAL_IMPROVEMENTS": _string_array(),
            "EVIDENCE": _string_array(),
        },
    )


def _string_array() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def _json_schema(*, required: set[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": sorted(required),
        "properties": properties,
    }
