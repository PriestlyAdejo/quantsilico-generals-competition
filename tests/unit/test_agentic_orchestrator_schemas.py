from __future__ import annotations

import pytest

from tools.agentic_orchestrator.schemas import (
    SCHEMA_VERSION,
    SchemaError,
    validate_report,
    validate_review,
    validate_task,
)


def task(*, verification_only: bool = False) -> dict:
    return {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "TASK_ID": "T-1",
        "PLAN_STAGE": "STAGE_0B",
        "GOAL": "Test one bounded task.",
        "FILES_OR_AREAS": ["tools/agentic_orchestrator"],
        "REPOSITORY_FACTS": ["repository is a Git worktree"],
        "IMPLEMENTATION_SPEC": "Implement the bounded change.",
        "ACCEPTANCE_CRITERIA": ["tests pass"],
        "TESTS_REQUIRED": ["pytest"],
        "FORBIDDEN_ACTIONS": ["paid resources"],
        "HUMAN_BOUNDARY": [],
        "EXPECTED_OUTPUT": "A validated report.",
        "VERIFICATION_ONLY": verification_only,
    }


def report() -> dict:
    return {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "STATUS": "COMPLETE",
        "TASK_ID": "T-1",
        "BASE_COMMIT": "a" * 40,
        "HEAD_COMMIT": "b" * 40,
        "FILES_CHANGED": ["example.py"],
        "TESTS_RUN": ["pytest"],
        "TESTS_PASSED": ["pytest"],
        "TESTS_FAILED": [],
        "EVIDENCE": ["diff"],
        "KNOWN_LIMITATIONS": [],
        "PLAN_CONFLICTS": [],
        "NEXT_SAFE_ACTION": "Review.",
    }


def review() -> dict:
    return {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "VERDICT": "ACCEPT",
        "TASK_ID": "T-1",
        "FINDINGS": [],
        "FAILED_CRITERIA": [],
        "REQUIRED_FIXES": [],
        "OPTIONAL_IMPROVEMENTS": [],
        "EVIDENCE": ["tests"],
    }


def test_strict_task_schema_rejects_unknown_keys() -> None:
    value = task()
    value["SURPRISE"] = True
    with pytest.raises(SchemaError, match="unknown keys"):
        validate_task(value)


def test_complete_report_requires_diff_unless_verification_only() -> None:
    value = report()
    value["FILES_CHANGED"] = []
    with pytest.raises(SchemaError, match="no diff"):
        validate_report(value, task=task())
    assert validate_report(value, task=task(verification_only=True)) == value


def test_complete_report_requires_tests() -> None:
    value = report()
    value["TESTS_RUN"] = []
    with pytest.raises(SchemaError, match="no tests"):
        validate_report(value, task=task())


def test_fix_first_requires_fixes() -> None:
    value = review()
    value["VERDICT"] = "FIX_FIRST"
    with pytest.raises(SchemaError, match="REQUIRED_FIXES"):
        validate_review(value, task=task())


def test_review_rejects_noncanonical_ship_verdict() -> None:
    value = review()
    value["VERDICT"] = "SHIP"
    with pytest.raises(SchemaError, match="invalid review VERDICT"):
        validate_review(value)
