from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.agentic_orchestrator.orchestrator import (
    ConcurrentWriterError,
    Orchestrator,
    OrchestratorError,
    RuntimePaths,
    WriterLock,
    atomic_write_json,
)
from tools.agentic_orchestrator.schemas import SCHEMA_VERSION, State

REPO = Path(__file__).resolve().parents[2]


def head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def test_atomic_write_json_replaces_complete_document(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_write_json(path, {"value": 1})
    atomic_write_json(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 2}
    assert not list(tmp_path.glob("*.tmp"))


def test_writer_lock_rejects_simultaneous_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "writer.lock"
    with WriterLock(lock_path):
        with pytest.raises(ConcurrentWriterError):
            with WriterLock(lock_path):
                pass
    assert not lock_path.exists()


def test_illegal_transition_is_rejected(tmp_path: Path) -> None:
    orchestrator = Orchestrator(repo=REPO, runtime=RuntimePaths(tmp_path))
    with pytest.raises(OrchestratorError, match="illegal transition"):
        orchestrator.transition(State.ACCEPTED, reason="invalid shortcut")


def test_dry_run_proves_repair_restart_and_pauses(tmp_path: Path) -> None:
    orchestrator = Orchestrator(repo=REPO, runtime=RuntimePaths(tmp_path))
    result = orchestrator.dry_run()
    assert result["STATE"] == "IDLE"
    assert all(result["DRY_RUN_PROOFS"].values())
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    transitions = [(event["FROM"], event["TO"]) for event in events]
    assert ("REVIEWING", "REPAIRING") in transitions
    assert ("REVIEWING", "ACCEPTED") in transitions
    assert ("ACCEPTED", "PAUSED_HUMAN_BOUNDARY") in transitions
    assert ("IDLE", "PAUSED_USAGE") in transitions
    restarted = Orchestrator(repo=REPO, runtime=RuntimePaths(tmp_path))
    assert restarted.state["STATE"] == "IDLE"


class InvalidThenValidArchitect:
    def __init__(self) -> None:
        self.calls = 0
        self.base = head()

    def architect(self, context: dict, *, correction: str | None = None) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {"bad": "schema"}
        return _task()

    def implement(self, task: dict, *, repair: dict | None = None) -> dict:
        return _report(self.base)

    def review(self, task: dict, report: dict, *, iteration: int) -> dict:
        return _review("ACCEPT")


def test_architect_schema_is_corrected_once(tmp_path: Path) -> None:
    adapter = InvalidThenValidArchitect()
    orchestrator = Orchestrator(repo=REPO, runtime=RuntimePaths(tmp_path))
    result = orchestrator.run_once(adapter)
    assert adapter.calls == 2
    assert result["STATE"] == "ACCEPTED"


class HumanBoundaryArchitect(InvalidThenValidArchitect):
    def architect(self, context: dict, *, correction: str | None = None) -> dict:
        value = _task()
        value["HUMAN_BOUNDARY"] = ["REQUIRES_NOW:PAID_RESOURCE_CREATE_OR_EXPAND"]
        return value


def test_active_human_boundary_pauses_before_implementation(tmp_path: Path) -> None:
    orchestrator = Orchestrator(repo=REPO, runtime=RuntimePaths(tmp_path))
    result = orchestrator.run_once(HumanBoundaryArchitect())
    assert result["STATE"] == "PAUSED_HUMAN_BOUNDARY"
    assert not (tmp_path / "implementation_report.json").exists()


def test_nonverification_completion_requires_active_state_update(tmp_path: Path) -> None:
    orchestrator = Orchestrator(repo=REPO, runtime=RuntimePaths(tmp_path))
    base = head()
    orchestrator.state["BASE_COMMIT"] = base
    task = _task()
    task["VERIFICATION_ONLY"] = False
    report = _report(base)
    report["HEAD_COMMIT"] = "WORKTREE"
    report["FILES_CHANGED"] = ["tools/example.py"]
    with pytest.raises(OrchestratorError, match="ACTIVE_STATE"):
        orchestrator._validate_repository_report(task, report)


class RepeatedFailureAdapter(InvalidThenValidArchitect):
    def __init__(self) -> None:
        super().__init__()
        self.implement_calls = 0

    def architect(self, context: dict, *, correction: str | None = None) -> dict:
        return _task()

    def implement(self, task: dict, *, repair: dict | None = None) -> dict:
        self.implement_calls += 1
        value = _report(self.base)
        value["STATUS"] = "PARTIAL"
        value["TESTS_FAILED"] = ["same failure"]
        return value

    def review(self, task: dict, report: dict, *, iteration: int) -> dict:
        value = _review("FIX_FIRST")
        value["FAILED_CRITERIA"] = ["same problem"]
        value["REQUIRED_FIXES"] = ["fix the same substantive problem"]
        return value


def test_same_problem_twice_escalates_before_absolute_cap(tmp_path: Path) -> None:
    orchestrator = Orchestrator(repo=REPO, runtime=RuntimePaths(tmp_path))
    adapter = RepeatedFailureAdapter()
    with pytest.raises(OrchestratorError, match="escalation threshold"):
        orchestrator.run_once(adapter)
    assert orchestrator.reload()["STATE"] == "BLOCKED"
    assert orchestrator.state["REPAIR_ITERATION"] == 2


def _task() -> dict:
    return {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "TASK_ID": "TEST_TASK",
        "PLAN_STAGE": "STAGE_0B",
        "GOAL": "Verify orchestration.",
        "FILES_OR_AREAS": ["var/agentic"],
        "REPOSITORY_FACTS": ["test repository"],
        "IMPLEMENTATION_SPEC": "Return deterministic records.",
        "ACCEPTANCE_CRITERIA": ["records validate"],
        "TESTS_REQUIRED": ["pytest"],
        "FORBIDDEN_ACTIONS": ["repository edit"],
        "HUMAN_BOUNDARY": [],
        "EXPECTED_OUTPUT": "Validated report.",
        "VERIFICATION_ONLY": True,
    }


def _report(base: str) -> dict:
    return {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "STATUS": "COMPLETE",
        "TASK_ID": "TEST_TASK",
        "BASE_COMMIT": base,
        "HEAD_COMMIT": base,
        "FILES_CHANGED": [],
        "TESTS_RUN": ["pytest"],
        "TESTS_PASSED": ["pytest"],
        "TESTS_FAILED": [],
        "EVIDENCE": ["test"],
        "KNOWN_LIMITATIONS": [],
        "PLAN_CONFLICTS": [],
        "NEXT_SAFE_ACTION": "review",
    }


def _review(verdict: str) -> dict:
    return {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "VERDICT": verdict,
        "TASK_ID": "TEST_TASK",
        "FINDINGS": [],
        "FAILED_CRITERIA": [],
        "REQUIRED_FIXES": [],
        "OPTIONAL_IMPROVEMENTS": [],
        "EVIDENCE": ["test"],
    }
