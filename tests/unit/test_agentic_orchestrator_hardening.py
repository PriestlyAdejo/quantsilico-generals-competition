"""Targeted tests for the Stage 0B schema-hardening behaviours.

Covers the uncommitted Codex hardening diff: END_COMMIT/PLAN_ID/PLAN_DEVIATIONS
report rules, stable PROBLEM_ID review rules, structured HUMAN_BOUNDARY,
recursive redaction, keyword boundary inference, PARTIAL-report validation,
and ToolingGateError routing in the CLI entrypoint.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.agentic_orchestrator import __main__ as main_module
from tools.agentic_orchestrator.orchestrator import (
    Orchestrator,
    OrchestratorError,
    RuntimePaths,
    ToolingGateError,
    _actual_changed_paths,
)
from tools.agentic_orchestrator.schemas import (
    SCHEMA_VERSION,
    SchemaError,
    validate_report,
    validate_review,
    validate_task,
)
from tools.agentic_orchestrator.subprocess_runner import sanitize_value

REPO = Path(__file__).resolve().parents[2]


def head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _boundary(**overrides: object) -> dict:
    value = {"REQUIRED": False, "ACTIONS": [], "REASON": ""}
    value.update(overrides)
    return value


def _task(**overrides: object) -> dict:
    value = {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "TASK_ID": "T-HARDENING",
        "PLAN_STAGE": "STAGE_0B",
        "GOAL": "Harden orchestration.",
        "FILES_OR_AREAS": ["tools/agentic_orchestrator"],
        "REPOSITORY_FACTS": ["fact"],
        "IMPLEMENTATION_SPEC": "Apply hardening.",
        "ACCEPTANCE_CRITERIA": ["tests pass"],
        "TESTS_REQUIRED": ["pytest"],
        "FORBIDDEN_ACTIONS": ["paid resources"],
        "HUMAN_BOUNDARY": _boundary(),
        "EXPECTED_OUTPUT": "Validated records.",
        "VERIFICATION_ONLY": True,
    }
    value.update(overrides)
    return value


def _report(base: str, **overrides: object) -> dict:
    value = {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "PLAN_ID": "MARATHON_REDESIGN_LOCKED_V1",
        "STATUS": "COMPLETE",
        "TASK_ID": "T-HARDENING",
        "BASE_COMMIT": base,
        "HEAD_COMMIT": base,
        "END_COMMIT": base,
        "FILES_CHANGED": ["tools/example.py"],
        "TESTS_RUN": ["pytest"],
        "TESTS_PASSED": ["pytest"],
        "TESTS_FAILED": [],
        "EVIDENCE": ["diff"],
        "KNOWN_LIMITATIONS": [],
        "PLAN_CONFLICTS": [],
        "PLAN_DEVIATIONS": [],
        "NEXT_SAFE_ACTION": "Review.",
    }
    value.update(overrides)
    return value


def _review(verdict: str, **overrides: object) -> dict:
    value = {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "VERDICT": verdict,
        "TASK_ID": "T-HARDENING",
        "PROBLEM_ID": "NONE",
        "FINDINGS": [],
        "FAILED_CRITERIA": [],
        "REQUIRED_FIXES": [],
        "OPTIONAL_IMPROVEMENTS": [],
        "EVIDENCE": ["review"],
    }
    value.update(overrides)
    return value


class TestReportHardening:
    def test_end_commit_must_equal_head_commit(self) -> None:
        base = "a" * 40
        with pytest.raises(SchemaError, match="END_COMMIT"):
            validate_report(_report(base, END_COMMIT="b" * 40))

    def test_plan_id_must_match_locked_programme(self) -> None:
        with pytest.raises(SchemaError, match="PLAN_ID"):
            validate_report(_report("a" * 40, PLAN_ID="SOME_OTHER_PLAN"))

    def test_plan_deviations_is_a_required_string_list(self) -> None:
        record = _report("a" * 40)
        del record["PLAN_DEVIATIONS"]
        with pytest.raises(SchemaError, match="PLAN_DEVIATIONS"):
            validate_report(record)


class TestReviewHardening:
    def test_fix_first_requires_stable_problem_id(self) -> None:
        review = _review("FIX_FIRST", REQUIRED_FIXES=["fix it"])
        with pytest.raises(SchemaError, match="PROBLEM_ID"):
            validate_review(review)

    def test_fix_first_with_problem_id_passes(self) -> None:
        review = _review(
            "FIX_FIRST", REQUIRED_FIXES=["fix it"], PROBLEM_ID="STABLE_PROBLEM_1"
        )
        assert validate_review(review)["PROBLEM_ID"] == "STABLE_PROBLEM_1"


class TestHumanBoundarySchema:
    def test_required_must_match_actions_presence(self) -> None:
        with pytest.raises(SchemaError, match="REQUIRED"):
            validate_task(_task(HUMAN_BOUNDARY=_boundary(REQUIRED=True)))

    def test_actions_require_a_reason(self) -> None:
        boundary = _boundary(
            REQUIRED=True, ACTIONS=["COMPETITION_UPLOAD"], REASON="   "
        )
        with pytest.raises(SchemaError, match="REASON"):
            validate_task(_task(HUMAN_BOUNDARY=boundary))

    def test_unknown_actions_are_rejected(self) -> None:
        boundary = _boundary(
            REQUIRED=True, ACTIONS=["NOT_A_REAL_ACTION"], REASON="attempt"
        )
        with pytest.raises(SchemaError, match="invalid action"):
            validate_task(_task(HUMAN_BOUNDARY=boundary))

    def test_exact_key_set_is_enforced(self) -> None:
        boundary = _boundary(EXTRA="nope")
        with pytest.raises(SchemaError, match="exactly REQUIRED"):
            validate_task(_task(HUMAN_BOUNDARY=boundary))


class TestSanitizeValue:
    def test_recursively_redacts_nested_structures(self) -> None:
        value = {
            "nested": {"auth": "token: abc123secret"},
            "list": ["bearer AAAA.BBBB.CCCC", 7, None],
        }
        sanitized = sanitize_value(value)
        assert "[REDACTED]" in sanitized["nested"]["auth"]
        assert "[REDACTED]" in sanitized["list"][0]
        assert sanitized["list"][1] == 7
        assert sanitized["list"][2] is None
        assert "abc123secret" not in json.dumps(sanitized)


class TestBoundaryInference:
    def _actions(self, tmp_path: Path, **task_overrides: object) -> set[str]:
        orchestrator = Orchestrator(repo=REPO, runtime=RuntimePaths(tmp_path))
        return orchestrator._human_boundary_actions(_task(**task_overrides))

    def test_declared_actions_are_preserved(self, tmp_path: Path) -> None:
        actions = self._actions(
            tmp_path,
            HUMAN_BOUNDARY=_boundary(
                REQUIRED=True,
                ACTIONS=["CREDENTIAL_OPERATION"],
                REASON="rotate token",
            ),
        )
        assert "CREDENTIAL_OPERATION" in actions

    def test_keywords_infer_boundaries(self, tmp_path: Path) -> None:
        actions = self._actions(
            tmp_path, GOAL="Force push the branch and create pod for training."
        )
        assert "FORCE_PUSH_OR_HISTORY_REWRITE" in actions
        assert "PAID_RESOURCE_CREATE_OR_EXPAND" in actions

    def test_pinned_engine_reference_is_inferred(self, tmp_path: Path) -> None:
        actions = self._actions(tmp_path, FILES_OR_AREAS=["third_party/generals-bots"])
        assert "PINNED_ENGINE_MODIFICATION" in actions

    def test_benign_task_infers_nothing(self, tmp_path: Path) -> None:
        assert self._actions(tmp_path) == set()


class TestPartialReportValidation:
    def test_partial_report_does_not_crash_on_dirty_worktree(self, tmp_path: Path) -> None:
        orchestrator = Orchestrator(repo=REPO, runtime=RuntimePaths(tmp_path))
        base = head()
        orchestrator.state["BASE_COMMIT"] = base
        report = _report(base, STATUS="PARTIAL", TESTS_FAILED=["one failure"])
        # Regression: non-COMPLETE reports previously raised NameError because
        # normalized_files/actual were only bound inside the COMPLETE branch.
        orchestrator._validate_repository_report(_task(), report)

    def test_partial_report_still_checks_end_commit(self, tmp_path: Path) -> None:
        orchestrator = Orchestrator(repo=REPO, runtime=RuntimePaths(tmp_path))
        base = head()
        orchestrator.state["BASE_COMMIT"] = base
        report = _report(
            base, STATUS="PARTIAL", HEAD_COMMIT="z" * 40, END_COMMIT="z" * 40
        )
        with pytest.raises(OrchestratorError, match="END_COMMIT"):
            orchestrator._validate_repository_report(_task(), report)

    def test_complete_report_rejects_files_mismatch(self, tmp_path: Path) -> None:
        orchestrator = Orchestrator(repo=REPO, runtime=RuntimePaths(tmp_path))
        base = head()
        orchestrator.state["BASE_COMMIT"] = base
        actual = sorted(_actual_changed_paths(REPO, base))
        report = _report(base, FILES_CHANGED=["not/real.py"])
        with pytest.raises(OrchestratorError, match="FILES_CHANGED"):
            orchestrator._validate_repository_report(
                _task(FILES_OR_AREAS=actual or ["not"]), report
            )


def _run_args(tmp_path: Path) -> list[str]:
    return [
        "--repo",
        str(REPO),
        "--runtime-dir",
        str(tmp_path / "runtime"),
        "run",
        "--once",
    ]


class TestToolingGateRouting:
    def _failing_adapter(self, classification: str, detail: str):
        def factory(**_kwargs: object):
            raise ToolingGateError(classification, detail)

        return factory

    def test_usage_exhausted_pauses_usage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            main_module,
            "LiveAgentAdapter",
            self._failing_adapter("USAGE_EXHAUSTED", "credits exhausted"),
        )
        exit_code = main_module.main(_run_args(tmp_path))
        assert exit_code == 2
        state = json.loads((tmp_path / "runtime" / "orchestrator_state.json").read_text())
        assert state["STATE"] == "PAUSED_USAGE"

    def test_other_gate_failure_blocks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            main_module,
            "LiveAgentAdapter",
            self._failing_adapter("MODEL_UNAVAILABLE", "Grok 4.6 not reported"),
        )
        exit_code = main_module.main(_run_args(tmp_path))
        assert exit_code == 2
        state = json.loads((tmp_path / "runtime" / "orchestrator_state.json").read_text())
        assert state["STATE"] == "BLOCKED"
