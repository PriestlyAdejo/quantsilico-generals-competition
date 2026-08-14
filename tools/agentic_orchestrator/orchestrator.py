"""Deterministic, crash-recoverable local agentic supervisor."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .schemas import (
    SCHEMA_VERSION,
    ReviewVerdict,
    SchemaError,
    State,
    report_json_schema,
    review_json_schema,
    task_json_schema,
    validate_report,
    validate_review,
    validate_task,
)
from .subprocess_runner import (
    CommandResult,
    ToolProbe,
    probe_codex,
    probe_cursor,
    run_command,
)

PLAN_ID = "MARATHON_REDESIGN_LOCKED_V1"
MAX_REPAIR_ITERATIONS = 3
SAME_PROBLEM_ESCALATION = 2

LEGAL_TRANSITIONS: dict[State, set[State]] = {
    State.IDLE: {
        State.ARCHITECTING,
        State.BLOCKED,
        State.PAUSED_HUMAN_BOUNDARY,
        State.PAUSED_USAGE,
    },
    State.ARCHITECTING: {
        State.IMPLEMENTING,
        State.BLOCKED,
        State.PAUSED_USAGE,
        State.PAUSED_HUMAN_BOUNDARY,
        State.FAILED,
    },
    State.IMPLEMENTING: {
        State.VALIDATING,
        State.BLOCKED,
        State.PAUSED_USAGE,
        State.PAUSED_HUMAN_BOUNDARY,
        State.FAILED,
    },
    State.VALIDATING: {State.REVIEWING, State.BLOCKED, State.FAILED},
    State.REVIEWING: {
        State.ACCEPTED,
        State.REPAIRING,
        State.BLOCKED,
        State.PAUSED_USAGE,
        State.PAUSED_HUMAN_BOUNDARY,
        State.FAILED,
    },
    State.REPAIRING: {
        State.VALIDATING,
        State.BLOCKED,
        State.PAUSED_USAGE,
        State.PAUSED_HUMAN_BOUNDARY,
        State.FAILED,
    },
    State.ACCEPTED: {State.IDLE, State.ARCHITECTING, State.PAUSED_HUMAN_BOUNDARY},
    State.BLOCKED: {State.IDLE},
    State.PAUSED_USAGE: {State.IDLE},
    State.PAUSED_HUMAN_BOUNDARY: {State.IDLE},
    State.FAILED: {State.IDLE},
}


class OrchestratorError(RuntimeError):
    """Base orchestration error."""


class ConcurrentWriterError(OrchestratorError):
    """Raised when another supervisor owns the runtime lock."""


class AgentInvocationError(OrchestratorError):
    """Raised for a classified external agent failure."""

    def __init__(self, role: str, result: CommandResult) -> None:
        super().__init__(
            f"{role} failed: {result.classification}: {result.stderr or result.stdout}"
        )
        self.role = role
        self.result = result


@dataclass(frozen=True)
class RuntimePaths:
    root: Path

    @property
    def state(self) -> Path:
        return self.root / "orchestrator_state.json"

    @property
    def task(self) -> Path:
        return self.root / "current_task.json"

    @property
    def report(self) -> Path:
        return self.root / "implementation_report.json"

    @property
    def review(self) -> Path:
        return self.root / "review.json"

    @property
    def events(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def lock(self) -> Path:
        return self.root / "writer.lock"

    @property
    def schemas(self) -> Path:
        return self.root / "schemas"


class AgentAdapter(Protocol):
    def architect(
        self, context: dict[str, Any], *, correction: str | None = None
    ) -> dict[str, Any]: ...

    def implement(
        self, task: dict[str, Any], *, repair: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...

    def review(
        self, task: dict[str, Any], report: dict[str, Any], *, iteration: int
    ) -> dict[str, Any]: ...


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OrchestratorError(f"expected object in {path}")
    return value


class WriterLock(AbstractContextManager["WriterLock"]):
    """Windows-safe exclusive lock; stale locks are never removed automatically."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def __enter__(self) -> WriterLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            owner = "UNKNOWN"
            try:
                owner = self.path.read_text(encoding="utf-8").strip()
            except OSError:
                pass
            raise ConcurrentWriterError(f"writer lock already exists: {owner}") from exc
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"PID": os.getpid(), "ACQUIRED_AT_UTC": utc_now()}, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.acquired = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False


def initial_state() -> dict[str, Any]:
    return {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "PLAN_ID": PLAN_ID,
        "STATE": State.IDLE.value,
        "TASK_ID": None,
        "REPAIR_ITERATION": 0,
        "SAME_PROBLEM_FAILURES": 0,
        "LAST_PROBLEM_FINGERPRINT": None,
        "BASE_COMMIT": None,
        "ACCEPTED_HEAD": None,
        "PAUSE_REASON": None,
        "RESUME_STATE": None,
        "TRANSITION_SEQUENCE": 0,
        "UPDATED_AT_UTC": utc_now(),
    }


class Orchestrator:
    def __init__(self, *, repo: Path, runtime: RuntimePaths) -> None:
        self.repo = repo.resolve()
        self.runtime = runtime
        self.runtime.root.mkdir(parents=True, exist_ok=True)
        self._write_schemas()
        if not self.runtime.state.exists():
            atomic_write_json(self.runtime.state, initial_state())
        self.state = self._load_state()

    def _write_schemas(self) -> None:
        atomic_write_json(self.runtime.schemas / "task.schema.json", task_json_schema())
        atomic_write_json(
            self.runtime.schemas / "implementation_report.schema.json", report_json_schema()
        )
        atomic_write_json(self.runtime.schemas / "review.schema.json", review_json_schema())

    def _load_state(self) -> dict[str, Any]:
        state = read_json(self.runtime.state)
        required = set(initial_state())
        if set(state) != required:
            raise OrchestratorError(
                f"orchestrator state keys mismatch; missing={sorted(required - state.keys())}, "
                f"unknown={sorted(state.keys() - required)}"
            )
        if state["SCHEMA_VERSION"] != SCHEMA_VERSION or state["PLAN_ID"] != PLAN_ID:
            raise OrchestratorError("orchestrator state identity mismatch")
        State(state["STATE"])
        return state

    def reload(self) -> dict[str, Any]:
        self.state = self._load_state()
        return self.state

    def transition(self, target: State, *, reason: str, **updates: Any) -> None:
        current = State(self.state["STATE"])
        if target not in LEGAL_TRANSITIONS[current]:
            raise OrchestratorError(f"illegal transition {current.value} -> {target.value}")
        before = current.value
        self.state.update(updates)
        self.state["STATE"] = target.value
        self.state["TRANSITION_SEQUENCE"] += 1
        self.state["UPDATED_AT_UTC"] = utc_now()
        atomic_write_json(self.runtime.state, self.state)
        append_event(
            self.runtime.events,
            {
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "SEQUENCE": self.state["TRANSITION_SEQUENCE"],
                "FROM": before,
                "TO": target.value,
                "REASON": reason,
                "TASK_ID": self.state["TASK_ID"],
                "AT_UTC": self.state["UPDATED_AT_UTC"],
            },
        )

    def pause_usage(self, reason: str) -> None:
        current = State(self.state["STATE"])
        if State.PAUSED_USAGE not in LEGAL_TRANSITIONS[current]:
            raise OrchestratorError(f"cannot pause usage from {current.value}")
        self.transition(
            State.PAUSED_USAGE,
            reason="usage unavailable",
            PAUSE_REASON=reason,
            RESUME_STATE=State.IDLE.value,
        )

    def pause_human_boundary(self, reason: str) -> None:
        current = State(self.state["STATE"])
        if State.PAUSED_HUMAN_BOUNDARY not in LEGAL_TRANSITIONS[current]:
            raise OrchestratorError(f"cannot pause human boundary from {current.value}")
        self.transition(
            State.PAUSED_HUMAN_BOUNDARY,
            reason="human boundary",
            PAUSE_REASON=reason,
            RESUME_STATE=State.IDLE.value,
        )

    def block(self, reason: str) -> None:
        current = State(self.state["STATE"])
        if State.BLOCKED not in LEGAL_TRANSITIONS[current]:
            raise OrchestratorError(f"cannot block from {current.value}: {reason}")
        self.transition(State.BLOCKED, reason="safe block", PAUSE_REASON=reason)

    def resume(self) -> None:
        current = State(self.state["STATE"])
        if current not in {
            State.BLOCKED,
            State.PAUSED_USAGE,
            State.PAUSED_HUMAN_BOUNDARY,
            State.FAILED,
            State.ACCEPTED,
        }:
            raise OrchestratorError(f"cannot resume from {current.value}")
        self.transition(
            State.IDLE,
            reason="operator resume",
            PAUSE_REASON=None,
            RESUME_STATE=None,
            TASK_ID=None,
            REPAIR_ITERATION=0,
            SAME_PROBLEM_FAILURES=0,
            LAST_PROBLEM_FINGERPRINT=None,
        )

    def status(self) -> dict[str, Any]:
        state = dict(self.reload())
        state["RUNTIME_DIR"] = str(self.runtime.root.resolve())
        state["TASK_PRESENT"] = self.runtime.task.is_file()
        state["REPORT_PRESENT"] = self.runtime.report.is_file()
        state["REVIEW_PRESENT"] = self.runtime.review.is_file()
        return state

    def tooling(self) -> dict[str, Any]:
        codex = probe_codex(self.repo)
        cursor = probe_cursor(self.repo)
        return {"CODEX": codex.as_dict(), "CURSOR_AGENT": cursor.as_dict()}

    def run_once(self, adapter: AgentAdapter) -> dict[str, Any]:
        with WriterLock(self.runtime.lock):
            self.reload()
            if State(self.state["STATE"]) != State.IDLE:
                raise OrchestratorError(f"run requires IDLE, got {self.state['STATE']}")
            base = _git(self.repo, "rev-parse", "HEAD")
            self.transition(State.ARCHITECTING, reason="request one bounded task", BASE_COMMIT=base)
            task = self._architect_with_one_correction(adapter)
            validate_task(task)
            atomic_write_json(self.runtime.task, task)
            self.state["TASK_ID"] = task["TASK_ID"]
            atomic_write_json(self.runtime.state, self.state)
            if any(item.startswith("REQUIRES_NOW:") for item in task["HUMAN_BOUNDARY"]):
                self.pause_human_boundary("; ".join(task["HUMAN_BOUNDARY"]))
                return self.status()
            self.transition(State.IMPLEMENTING, reason="schema-valid bounded task")
            report = adapter.implement(task)
            return self._process_report_and_reviews(adapter, task, report)

    def _architect_with_one_correction(self, adapter: AgentAdapter) -> dict[str, Any]:
        context = self._architect_context()
        try:
            return validate_task(adapter.architect(context))
        except SchemaError as first:
            correction = (
                f"Your previous response violated the task schema: {first}. Return only valid JSON."
            )
            try:
                return validate_task(adapter.architect(context, correction=correction))
            except SchemaError as second:
                self.transition(
                    State.BLOCKED, reason="architect schema invalid twice", PAUSE_REASON=str(second)
                )
                raise OrchestratorError("architect returned invalid schema twice") from second

    def _process_report_and_reviews(
        self, adapter: AgentAdapter, task: dict[str, Any], report: dict[str, Any]
    ) -> dict[str, Any]:
        while True:
            report = validate_report(report, task=task)
            atomic_write_json(self.runtime.report, report)
            self.transition(State.VALIDATING, reason="implementation report schema valid")
            self._validate_repository_report(task, report)
            self.transition(State.REVIEWING, reason="implementation evidence ready")
            review = validate_review(
                adapter.review(task, report, iteration=int(self.state["REPAIR_ITERATION"])),
                task=task,
            )
            atomic_write_json(self.runtime.review, review)
            verdict = ReviewVerdict(review["VERDICT"])
            if verdict == ReviewVerdict.ACCEPT:
                self.transition(
                    State.ACCEPTED,
                    reason="independent review accepted",
                    ACCEPTED_HEAD=report["HEAD_COMMIT"],
                    PAUSE_REASON=None,
                )
                return self.status()
            if verdict == ReviewVerdict.HUMAN_BOUNDARY:
                self.pause_human_boundary("; ".join(review["FINDINGS"] + review["REQUIRED_FIXES"]))
                return self.status()
            if verdict in {ReviewVerdict.RETHINK, ReviewVerdict.INSUFFICIENT_EVIDENCE}:
                self.transition(
                    State.BLOCKED,
                    reason=verdict.value,
                    PAUSE_REASON="; ".join(review["FAILED_CRITERIA"] + review["REQUIRED_FIXES"]),
                )
                return self.status()
            self._prepare_repair(review)
            report = adapter.implement(task, repair=review)

    def _prepare_repair(self, review: dict[str, Any]) -> None:
        iteration = int(self.state["REPAIR_ITERATION"]) + 1
        fingerprint = hashlib.sha256(
            json.dumps(review["REQUIRED_FIXES"], sort_keys=True).encode("utf-8")
        ).hexdigest()
        repeated = int(self.state["SAME_PROBLEM_FAILURES"])
        if fingerprint == self.state["LAST_PROBLEM_FINGERPRINT"]:
            repeated += 1
        else:
            repeated = 1
        if iteration > MAX_REPAIR_ITERATIONS or repeated >= SAME_PROBLEM_ESCALATION:
            self.transition(
                State.BLOCKED,
                reason="repair escalation threshold",
                REPAIR_ITERATION=iteration,
                SAME_PROBLEM_FAILURES=repeated,
                LAST_PROBLEM_FINGERPRINT=fingerprint,
                PAUSE_REASON="architect escalation required after repeated substantive failure",
            )
            raise OrchestratorError("repair escalation threshold reached")
        self.transition(
            State.REPAIRING,
            reason="review requested bounded repair",
            REPAIR_ITERATION=iteration,
            SAME_PROBLEM_FAILURES=repeated,
            LAST_PROBLEM_FINGERPRINT=fingerprint,
        )

    def _validate_repository_report(self, task: dict[str, Any], report: dict[str, Any]) -> None:
        current_head = _git(self.repo, "rev-parse", "HEAD")
        if report["HEAD_COMMIT"] not in {current_head, "WORKTREE"}:
            raise OrchestratorError("implementation report HEAD_COMMIT does not match repository")
        if report["BASE_COMMIT"] != self.state["BASE_COMMIT"]:
            raise OrchestratorError("implementation report BASE_COMMIT mismatch")
        if report["STATUS"] == "COMPLETE" and report["TESTS_FAILED"]:
            raise OrchestratorError("COMPLETE report contains failed tests")
        if not task.get("VERIFICATION_ONLY", False) and report["STATUS"] == "COMPLETE":
            active_state = "experiments/marathon/ACTIVE_STATE.json"
            normalized_files = {item.replace("\\", "/") for item in report["FILES_CHANGED"]}
            if active_state not in normalized_files:
                raise OrchestratorError(
                    "COMPLETE implementation did not report the canonical ACTIVE_STATE update"
                )
            actual = set(_git_lines(self.repo, "status", "--porcelain=v1", "-uall"))
            if not actual and report["HEAD_COMMIT"] == self.state["BASE_COMMIT"]:
                raise OrchestratorError(
                    "COMPLETE implementation produced no commit or worktree diff"
                )

    def _architect_context(self) -> dict[str, Any]:
        active = read_json(self.repo / "experiments" / "marathon" / "ACTIVE_STATE.json")
        return {
            "PLAN_ID": PLAN_ID,
            "ACTIVE_STATE": active,
            "GIT_HEAD": _git(self.repo, "rev-parse", "HEAD"),
            "GIT_STATUS": _git_lines(self.repo, "status", "--porcelain=v1", "-uall"),
            "PREVIOUS_REVIEW": read_json(self.runtime.review)
            if self.runtime.review.exists()
            else None,
            "AUTHORITY_PATHS": [
                "AGENTS.md",
                "docs/marathon/EXECUTION_PLAN.md",
                "docs/marathon/AGENTIC_EXECUTION_PROTOCOL.md",
                "docs/marathon/DESIGN_AUTHORITY.md",
                "docs/marathon/EVIDENCE_LEDGER.md",
                "docs/marathon/CONTROL_ENGINEERING.md",
                "configs/marathon/programme.yaml",
                "experiments/marathon/ACTIVE_STATE.json",
            ],
        }

    def dry_run(self) -> dict[str, Any]:
        adapter = DryRunAdapter(base_commit=_git(self.repo, "rev-parse", "HEAD"))
        result = self.run_once(adapter)
        restarted = Orchestrator(repo=self.repo, runtime=self.runtime)
        if restarted.state["STATE"] != State.ACCEPTED:
            raise OrchestratorError("restart did not recover ACCEPTED state")
        restarted.pause_human_boundary("SIMULATED_PAID_RESOURCE_BOUNDARY")
        restarted.resume()
        restarted.pause_usage("SIMULATED_CURSOR_QUOTA_EXHAUSTION")
        restarted.resume()
        final = restarted.status()
        final["DRY_RUN_PROOFS"] = {
            "ARCHITECT_TASK_SCHEMA": True,
            "IMPLEMENTER_PROPOSAL_AND_REPAIR": True,
            "FRESH_REVIEW_FIX_LOOP": True,
            "ACCEPT_ADVANCE": True,
            "RESTART_RECOVERY": True,
            "HUMAN_BOUNDARY_PAUSE": True,
            "USAGE_PAUSE": True,
            "NO_REPOSITORY_EDITS": True,
        }
        final["ACCEPTED_RESULT"] = result
        return final


class DryRunAdapter:
    """Deterministic in-process agents used only to prove supervisor semantics."""

    def __init__(self, *, base_commit: str) -> None:
        self.base_commit = base_commit
        self.architect_calls = 0
        self.implement_calls = 0
        self.review_calls = 0

    def architect(
        self, context: dict[str, Any], *, correction: str | None = None
    ) -> dict[str, Any]:
        self.architect_calls += 1
        return {
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "TASK_ID": "DRY_RUN_001",
            "PLAN_STAGE": "STAGE_0B",
            "GOAL": "Prove deterministic orchestration without editing the repository.",
            "FILES_OR_AREAS": ["var/agentic/dry-run"],
            "REPOSITORY_FACTS": [f"HEAD={context['GIT_HEAD']}"],
            "IMPLEMENTATION_SPEC": "Return a synthetic proposal and then a repaired report.",
            "ACCEPTANCE_CRITERIA": ["State and records persist", "Review repair loop terminates"],
            "TESTS_REQUIRED": ["synthetic_state_transition_test"],
            "FORBIDDEN_ACTIONS": ["repository edits", "external service mutation"],
            "HUMAN_BOUNDARY": [],
            "EXPECTED_OUTPUT": "Schema-valid synthetic implementation report.",
            "VERIFICATION_ONLY": True,
        }

    def implement(
        self, task: dict[str, Any], *, repair: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.implement_calls += 1
        passed = repair is not None
        return {
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "STATUS": "COMPLETE" if passed else "PARTIAL",
            "TASK_ID": task["TASK_ID"],
            "BASE_COMMIT": self.base_commit,
            "HEAD_COMMIT": self.base_commit,
            "FILES_CHANGED": [],
            "TESTS_RUN": ["synthetic_state_transition_test"],
            "TESTS_PASSED": ["synthetic_state_transition_test"] if passed else [],
            "TESTS_FAILED": [] if passed else ["synthetic_requires_one_repair"],
            "EVIDENCE": ["dry-run adapter; no repository edit"],
            "KNOWN_LIMITATIONS": ["No live model invocation"],
            "PLAN_CONFLICTS": [],
            "NEXT_SAFE_ACTION": "Review the synthetic report.",
        }

    def review(
        self, task: dict[str, Any], report: dict[str, Any], *, iteration: int
    ) -> dict[str, Any]:
        self.review_calls += 1
        fixed = report["STATUS"] == "COMPLETE"
        return {
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "VERDICT": "ACCEPT" if fixed else "FIX_FIRST",
            "TASK_ID": task["TASK_ID"],
            "FINDINGS": ["Synthetic report is schema-valid."],
            "FAILED_CRITERIA": [] if fixed else ["Repair path not yet exercised"],
            "REQUIRED_FIXES": [] if fixed else ["Exercise the deterministic repair path"],
            "OPTIONAL_IMPROVEMENTS": [],
            "EVIDENCE": [f"review iteration {iteration}"],
        }


class LiveAgentAdapter:
    """CLI-backed adapter. It refuses to run until exact tooling gates pass."""

    def __init__(
        self, *, repo: Path, runtime: RuntimePaths, requested_cursor_identity: str = "Grok 4.6"
    ) -> None:
        self.repo = repo
        self.runtime = runtime
        self.codex: ToolProbe = probe_codex(repo)
        self.cursor: ToolProbe = probe_cursor(repo)
        if self.codex.status != "READY":
            raise OrchestratorError(f"Codex tooling not ready: {self.codex.detail}")
        if self.cursor.status != "READY":
            raise OrchestratorError(f"Cursor Agent tooling not ready: {self.cursor.detail}")
        self.cursor_model = _resolve_model(self.cursor.models, requested_cursor_identity)
        if self.cursor_model is None:
            raise OrchestratorError(
                f"requested Cursor model {requested_cursor_identity!r} is unavailable; "
                f"reported models={list(self.cursor.models)!r}"
            )

    def architect(
        self, context: dict[str, Any], *, correction: str | None = None
    ) -> dict[str, Any]:
        prompt = {
            "ROLE": "CODEX_ARCHITECT_READ_ONLY",
            "INSTRUCTION": (
                "Identify exactly one bounded next task and return only schema-valid JSON."
            ),
            "CONTEXT": context,
            "SCHEMA_CORRECTION": correction,
        }
        return self._codex_json("architect", prompt, self.runtime.schemas / "task.schema.json")

    def implement(
        self, task: dict[str, Any], *, repair: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if self.cursor.executable is None:
            raise OrchestratorError("Cursor executable unavailable")
        prompt = json.dumps(
            {
                "ROLE": "CURSOR_IMPLEMENTER_SOLE_WRITER",
                "TASK": task,
                "REPAIR_REVIEW": repair,
                "AUTHORITY_PATHS": [
                    "AGENTS.md",
                    "docs/marathon/EXECUTION_PLAN.md",
                    "docs/marathon/AGENTIC_EXECUTION_PROTOCOL.md",
                    "experiments/marathon/ACTIVE_STATE.json",
                ],
                "OUTPUT": "Return only the required implementation-report JSON object.",
            },
            sort_keys=True,
        )
        argv = [
            self.cursor.executable,
            "-p",
            "--output-format",
            "text",
            "--model",
            self.cursor_model,
            prompt,
        ]
        result = run_command(argv, cwd=self.repo, timeout_seconds=1800)
        if not result.ok:
            raise AgentInvocationError("CURSOR_IMPLEMENTER", result)
        return _parse_json_object(result.stdout)

    def review(
        self, task: dict[str, Any], report: dict[str, Any], *, iteration: int
    ) -> dict[str, Any]:
        diff = _git(self.repo, "diff", f"{report['BASE_COMMIT']}..HEAD")
        prompt = {
            "ROLE": "CODEX_REVIEWER_READ_ONLY_FRESH_CONTEXT",
            "TASK": task,
            "IMPLEMENTATION_REPORT": report,
            "BASE_TO_HEAD_DIFF": diff,
            "REVIEW_ITERATION": iteration,
            "INSTRUCTION": "Review only supplied evidence and return only schema-valid JSON.",
        }
        return self._codex_json("review", prompt, self.runtime.schemas / "review.schema.json")

    def _codex_json(self, role: str, prompt: dict[str, Any], schema: Path) -> dict[str, Any]:
        if self.codex.executable is None:
            raise OrchestratorError("Codex executable unavailable")
        output = self.runtime.root / f"{role}_last_message.json"
        argv = [
            self.codex.executable,
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "-C",
            str(self.repo),
            "--output-schema",
            str(schema),
            "--output-last-message",
            str(output),
            "-",
        ]
        result = run_command(
            argv,
            cwd=self.repo,
            stdin_text=json.dumps(prompt, sort_keys=True),
            timeout_seconds=900,
        )
        if not result.ok:
            raise AgentInvocationError(f"CODEX_{role.upper()}", result)
        return read_json(output)


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"agent output is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaError("agent output must be a JSON object")
    return value


def _resolve_model(models: tuple[str, ...], requested: str) -> str | None:
    requested_normalized = " ".join(requested.casefold().split())
    for model in models:
        if " ".join(model.casefold().split()) == requested_normalized:
            return model
    return None


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise OrchestratorError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _git_lines(repo: Path, *args: str) -> list[str]:
    output = _git(repo, *args)
    return output.splitlines() if output else []
