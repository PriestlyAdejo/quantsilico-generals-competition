"""Tests for the deterministic Qoder guardrail hooks (.qoder/hooks/*.py).

Each hook is exercised through its real stdin/stdout contract so the hook
behaviour itself is evidence, not prose.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOOKS = REPO / ".qoder" / "hooks"


def run_hook(name: str, event: object) -> dict:
    completed = subprocess.run(
        [sys.executable, str(HOOKS / name)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


class TestGuardDestructive:
    def test_force_push_denied(self) -> None:
        result = run_hook("guard_destructive.py", {"command": "git push --force origin main"})
        assert result["permission"] == "deny"

    def test_hard_reset_denied(self) -> None:
        result = run_hook("guard_destructive.py", {"command": "git reset --hard HEAD~3"})
        assert result["permission"] == "deny"

    def test_destructive_clean_denied(self) -> None:
        result = run_hook("guard_destructive.py", {"command": "git clean -fd experiments/"})
        assert result["permission"] == "deny"

    def test_evidence_directory_deletion_denied(self) -> None:
        result = run_hook(
            "guard_destructive.py",
            {"command": "Remove-Item -Recurse experiments/manifests"},
        )
        assert result["permission"] == "deny"

    def test_history_rewrite_denied(self) -> None:
        result = run_hook("guard_destructive.py", {"command": "git filter-branch --all"})
        assert result["permission"] == "deny"

    def test_branch_force_delete_asks(self) -> None:
        result = run_hook("guard_destructive.py", {"command": "git branch -D old-work"})
        assert result["permission"] == "ask"

    def test_ordinary_command_allowed(self) -> None:
        result = run_hook(
            "guard_destructive.py",
            {"command": "git push origin research/phase9g-competition-native-jax-preovernight-v1"},
        )
        assert result["permission"] == "allow"

    def test_tool_input_command_shape(self) -> None:
        result = run_hook(
            "guard_destructive.py",
            {"tool_input": {"command": "git reset --hard"}},
        )
        assert result["permission"] == "deny"

    def test_malformed_input_fails_open(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(HOOKS / "guard_destructive.py")],
            input="{not json",
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0
        assert json.loads(completed.stdout)["permission"] == "allow"


class TestPostEditRuff:
    def test_watched_clean_file_emits_no_context(self) -> None:
        result = run_hook(
            "post_edit_ruff.py",
            {"file_path": str(REPO / "tools" / "agentic_orchestrator" / "schemas.py")},
        )
        assert result == {}

    def test_unwatched_file_ignored(self) -> None:
        result = run_hook("post_edit_ruff.py", {"file_path": str(REPO / "README.md")})
        assert result == {}


class TestEvidenceGate:
    def test_claim_without_evidence_triggers_followup(self) -> None:
        result = run_hook("evidence_gate.py", {"last_message": "All gates PASS, stage COMPLETE."})
        assert "followup_message" in result

    def test_claim_with_evidence_passes(self) -> None:
        result = run_hook(
            "evidence_gate.py",
            {
                "last_message": (
                    "Gate PASS: 41 passed in pytest; evidence recorded in EV-0008 and "
                    "docs/marathon/EVIDENCE_LEDGER.md."
                )
            },
        )
        assert result == {}

    def test_no_claim_passes(self) -> None:
        result = run_hook("evidence_gate.py", {"last_message": "Continuing to the next stage."})
        assert result == {}
