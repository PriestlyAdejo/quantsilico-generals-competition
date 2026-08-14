from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tools.agentic_orchestrator.subprocess_runner import (
    classify_failure,
    run_command,
    safe_environment,
    sanitize_text,
)


def test_run_command_uses_argv_and_captures_output(tmp_path: Path) -> None:
    result = run_command(
        [sys.executable, "-c", "print('safe output')"],
        cwd=tmp_path,
        timeout_seconds=5,
    )
    assert result.ok
    assert result.stdout.strip() == "safe output"
    assert result.argv[0] == sys.executable


def test_run_command_classifies_timeout(tmp_path: Path) -> None:
    result = run_command(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        cwd=tmp_path,
        timeout_seconds=0.05,
    )
    assert result.classification == "TIMEOUT"
    assert result.returncode == 124


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("quota exceeded for account", "USAGE_EXHAUSTED"),
        ("Please log in", "AUTHENTICATION_REQUIRED"),
        ("Access is denied", "UNAVAILABLE"),
        ("ordinary crash", "PROCESS_FAILURE"),
    ],
)
def test_failure_classification(text: str, expected: str) -> None:
    assert classify_failure(1, "", text) == expected


def test_sanitize_text_redacts_credentials() -> None:
    value = sanitize_text("Authorization: Bearer abc.def\napi_key=super-secret")
    assert "abc.def" not in value
    assert "super-secret" not in value
    assert "[REDACTED]" in value


def test_safe_environment_excludes_sensitive_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE_API_KEY", "secret")
    monkeypatch.setenv("ORCHESTRATOR_SAFE_VALUE", "visible")
    env = safe_environment()
    assert "EXAMPLE_API_KEY" not in env
    assert env["ORCHESTRATOR_SAFE_VALUE"] == "visible"
    with pytest.raises(ValueError, match="sensitive"):
        safe_environment({"NEW_TOKEN": "secret"})


def test_missing_executable_is_unavailable(tmp_path: Path) -> None:
    missing = tmp_path / ("missing.exe" if os.name == "nt" else "missing")
    result = run_command([str(missing), "--version"], cwd=tmp_path)
    assert result.classification == "UNAVAILABLE"
