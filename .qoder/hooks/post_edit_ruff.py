"""PostToolUse hook: narrow ruff lint after edits to orchestrator sources.

Reads a hook event JSON object on stdin. If the edited path is under
``tools/agentic_orchestrator`` or an orchestrator test module, runs ruff on
that file only and reports findings as ``additional_context``. Never blocks.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WATCHED_PREFIXES = ("tools/agentic_orchestrator", "tests/unit/test_agentic_orchestrator")


def _extract_path(event: dict) -> str:
    for key in ("file_path", "path"):
        if isinstance(event.get(key), str):
            return event[key]
    tool_input = event.get("tool_input")
    if isinstance(tool_input, dict):
        for key in ("file_path", "path", "target_file"):
            if isinstance(tool_input.get(key), str):
                return tool_input[key]
    return ""


def _watched(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return any(prefix in normalized for prefix in WATCHED_PREFIXES)


def main() -> int:
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
        if not isinstance(event, dict):
            event = {}
    except (OSError, ValueError):
        event = {}
    path = _extract_path(event)
    if not path or not _watched(path) or not Path(path).exists():
        print(json.dumps({}))
        return 0
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "ruff", "check", path],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        print(json.dumps({}))
        return 0
    if completed.returncode == 0:
        print(json.dumps({}))
        return 0
    output = (completed.stdout or completed.stderr).strip()[:4000]
    print(
        json.dumps(
            {
                "additional_context": (
                    f"Ruff found issues in {path} after your edit; fix them before "
                    f"continuing:\n{output}"
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
