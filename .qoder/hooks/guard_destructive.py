"""PreToolUse guardrail: block destructive git and evidence-directory deletion.

Reads a hook event JSON object on stdin (fields best-effort: ``command`` or
``tool_input``), writes a permission decision JSON to stdout.

Decision contract:
- ``deny``  for destructive git (force push, hard reset, destructive clean,
  history rewrite) and deletion under protected evidence directories.
- ``ask``   for deletion of unmerged branches/worktrees (may be legitimate).
- ``allow`` otherwise.

Exit code is always 0 when a decision was produced; the JSON payload carries
the verdict. Fail-open on unreadable input (the rule/skill layer still
applies), so a malformed event never wedges the harness.
"""

from __future__ import annotations

import json
import re
import sys

DENY_PATTERNS = (
    re.compile(r"git\s+push\s+[^|;&]*(-f\b|--force\b|--mirror|--delete)"),
    re.compile(r"git\s+reset\s+[^|;&]*--hard"),
    re.compile(r"git\s+clean\s+[^|;&]*-[a-z]*f"),
    re.compile(r"git\s+push\s+[^|;&]*--force-with-lease"),
    re.compile(r"git\s+filter-branch|git\s+filter-repo"),
    re.compile(r"Remove-Item\s+[^|;&]*(experiments|models|replays)[/\\]", re.I),
    re.compile(r"\brm\s+(-[a-z]*[rf][a-z]*\s+)+[^|;&]*(experiments|models|replays)/", re.I),
    re.compile(r"rmdir\s+/s[^|;&]*(experiments|models|replays)", re.I),
)

ASK_PATTERNS = (
    re.compile(r"git\s+branch\s+-D\b"),
    re.compile(r"git\s+worktree\s+remove"),
    re.compile(r"git\s+push\s+[^|;&]*:\s*\S+"),
)


def _extract_command(event: dict) -> str:
    if isinstance(event.get("command"), str):
        return event["command"]
    tool_input = event.get("tool_input")
    if isinstance(tool_input, dict):
        for key in ("command", "cmd", "script"):
            if isinstance(tool_input.get(key), str):
                return tool_input[key]
    return ""


def decide(event: dict) -> dict:
    command = _extract_command(event)
    for pattern in DENY_PATTERNS:
        if pattern.search(command):
            return {
                "permission": "deny",
                "agent_message": (
                    "Blocked by marathon-evidence-contract hook: destructive git or "
                    "evidence-directory deletion requires explicit human authorization."
                ),
            }
    for pattern in ASK_PATTERNS:
        if pattern.search(command):
            return {
                "permission": "ask",
                "user_message": (
                    "This command deletes branches/worktrees or pushes deletions. "
                    "Confirm uniqueness is proven before proceeding."
                ),
            }
    return {"permission": "allow"}


def main() -> int:
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
        if not isinstance(event, dict):
            event = {}
    except (OSError, ValueError):
        event = {}
    print(json.dumps(decide(event), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
