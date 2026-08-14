"""Stop/completion hook: PASS/COMPLETE claims must carry named evidence.

Reads the stop-event JSON on stdin and scans any textual payload for
``PASS``/``COMPLETE`` claims. A claim is considered evidence-backed when the
same payload also names concrete evidence (artefact paths, ledger entries,
captured outputs). When claims lack evidence, emit a ``followup_message``
asking the agent to attach evidence before stopping. Never blocks completion.
"""

from __future__ import annotations

import json
import re
import sys

CLAIM_RE = re.compile(r"\b(PASS(ED)?|COMPLETE(D)?)\b")
EVIDENCE_MARKERS = (
    re.compile(r"EV-\d{4}"),
    re.compile(r"(experiments|var|models|replays|docs|tests)/[\w./-]+"),
    re.compile(r"[\w./-]+\.(json|jsonl|log|txt|csv|zip|npz)"),
    re.compile(r"pytest|ruff|compileall|dry-run|Get-FileHash|SHA-?256", re.I),
    re.compile(r"\b[0-9]+\s+passed\b", re.I),
)


def _collect_text(value: object, parts: list[str]) -> None:
    if isinstance(value, str):
        parts.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_text(item, parts)
    elif isinstance(value, list):
        for item in value:
            _collect_text(item, parts)


def needs_evidence(event: dict) -> bool:
    parts: list[str] = []
    _collect_text(event, parts)
    text = "\n".join(parts)
    if not CLAIM_RE.search(text):
        return False
    return not any(marker.search(text) for marker in EVIDENCE_MARKERS)


def main() -> int:
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
        if not isinstance(event, dict):
            event = {}
    except (OSError, ValueError):
        event = {}
    if needs_evidence(event):
        print(
            json.dumps(
                {
                    "followup_message": (
                        "You claimed PASS/COMPLETE without named evidence. Before "
                        "stopping, attach the command run, captured output, artefact "
                        "path, or ledger entry (EV-xxxx) that backs each claim, and "
                        "update experiments/marathon/ACTIVE_STATE.json."
                    )
                },
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps({}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
