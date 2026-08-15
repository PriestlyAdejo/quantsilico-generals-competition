"""Marathon continuous-execution stop gate (QODER-CONTINUOUS-EXECUTION-2026-08-15).

Stop hook for the Marathon programme. Bounded tasks are checkpoints, not
session termination conditions: while an executable NEXT_SAFE_ACTION exists
(or canonical stages remain incomplete without a recorded hard blocker that
blocks ALL remaining work), stopping is refused and a continuation message is
injected via ``followup_message``.

Fail-safety rules:
- Malformed ACTIVE_STATE is conservative: stop is refused with a diagnostic
  asking for reconciliation (never silently claims completion).
- Hard blockers allow stop only when no independent executable work remains.
- A manual emergency stop (env MARATHON_EMERGENCY_STOP=1 or a marker file)
  always wins; the block counter also escalates to allow stop after a
  runaway threshold, so no infinite loop can form.
- The evidence gate runs first and its follow-up takes precedence.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ACTIVE_STATE_PATH = REPO / "experiments/marathon/ACTIVE_STATE.json"
RUNPOD_LEDGER_PATH = REPO / "experiments/marathon/runpod_resources.json"
EMERGENCY_MARKER = REPO / "var/marathon_takeover/EMERGENCY_STOP"
BLOCK_COUNTER_PATH = REPO / "var/marathon_takeover/stop_gate_blocks.json"
MAX_BLOCKS = 200

CANONICAL_STAGES = (
    "STAGE_1",
    "STAGE_2",
    "STAGE_3",
    "STAGE_4A",
    "STAGE_4B",
    "STAGE_5",
    "STAGE_6",
    "STAGE_7",
)
COMPLETE_TOKENS = ("COMPLETE", "MERGED", "DONE")


def _stage_done(name: str, stages: dict) -> bool:
    entry = stages.get(name)
    if isinstance(entry, dict):
        status = str(entry.get("STATUS", ""))
    else:
        status = str(entry or "")
    return any(token in status for token in COMPLETE_TOKENS)


def evaluate_stop(
    state: object,
    *,
    emergency_stop: bool = False,
    prior_blocks: int = 0,
) -> dict:
    """Decide whether the Marathon session may stop.

    Returns {"decision": "ALLOW"|"BLOCK", "reason": str, "message": str}.
    """
    if emergency_stop:
        return {
            "decision": "ALLOW",
            "reason": "MANUAL_EMERGENCY_STOP",
            "message": "Manual emergency stop acknowledged.",
        }
    if prior_blocks >= MAX_BLOCKS:
        return {
            "decision": "ALLOW",
            "reason": "RUNAWAY_BLOCK_LIMIT",
            "message": (
                f"The stop gate blocked {prior_blocks} consecutive stops. Escalating "
                "to operator review; inspect experiments/marathon/ACTIVE_STATE.json "
                "and var/marathon_takeover/stop_gate_blocks.json before resuming."
            ),
        }
    if state is None:
        return {
            "decision": "BLOCK",
            "reason": "MALFORMED_ACTIVE_STATE",
            "message": (
                "experiments/marathon/ACTIVE_STATE.json is missing or unparsable. "
                "The Marathon cannot be declared complete from unknown state. "
                "Reconcile ACTIVE_STATE (protocol section 7 key set) and continue "
                "the canonical programme."
            ),
        }
    if not isinstance(state, dict):
        return {
            "decision": "BLOCK",
            "reason": "MALFORMED_ACTIVE_STATE",
            "message": "ACTIVE_STATE is not a JSON object; reconcile it and continue.",
        }

    stages = state.get("STAGES", {})
    if not isinstance(stages, dict):
        stages = {}
    completion = state.get("COMPLETION", {})
    if not isinstance(completion, dict):
        completion = {}

    stages_done = all(_stage_done(name, stages) for name in CANONICAL_STAGES)
    final_ok = (
        bool(completion.get("FINAL_ADVERSARIAL_REVIEW"))
        and bool(completion.get("INTEGRATION_MERGED_TO_MAIN"))
        and bool(completion.get("POST_MERGE_PROOF"))
    )
    if stages_done and final_ok:
        return {
            "decision": "ALLOW",
            "reason": "MARATHON_COMPLETE",
            "message": "Canonical Marathon complete with post-merge proof. Stopping allowed.",
        }

    next_action = str(state.get("NEXT_SAFE_ACTION") or "").strip()
    blockers = state.get("BLOCKERS") or []
    if not isinstance(blockers, list):
        blockers = []
    scope_all = any(
        isinstance(item, dict)
        and str(item.get("SCOPE", "")).upper() in {"ALL", "ALL_REMAINING_WORK", "PROGRAMME"}
        for item in blockers
    )
    if scope_all:
        return {
            "decision": "ALLOW",
            "reason": "HARD_BLOCKER_ALL_WORK",
            "message": (
                "A recorded hard blocker covers all remaining work. Stopping allowed; "
                "the blocker must remain recorded in ACTIVE_STATE."
            ),
        }

    incomplete = [name for name in CANONICAL_STAGES if not _stage_done(name, stages)]
    if next_action:
        return {
            "decision": "BLOCK",
            "reason": "EXECUTABLE_NEXT_SAFE_ACTION",
            "message": (
                "The Marathon is not complete. ACTIVE_STATE.NEXT_SAFE_ACTION is: "
                f"{next_action}. Continue autonomously from that action. Do not "
                "produce the final report yet. Bounded-task completion is a "
                "checkpoint, not a session termination condition."
            ),
        }
    if incomplete:
        return {
            "decision": "BLOCK",
            "reason": "INCOMPLETE_STAGES",
            "message": (
                "Canonical Marathon stages remain incomplete: "
                f"{', '.join(incomplete)}. Reconcile ACTIVE_STATE and continue "
                "with the next dependency-safe action."
            ),
        }
    return {
        "decision": "BLOCK",
        "reason": "MALFORMED_ACTIVE_STATE",
        "message": (
            "ACTIVE_STATE shows neither full completion nor a NEXT_SAFE_ACTION. "
            "Reconcile state (including STAGES/COMPLETION fields) before stopping."
        ),
    }


def _emergency_active() -> bool:
    if os.environ.get("MARATHON_EMERGENCY_STOP") == "1":
        return True
    try:
        return EMERGENCY_MARKER.exists()
    except OSError:
        return False


def runpod_unexplained_running(repo_root: Path | None = None) -> list[str]:
    """RUNPOD-ZERO-IDLE-BURN-2026-08-15 stop-gate check (amendment section 10).

    Returns the names of ledger resources still recorded as ACTIVE-paid
    without a verified workload disposition. The hook itself cannot query
    RunPod; the resource ledger is the reconciliation surface, and session
    recovery must re-run scripts/dev/runpod_idle_watchdog.py against live
    state (amendment section 11). Missing ledger file means no declared
    paid resources (fail-open).
    """
    root = repo_root or REPO
    path = root / "experiments/marathon/runpod_resources.json"
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    problems = []
    for record in ledger.get("resources", []):
        if not isinstance(record, dict):
            continue
        status = str(record.get("status", "")).upper()
        if status in {"RUNNING", "ACTIVE_UNVERIFIED", "RUNNING_UNVERIFIED"}:
            problems.append(str(record.get("name") or record.get("pod_id")))
    return problems


def _read_prior_blocks() -> int:
    try:
        return int(json.loads(BLOCK_COUNTER_PATH.read_text(encoding="utf-8")).get("blocks", 0))
    except (OSError, ValueError, TypeError):
        return 0


def _write_prior_blocks(count: int) -> None:
    try:
        BLOCK_COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        BLOCK_COUNTER_PATH.write_text(
            json.dumps({"blocks": count}) + "\n", encoding="utf-8"
        )
    except OSError:
        pass


def main() -> int:
    # Evidence gate runs first; its follow-up takes precedence.
    try:
        from qoder_evidence_gate import needs_evidence  # type: ignore
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from evidence_gate import needs_evidence  # noqa: E402
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

    if not _emergency_active():
        unexplained = runpod_unexplained_running()
        if unexplained:
            prior = _read_prior_blocks()
            _write_prior_blocks(prior + 1)
            print(
                json.dumps(
                    {
                        "followup_message": (
                            "RUNPOD-ZERO-IDLE-BURN-2026-08-15: paid resource(s) "
                            f"{', '.join(unexplained)} are recorded RUNNING without "
                            "a verified active workload. Before stopping: run "
                            "scripts/dev/runpod_idle_watchdog.py --stop-idle, fetch "
                            "any outstanding artefacts, stop idle pods, and update "
                            "experiments/marathon/runpod_resources.json. Do not end "
                            "a session leaving unexplained paid compute running."
                        )
                    },
                    sort_keys=True,
                )
            )
            return 0

    try:
        state = json.loads(ACTIVE_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = None

    prior = _read_prior_blocks()
    outcome = evaluate_stop(
        state, emergency_stop=_emergency_active(), prior_blocks=prior
    )
    if outcome["decision"] == "ALLOW":
        _write_prior_blocks(0)
        print(json.dumps({}, sort_keys=True))
        return 0
    _write_prior_blocks(prior + 1)
    print(json.dumps({"followup_message": outcome["message"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
