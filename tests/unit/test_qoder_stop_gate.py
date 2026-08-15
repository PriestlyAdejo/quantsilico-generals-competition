"""Deterministic tests for the Marathon continuous-execution stop gate.

Scenarios per QODER-CONTINUOUS-EXECUTION-2026-08-15:
1. Stage 2 IN_PROGRESS + NEXT_SAFE_ACTION -> stop blocked, continue injected.
2. All canonical stages COMPLETE + post-merge proof -> stop allowed.
3. Hard blocker covering ALL remaining work -> stop allowed.
4. Hard blocker + independent executable work remains -> stop blocked.
5. Missing/malformed ACTIVE_STATE -> conservative block with diagnostic.
6. Evidence-gate failure retains precedence at stop.
7. Manual emergency stop remains possible; runaway counter escalates.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HOOKS = REPO / ".qoder" / "hooks"

sys.path.insert(0, str(HOOKS))
import marathon_stop_gate as gate  # noqa: E402


def _stages(**overrides: str) -> dict:
    base = {
        name: {"STATUS": "COMPLETE_MERGED"} for name in gate.CANONICAL_STAGES
    }
    base.update(overrides)
    return base


def _completion(done: bool = True) -> dict:
    return {
        "FINAL_ADVERSARIAL_REVIEW": done,
        "INTEGRATION_MERGED_TO_MAIN": done,
        "POST_MERGE_PROOF": done,
    }


def test_stage2_in_progress_with_next_action_blocks_stop() -> None:
    state = {
        "STAGES": _stages(STAGE_2={"STATUS": "IN_PROGRESS"}, STAGE_3={"STATUS": "PENDING"}),
        "COMPLETION": _completion(False),
        "NEXT_SAFE_ACTION": "Package the baseline checkpoint as a protocol agent.",
        "BLOCKERS": [],
    }
    outcome = gate.evaluate_stop(state)
    assert outcome["decision"] == "BLOCK"
    assert outcome["reason"] == "EXECUTABLE_NEXT_SAFE_ACTION"
    assert "Package the baseline checkpoint" in outcome["message"]
    assert "Do not" in outcome["message"] and "final report" in outcome["message"]


def test_full_completion_allows_stop() -> None:
    state = {
        "STAGES": _stages(),
        "COMPLETION": _completion(True),
        "NEXT_SAFE_ACTION": "",
        "BLOCKERS": [],
    }
    outcome = gate.evaluate_stop(state)
    assert outcome["decision"] == "ALLOW"
    assert outcome["reason"] == "MARATHON_COMPLETE"


def test_completion_without_post_merge_proof_still_blocks() -> None:
    state = {
        "STAGES": _stages(),
        "COMPLETION": _completion(False),
        "NEXT_SAFE_ACTION": "",
        "BLOCKERS": [],
    }
    outcome = gate.evaluate_stop(state)
    assert outcome["decision"] == "BLOCK"


def test_hard_blocker_covering_all_work_allows_stop() -> None:
    state = {
        "STAGES": _stages(STAGE_4A={"STATUS": "PENDING"}),
        "COMPLETION": _completion(False),
        "NEXT_SAFE_ACTION": "",
        "BLOCKERS": [{"ID": "X", "SCOPE": "ALL", "DETAIL": "unrecoverable"}],
    }
    outcome = gate.evaluate_stop(state)
    assert outcome["decision"] == "ALLOW"
    assert outcome["reason"] == "HARD_BLOCKER_ALL_WORK"


def test_hard_blocker_with_independent_work_blocks_stop() -> None:
    state = {
        "STAGES": _stages(STAGE_4A={"STATUS": "PENDING"}),
        "COMPLETION": _completion(False),
        "NEXT_SAFE_ACTION": "Continue Stage 3 registry work that does not need the blocker.",
        "BLOCKERS": [
            {
                "ID": "CURSOR_AGENT_CLI_UNAVAILABLE",
                "SCOPE": "LIVE_ORCHESTRATION_ACCEPTANCE_ONLY",
                "DETAIL": "external gate",
            }
        ],
    }
    outcome = gate.evaluate_stop(state)
    assert outcome["decision"] == "BLOCK"
    assert outcome["reason"] == "EXECUTABLE_NEXT_SAFE_ACTION"


def test_missing_active_state_blocks_conservatively() -> None:
    outcome = gate.evaluate_stop(None)
    assert outcome["decision"] == "BLOCK"
    assert outcome["reason"] == "MALFORMED_ACTIVE_STATE"
    assert "Reconcile" in outcome["message"]


def test_non_object_active_state_blocks_conservatively() -> None:
    outcome = gate.evaluate_stop(["not", "a", "mapping"])
    assert outcome["decision"] == "BLOCK"
    assert outcome["reason"] == "MALFORMED_ACTIVE_STATE"


def test_incomplete_stages_without_next_action_block() -> None:
    state = {
        "STAGES": _stages(STAGE_5={"STATUS": "PENDING"}),
        "COMPLETION": _completion(False),
        "NEXT_SAFE_ACTION": "",
        "BLOCKERS": [],
    }
    outcome = gate.evaluate_stop(state)
    assert outcome["decision"] == "BLOCK"
    assert outcome["reason"] == "INCOMPLETE_STAGES"
    assert "STAGE_5" in outcome["message"]


def test_manual_emergency_stop_allows_stop() -> None:
    state = {
        "STAGES": _stages(STAGE_2={"STATUS": "IN_PROGRESS"}),
        "COMPLETION": _completion(False),
        "NEXT_SAFE_ACTION": "Do more work.",
        "BLOCKERS": [],
    }
    outcome = gate.evaluate_stop(state, emergency_stop=True)
    assert outcome["decision"] == "ALLOW"
    assert outcome["reason"] == "MANUAL_EMERGENCY_STOP"


def test_runaway_block_limit_escalates_to_allow() -> None:
    state = {
        "STAGES": _stages(STAGE_2={"STATUS": "IN_PROGRESS"}),
        "COMPLETION": _completion(False),
        "NEXT_SAFE_ACTION": "Do more work.",
        "BLOCKERS": [],
    }
    outcome = gate.evaluate_stop(state, prior_blocks=gate.MAX_BLOCKS)
    assert outcome["decision"] == "ALLOW"
    assert outcome["reason"] == "RUNAWAY_BLOCK_LIMIT"


def _run_stop_hook(event: object) -> dict:
    completed = subprocess.run(
        [sys.executable, str(HOOKS / "marathon_stop_gate.py")],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_hook_contract_blocks_against_real_active_state() -> None:
    """The live ACTIVE_STATE has an executable NEXT_SAFE_ACTION -> BLOCK."""
    raw = json.loads(
        (REPO / "experiments/marathon/ACTIVE_STATE.json").read_text(encoding="utf-8")
    )
    if not str(raw.get("NEXT_SAFE_ACTION", "")).strip():
        pytest.skip("live ACTIVE_STATE no longer carries a NEXT_SAFE_ACTION")
    result = _run_stop_hook({"summary": "ending this bounded task cleanly"})
    assert "followup_message" in result
    assert "Marathon is not complete" in result["followup_message"]


def test_hook_contract_evidence_gate_precedence() -> None:
    result = _run_stop_hook({"final_message": "All gates PASS."})
    assert "followup_message" in result
    assert "without named evidence" in result["followup_message"]


def test_hook_contract_emergency_stop_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARATHON_EMERGENCY_STOP", "1")
    completed = subprocess.run(
        [sys.executable, str(HOOKS / "marathon_stop_gate.py")],
        input=json.dumps({"final_message": "stopping now"}),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {}
