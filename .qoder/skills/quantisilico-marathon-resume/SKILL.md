---
name: quantisilico-marathon-resume
description: Resumes the QuantSilico marathon from durable repository state without chat history. Reads experiments/marathon/ACTIVE_STATE.json, resolves SELF commit sentinels, verifies state against git reality, runs cheap targeted gates, and emits exactly one next bounded safe action. Use when starting a new session on this repository, when a session was interrupted, or when asked what to do next in the marathon.
---

# QuantSilico Marathon Resume

Repository files are shared memory; chat history is never required.

## Workflow

1. **Read the authorities** (in order, skipping none that exist):
   `AGENTS.md`, `docs/marathon/EXECUTION_PLAN.md`, `docs/marathon/AGENTIC_EXECUTION_PROTOCOL.md`,
   `experiments/marathon/ACTIVE_STATE.json`, `docs/marathon/EVIDENCE_LEDGER.md` (open items).

2. **Resolve sentinels**:
   ```powershell
   git log -1 --format=%H -- experiments/marathon/ACTIVE_STATE.json
   git status --porcelain=v1 --branch
   ```
   `HEAD_COMMIT: SELF` / `LAST_VALIDATED_COMMIT: SELF` mean the commit containing that ACTIVE_STATE version.

3. **Verify state vs reality** (cheap checks only):
   - `DIRTY_FILES` matches `git status` (empty list = audited clean; otherwise stale state is a finding).
   - Branch/HEAD match the recorded worktree expectation.
   - `TESTS_COMPLETED` claims look consistent with current HEAD; if suspicious, re-run the narrowest relevant gate.

4. **Run the narrowest gate appropriate to `CURRENT_TASK`** — never the full expensive suite first. Examples:
   - Orchestrator work: the three/four `tests/unit/test_agentic_orchestrator_*` modules.
   - Baseline work: checkpoint hash spot-checks before any training command.

5. **Emit exactly one next bounded safe action** taken from (or corrected against) `NEXT_SAFE_ACTION`:
   - one task, explicit scope (files/areas), explicit acceptance gate, explicit evidence to record.
   - If `BLOCKERS` contains a genuine external blocker, continue only independent workstreams and record it.

6. **Before stopping for any reason**, update `ACTIVE_STATE.json` with the exact protocol §7 key set and real values. Unknown state is `UNKNOWN` with a reason; never infer empty from missing access.

## Hard rules during resume

- No destructive git (`reset --hard`, `clean -fd`, force push) and no deletion under `experiments/`, `models/`, `replays/`.
- Do not kill processes whose identity is unknown (WSL workload rule).
- Do not claim PASS without a captured command result.
- Two failed attempts at the same substantive problem -> architect escalation, not more retries.
