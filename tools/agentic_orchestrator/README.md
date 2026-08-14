# Local Agentic Orchestrator

This package implements Stage 0B amendment `ORCH-0001`. It coordinates one bounded architect → implementer → independent-review unit at a time while the repository remains durable shared memory.

## Safety properties

- Codex architect and reviewer subprocesses use fresh ephemeral, read-only `codex exec` calls.
- The designated implementer is the only writer; an exclusive runtime lock prevents concurrent supervisors.
- Task, report, review, and state records use strict versioned schemas.
- JSON state writes use same-directory temporary files, `fsync`, and atomic replacement.
- Every state transition is appended to `events.jsonl` and persisted before the next agent call.
- A non-verification task cannot be accepted unless its report includes the canonical `ACTIVE_STATE.json` update; evidence-ledger and coherent-commit requirements remain part of the bounded task acceptance criteria.
- Three review/repair cycles are the absolute ceiling. Two repeated substantive-problem fingerprints escalate earlier.
- Usage exhaustion pauses as `PAUSED_USAGE`; a human-controlled operation pauses as `PAUSED_HUMAN_BOUNDARY`.
- A pause never terminates an independent trainer.
- Subprocesses use argument arrays with `shell=False`; sensitive environment names are excluded and captured output is redacted.
- The default is one task. `--until-human-boundary` still has an internal five-task ceiling.

Runtime files live under ignored `var/agentic/`:

```text
orchestrator_state.json
current_task.json
implementation_report.json
review.json
events.jsonl
schemas/*.schema.json
writer.lock                 # present only while a supervisor owns the writer role
```

Stale locks are never deleted automatically. Inspect the recorded PID and process state before removing one.

## Commands

Run from the repository root:

```powershell
python -m tools.agentic_orchestrator status
python -m tools.agentic_orchestrator tooling
python -m tools.agentic_orchestrator dry-run
python -m tools.agentic_orchestrator run --once
python -m tools.agentic_orchestrator run --max-tasks 5
python -m tools.agentic_orchestrator run --until-human-boundary
python -m tools.agentic_orchestrator pause --reason "operator request"
python -m tools.agentic_orchestrator resume
```

`tooling` never prints credentials. It reports executable, version, authentication status, configured Codex model, and Cursor's reported model list. The live adapter requires an exact model display match; it does not silently substitute a model. Override a verified Cursor display identity with `--cursor-model-display`.

`CODEX_CLI_PATH` and `CURSOR_AGENT_CLI_PATH` may point to already installed official executables. These variables are paths, not credentials. The supervisor does not install tools or handle authentication secrets.

## Dry run versus live acceptance

`dry-run` uses deterministic in-process stand-ins and a separate `var/agentic/dry-run/` state tree. It proves schema flow, repair, acceptance, restart recovery, quota pause, human-boundary pause, and no repository edits. It does not claim a live model invocation.

Live execution refuses to start unless both CLI probes pass and the requested Cursor model identity is reported exactly. The current repository evidence records Cursor CLI/usage as unavailable, so live Cursor end-to-end acceptance remains pending rather than silently falling back.

Do not use unrestricted/yolo modes. Do not feed `.env`, tokens, private keys, credentials, or untracked private evidence into prompts.
