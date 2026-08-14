---
name: ultra-review
description: Forensic repository review protocol for the QuantSilico marathon. Captures real git state, reconciles authority documents against code/state, runs the actual test gates, diffs recorded state against reality, and attempts to disprove completion claims. Use when auditing marathon stage completion, reviewing takeover state, performing the final adversarial review, or whenever a PASS/COMPLETE claim must be verified against evidence.
---

# Ultra Review

Evidence-first adversarial review. The reviewer attempts to DISPROVE completion; claims survive only when real evidence supports them.

## Prime directive

Hierarchy of truth (never reverse it):

```
implementation -> real command execution -> machine-readable result
-> validation/gate -> captured evidence -> documentation/report
```

Distinguish: instruction vs implementation; test code vs passing test; state claiming PASS vs actual PASS; report vs underlying result data.

## Workflow

Copy this checklist and track progress:

```
Ultra-Review Progress:
- [ ] 1. Capture git state (read-only)
- [ ] 2. Reconcile authorities
- [ ] 3. Run the real gates
- [ ] 4. Diff state files vs reality
- [ ] 5. Adversarial disproval pass
- [ ] 6. Emit structured report
```

### 1. Capture git state (never mutate)

```powershell
git status --porcelain=v1 --branch
git log --oneline -15
git diff --stat
git worktree list
git submodule status
```

Record branch, ahead/behind counts, dirty files, worktrees. No reset/clean/revert, ever.

### 2. Reconcile authorities

Read in order: `AGENTS.md`, `docs/marathon/EXECUTION_PLAN.md`, `docs/marathon/AGENTIC_EXECUTION_PROTOCOL.md`, `docs/marathon/DESIGN_AUTHORITY.md`, `docs/marathon/EVIDENCE_LEDGER.md`, `docs/marathon/CONTROL_ENGINEERING.md`, `configs/marathon/programme.yaml`, `experiments/marathon/ACTIVE_STATE.json`.

Source-of-truth hierarchy: code/results/state > configuration > authority docs > chat narrative. Any disagreement is a finding, not something to silently resolve.

### 3. Run the real gates

Execute the commands recorded in `ACTIVE_STATE.TESTS_COMPLETED` (and the stage's acceptance gates) and compare actual output against the recorded `RESULT`. A claimed PASS that does not reproduce is a BLOCKING finding.

Typical Stage 0B gate set (run from repo root with `.venv`):

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/test_agentic_orchestrator_schemas.py tests/unit/test_agentic_orchestrator_subprocess.py tests/unit/test_agentic_orchestrator_workflow.py tests/unit/test_agentic_orchestrator_hardening.py -q
.venv\Scripts\python.exe -m ruff check tools/agentic_orchestrator
.venv\Scripts\python.exe -m tools.agentic_orchestrator dry-run
```

### 4. Diff state vs reality

- `ACTIVE_STATE.DIRTY_FILES` vs `git status` output.
- `ACTIVE_STATE.TESTS_COMPLETED` vs re-run results.
- Ledger hashes vs actual file hashes (`Get-FileHash -Algorithm SHA256`).
- `HEAD_COMMIT: SELF` resolves via `git log -1 --format=%H -- experiments/marathon/ACTIVE_STATE.json`.
- Empty lists mean "audited and none found"; missing audits must be `UNKNOWN` with a reason.

### 5. Adversarial disproval checklist

Attempt to disprove each of:

- Completion (is every claimed stage backed by a gate run this session or a captured artefact?)
- Test adequacy (do tests assert behaviour, or merely existence?)
- Evidence (every PASS has a command + captured output + ledger entry with source class)
- State consistency (ACTIVE_STATE vs git vs ledger mutually agree)
- Scope (no out-of-scope edits; no unexplained dirty files)
- Provenance (checkpoint/config/seed/commit recorded for every experiment claim)
- Recovery safety (a fresh agent can resume from repository files alone)
- Git safety (no force push, no history rewrite, no deleted unique evidence)

### 6. Report format

```markdown
# Ultra-Review Report

## Verdict: ACCEPT | FIX_FIRST | RETHINK | INSUFFICIENT_EVIDENCE | HUMAN_BOUNDARY
## Position
<stage/gate and the specific evidence for it>
## Findings
- [BLOCKING|MAJOR|MINOR] <finding> — evidence: <command/path/hash>
## Reproduced gates
| Command | Claimed | Actual |
## Unresolved uncertainties
```

Never emit ACCEPT while a BLOCKING finding stands. Never fabricate missing evidence; mark it UNAVAILABLE.
