# Marathon Agentic Execution Protocol

**Applies to plan:** `MARATHON_REDESIGN_LOCKED_V1`

**Purpose:** Make cross-model work durable, reviewable, and independent of chat history.

## 1. Roles

### Codex GPT-5.6 Sol

Codex GPT-5.6 Sol is the architect, research planner, material-conflict resolver, and fresh independent reviewer. It owns architectural interpretation, evidence-policy review, research-order challenges, and decisions that would change acceptance criteria. It should review implementation from repository evidence rather than relying on the implementing model's narrative.

### Cursor Grok 4.6

Cursor Grok 4.6 is the default bulk implementer. It executes one bounded repository task at a time, tests it, records evidence, and updates handoff state. It may identify a plan defect, but it may not silently redesign the programme or weaken a gate.

The human owner may assign a different model. The same implementer/reviewer separation and evidence rules still apply.

## 2. Shared memory

Repository files are shared memory. Chat history is never required context and is not an authority. Before Marathon work, read root `AGENTS.md`, `EXECUTION_PLAN.md`, this protocol, every existing Marathon authority named in root `AGENTS.md`, and `ACTIVE_STATE.json`.

Commands, decisions, evidence, hashes, open processes, incomplete tests, and next actions must be persisted. A handoff that exists only in a chat message is incomplete.

## 3. Material conflicts

An implementer may make local implementation choices inside an approved bounded task. It must surface a disagreement that changes architecture, safety, experimental meaning, authority, cost, destructive scope, or acceptance criteria.

Use this exact structure:

```text
PLAN_CONFLICT
PLAN_ID: MARATHON_REDESIGN_LOCKED_V1
PLAN_STATEMENT: <the controlling plan text or requirement>
REPOSITORY_EVIDENCE: <paths, commits, hashes, commands, observed behaviour>
PROPOSED_CORRECTION: <smallest viable correction and alternatives>
AFFECTED_ACCEPTANCE_CRITERIA: <tests/gates that would change>
SAFE_WORK_COMPLETED: <work that remains valid regardless of the decision>
NEXT_DECISION_REQUIRED: <architect or human decision>
```

Record the conflict in `ACTIVE_STATE.json` and the evidence ledger when it exists. Continue only with safe work that cannot prejudice the decision.

## 4. Implementation completion contract

Implementation status is exactly one of:

- `COMPLETE`
- `PARTIAL`
- `BLOCKED`
- `REFUSED`
- `FAILED`

Every implementation report contains:

```text
STATUS
PLAN_ID
BASE_COMMIT
END_COMMIT
FILES_CHANGED
TESTS_RUN
TESTS_PASSED
TESTS_FAILED
EVIDENCE
KNOWN_LIMITATIONS
PLAN_DEVIATIONS
NEXT_SAFE_ACTION
```

Silent-success, empty-diff, and test-free completion are prohibited for implementation work. `COMPLETE` requires a material diff within the authorized scope, relevant tests, acceptance evidence, and updated handoff state. A verification-only task may have an empty diff only when it was explicitly scoped as verification and reports its commands and evidence.

Do not invent a passing result. If a test could not run, state why under `TESTS_FAILED` or `TESTS_PENDING` and use `PARTIAL`, `BLOCKED`, `REFUSED`, or `FAILED` as appropriate.

Two failed attempts at the same substantive problem trigger architect escalation. Preserve both attempts' evidence and stop uncontrolled retries or broad rewrites.

## 5. Independent review

Reviewer verdicts are exactly one of:

- `ACCEPT`
- `FIX_FIRST`
- `RETHINK`
- `INSUFFICIENT_EVIDENCE`
- `HUMAN_BOUNDARY`

The reviewer checks the diff, repository truth, tests, evidence, scope, plan gates, rollback, and handoff state. It does not accept an implementer's success claim in place of evidence. Material reviewer disagreement follows `PLAN_CONFLICT`.

`ACCEPT` is the automation-facing spelling of the former `SHIP` verdict. Historical `SHIP` records remain interpretable as `ACCEPT`, but new records emit `ACCEPT`. `HUMAN_BOUNDARY` pauses without treating the implementation as failed.

## 5.1 Local orchestration (`ORCH-0001`)

The Stage 0B supervisor implements this protocol; it does not replace it. Codex architect/reviewer subprocesses are read-only. The designated implementer is the only writer. The supervisor validates structured task, implementation-report, and review records, persists every transition atomically under ignored runtime state, and recovers after interruption. Three review/repair cycles are the absolute task ceiling, while two failed attempts carrying the same substantive-problem fingerprint trigger earlier architect escalation.

The desired model identity is a requirement, not an undocumented CLI alias. The supervisor queries installed CLI configuration/model listings where supported, records the resolved identity, and fails or pauses loudly when unavailable. It never silently selects another model family.

Live orchestration must pause as `PAUSED_USAGE` on quota exhaustion and `PAUSED_HUMAN_BOUNDARY` before paid-resource creation or expansion, repository visibility changes, competition uploads, destructive evidence removal, force pushes/history rewriting, pinned-engine edits, credential operations, or an explicit human boundary. A model-usage pause must not kill an independent healthy trainer.

## 6. Human-controlled operations

Only the human owner may authorize:

- new or expanded paid resources;
- repository visibility changes and `REPOSITORY_PRIVACY_CUTOVER`;
- destructive removal of potentially unique evidence;
- competition uploads;
- pushes unless the active bounded task expressly permits them.

The dashboard never receives arbitrary shell, arbitrary filesystem, Git mutation, paid-resource, repository-visibility, push, or competition-upload authority. Dashboard jobs are durable, typed, allowlisted, and configuration-ID-driven.

## 7. Handoff state

`experiments/marathon/ACTIVE_STATE.json` is updated before every session stops. It contains at minimum these exact uppercase keys:

- `STATUS`
- `CURRENT_STAGE`
- `CURRENT_TASK`
- `BASE_COMMIT`
- `HEAD_COMMIT`
- `WORKTREE`
- `DIRTY_FILES`
- `LAST_VALIDATED_COMMIT`
- `TESTS_COMPLETED`
- `TESTS_PENDING`
- `ACTIVE_LOCAL_PROCESSES`
- `ACTIVE_REMOTE_PROCESSES`
- `CURRENT_EXPERIMENT`
- `CURRENT_CHECKPOINT`
- `NEXT_SAFE_ACTION`
- `BLOCKERS`
- `PLAN_DEVIATIONS`

It also carries `SCHEMA_VERSION`, `PLAN_ID`, `UPDATED_AT_UTC`, and `OWNER_ROLE`.

An empty process or dirty-file list means the state was explicitly audited and no entries were found. If it was not audited, use `UNKNOWN` with a reason. Never infer empty from missing access.

`HEAD_COMMIT: "SELF"` and `LAST_VALIDATED_COMMIT: "SELF"` mean the commit that contains that version of `ACTIVE_STATE.json`. Resolve `SELF` with:

```powershell
git log -1 --format=%H -- experiments/marathon/ACTIVE_STATE.json
```

The sentinel avoids an impossible commit self-hash. A later dirty session replaces or contextualizes these fields as appropriate and lists exact dirty files.

## 8. Machine-instruction audit at architect handoff

Audit date: `2026-08-14`.

- `C:\Users\pries\.codex\AGENTS.md` exists and is zero bytes; it contributes no conflict.
- No home, ancestor, repository, or Codex `AGENTS.override.md` was found.
- No repository or user Cursor rules were found.
- Codex configuration defaults to `gpt-5.6-sol` with high reasoning, consistent with the architect role.
- The prior repository `AGENTS.md` described the currently public repository as private and left “one stage at a time” ambiguous relative to independent Stage 4A/4B tracks. The root contract was clarified: bounded work remains one task per worktree while independent research/platform work may proceed after shared prerequisites and without conflicting edits.
- No machine-wide configuration was deleted or rewritten during this handoff.

Future agents must report newly discovered higher-precedence conflicts instead of deleting or silently overriding them.
