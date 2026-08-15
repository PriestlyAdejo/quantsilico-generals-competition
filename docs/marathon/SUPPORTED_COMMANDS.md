# Marathon supported-command inventory (Stage 4B)

Canonical operations surface for Marathon executors and loop iterations.
Risk classes: SAFE (read-only/deterministic), MUTATING_LOCAL (writes repo
state; commit afterwards), REMOTE_SPEND (paid RunPod actions), MANUAL_ONLY
(human-controlled boundaries — never automate).

## Resume and state

| Command | Class | Purpose |
| --- | --- | --- |
| `.venv/Scripts/python.exe scripts/dev/marathon_execution_lease.py acquire\|heartbeat\|status\|release --owner <id>` | MUTATING_LOCAL | Loop mutual exclusion (exit 3 held, 4 stale takeover) |
| read `experiments/marathon/ACTIVE_STATE.json` | SAFE | Canonical resume truth (`NEXT_SAFE_ACTION`) |
| read `docs/marathon/EVIDENCE_LEDGER.md`, `experiments/marathon/registry/` | SAFE | Evidence + Stage-3 registry |

## RunPod lifecycle (zero-idle-burn: reconcile FIRST on every resume)

| Command | Class | Purpose |
| --- | --- | --- |
| `.venv/Scripts/python.exe scripts/dev/runpod_idle_watchdog.py` | SAFE | Enumerate RUNNING pods, verify workloads over SSH |
| `.venv/Scripts/python.exe scripts/dev/runpod_idle_watchdog.py --stop-idle` | REMOTE_SPEND | Stop idle/finished pods AFTER preservation; billing-logged |
| `.venv/Scripts/python.exe scripts/dev/runpod_probe.py` | SAFE | Account/pod inventory snapshot |
| pod create/restart via `var/marathon_takeover/runpod_introspect.py` helpers | REMOTE_SPEND | Duplicate-capacity check REQUIRED before create |

## Research funnel

| Command | Class | Purpose |
| --- | --- | --- |
| `.venv/Scripts/python.exe scripts/training/run_sh_r1_arm.py --arm-id ... --num-envs ... --rollout-len ... [--min-generals-distance N]` | MUTATING_LOCAL (+REMOTE_SPEND when on pod) | Screening arm runner (pool-fixed; refuses to overwrite telemetry) |
| `.venv/Scripts/python.exe scripts/evaluation/run_marathon_paired_eval.py --candidate-id ... --candidate-main ... --opponent id=path ...` | MUTATING_LOCAL | EV-0017 seat-swapped paired evaluation; resumable per run dir |
| `.venv/Scripts/python.exe scripts/analysis/serving_sanity_probe.py` | SAFE | Serving integrity gate (EV-0034 standing precondition) |
| `.venv/Scripts/python.exe scripts/dev/register_*.py` | MUTATING_LOCAL | Registry records (register BEFORE launch) |

## Replay / external telemetry

| Command | Class | Purpose |
| --- | --- | --- |
| `.venv/Scripts/python.exe scripts/data/fetch_elite_replays.py --top N --per-player N` | SAFE (public API, throttled) | Immutable replay snapshot (charter: pilots stay tiny) |
| `.venv/Scripts/python.exe scripts/data/capture_leaderboard_snapshot.py` | SAFE (public API) | Timestamped leaderboard telemetry feed |

## Packaging (Stage 4B)

| Command | Class | Purpose |
| --- | --- | --- |
| `.venv/Scripts/python.exe scripts/packaging/allocate_submission_outbox.py ...` | MUTATING_LOCAL | Deterministic `submission/outbox/qs-<candidate>-vNNN-<date>.zip` |
| `.venv/Scripts/python.exe scripts/dev/repo_cleanup_inventory.py --out ...` | SAFE | Classification inventory; DRY RUN ONLY (no deletions) |
| `.venv/Scripts/python.exe -m pytest dashboard/backend/tests -q` (via .venv-training) | SAFE | Dashboard contract tests |

## MANUAL_ONLY (human-controlled boundaries)

- Competition upload / portal submission (outbox is allocation only).
- New cash spend / payment methods; repository visibility changes.
- Force-push, history rewrite, destructive removal of unique evidence.
- Credential rotation, pinned-engine edits.
