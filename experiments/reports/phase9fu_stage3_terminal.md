# Phase 9FU Stage 3 terminal report

Created: 2026-08-05T19:36:39.011924+00:00
Revised: 2026-08-06 (escape corruption repair; PROTOCOL_FAULT evaluator; pre-overnight supersession)

## Authorization

Phase A completed. Status superseded by `PRE_OVERNIGHT_REBUILD` (see
`experiments/manifests/phase_master_authorization.json`).
Overnight training, upload, portal mutation, rented compute, and Phase 10 remain
**unauthorized**.

## Starting state

- Branch: `research/phase9fs-ppo-semantics-repair-v1`
- Commit: `ab7c9c0f541c47300406b8c1ae5932ba698b3973`
- Evaluator case: **CASE_2** (opaque Hybrid incomplete) → aborted and repaired

## V001 identity resolution (frozen ZIP)

Frozen package `main.py` constructor: `heuristic_v2f_plus_planner_terminal_fix`
Aliases (not historical runtime): `…_form`, `…_force`

## Tactical V2

- Local diagnostic: score_rate 0.5 vs V001, suite_mean 0.671875 → not V002-eligible
- Portal: NOT_QUALIFIED (Hunter L/L/L)
- Combined: PORTAL_REJECTED / INELIGIBLE_FOR_V002 / DO_NOT_REUPLOAD
- Recorded in `submission/roles/rejected.json`

## Evaluator recovery

- Resumable evaluator shipped with progress/checkpoint/resume/CLI/heartbeat/SIGINT
- Protocol v1 unchanged (SHA `78edc3b31dabaeeb24906422effd7b3f98525cac6dcd92de8166b8ff1fdbdd0f`)
- Silent `except → PASS` replaced with `PROTOCOL_FAULT` recording (scored=false)
- Tests: `tests/unit/test_phase9fu_paired_eval.py`

## Hybrid gates

Measurement gate: PASS (SOURCE_RUNTIME_ONLY)
Package prechecks: PASS (Windows handshake PASS; Linux NOT_RUN)
Package-runtime confirmation: PASS

## Hybrid Stage 3 results (protocol v1)

- Direct vs V001: 8W/10D/14L, score_rate **0.40625**, draw_rate 0.3125
- Suite mean: **0.40625**
- Classification: **REJECTED**
  - direct_improvement_vs_even: -0.09375
  - pairs_ok: True, suite_ok: False, draw_ok: True

## Seed+1000 baseline convention audit

See `experiments/manifests/phase9fu_seed_offset_audit.json`.
Baseline pairs use `seed + 1000` relative to the shared protocol seed list.
This matches the Phase A repaired evaluator and does **not** change the Hybrid
rejection (direct 0.40625 already decisive). Hybrid Stage 3 was **not** re-run.

## Recommendation

`NO_CANDIDATE_CURRENTLY_RECOMMENDED`

`UPLOAD_THIS.md` and `recommended.json` agree. No upload. Do not re-upload V001
or Tactical V2. Do not recommend Hybrid.

## Hard stop confirmation (Phase A)

- No portal mutation
- No overnight / Phase 10 started under Phase A
- Phase 9FU Stage 3 evidence frozen for competition-native JAX pre-overnight rebuild
