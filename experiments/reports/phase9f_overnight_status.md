# Phase 9F overnight execution status

**Updated:** 2026-08-05T01:04+01:00 (approx)  
**T0:** 2026-08-05T01:41:56+01:00  
**Deadline:** 2026-08-05T08:00:00+01:00  
**Stop collection:** 2026-08-05T07:15:00+01:00  
**Recovery tag:** `phase9f-mandatory-rl-resume-ab7c9c0`  
**Plan v2 sha256:** `cafe3f473e2d81727f6b89fcf0c3df7a4cf90db21dcf4faccc786173462e6d3c`

## Completed

1. Verified branch `research/phase9f-autonomous-rebuild-v1` @ `ab7c9c0`; no conflicting trainers.
2. Package metadata corrected (≤15m): canonical portal `heuristic_v2f_plus_planner_terminal_fix` / `QS-P9F-PORTAL-V0`; `official_upload_ready=false`; physical ZIP moves deferred (`LEGACY_MISLABELLED`).
3. Plan v2 + research traceability locked.
4. Persistent actors + sync PPO freshness: **PASS** (`phase9f_persistent_actor_gate.json`). Episode-boundary resume fallback recorded.
5. Canonical `BeliefMemory` wired into actors.
6. CNN vertical slice **PASS**:
   - teachers: portal + aggressive/defensive/castle/deathtouch
   - train_n=9340, val_n=2334
   - BC val_action_acc≈0.66 (`QS-P9F-CNN-RANKER-V1` / `recurrent_cnn_v2`)
   - sync PPO pilot: 16 updates, persistent+synchronous
7. Overnight matched PPO **RUNNING**: `RL_CONTROL` then `RL_CURRICULUM` from BC checkpoint until 07:15.

## In progress

- Overnight script: `scripts/run_phase9f_overnight_ppo.py`
- Log: `experiments/manifests/_phase9f_overnight_ppo.log`
- Output: `experiments/phase9f_overnight_ppo/{rl_control,rl_curriculum}/`

## Still due after 07:15

- Finalist tournament (provisional labels if under-sampled)
- Linux/CPU qualification levels
- Physical package migration + strict `dist/upload_ready` only if official
- Clean exit by 08:00

## Exact resume (if process dies)

```text
.\.venv-training\Scripts\python.exe -u scripts/run_phase9f_overnight_ppo.py --device cuda --arm both --rollout-steps 256 --stop-at 2026-08-05T07:15:00+01:00
```

Weights-only BC warm start:
`experiments/phase9f_cnn_ranker_v1/checkpoints/bc/model.json`
