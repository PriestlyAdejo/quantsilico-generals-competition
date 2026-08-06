# Phase 9FU evaluator recovery

## Problem

`scripts/phase9fu_paired_eval.py` printed only one line per candidate and wrote
results only after the whole candidate finished. The Hybrid run was aborted as
`ABORTED_OPAQUE_NO_CHECKPOINT`.

## Repair (protocol v1 retained)

Implementation amendment:
`experiments/manifests/phase9fu_evaluator_implementation_amendment_v1.json`

Added:

- per-game / per-pair progress JSON events
- atomic partial checkpoints under `experiments/manifests/phase9fu_eval_<id>.partial.json`
- `--resume-from`, `--candidate`, `--opponent`
- `--max-game-wall-s`, `--max-candidate-wall-s` → `INCOMPLETE_TIMEOUT` (not scored)
- heartbeat ≤60s with RSS and progress
- SIGINT/SIGTERM flush of completed pairs only
- explicit `evaluation_runtime: source_tree_policy_factory`

Frozen seeds, maps, seats, opponents, scoring and thresholds were not modified.
