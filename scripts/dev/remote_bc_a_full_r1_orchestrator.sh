#!/usr/bin/env bash
# BC-A-FULL-R1 A40 fallback orchestrator (run ON the pod via
# remote_orchestrator_with_stop.sh).
# Predeclared plan: experiments/marathon/bc_a_full_round_1_plan.yaml
# Registry: experiment#bc-a-full-r1#abcfcd746ac7
# Fallback trigger (predeclared budget clause): local CPU throughput
# insufficient at scale - feature phase wall 13074s (>2h budget) and the
# local trainer was killed at epoch 5/40 by RAM contention during the
# concurrent T2 arbiter (2026-08-16T14:04Z, honest record). Same seed,
# same sealed shards, deterministic rerun of the identical experiment.
# PPO_SEMANTICS: OFF_POLICY_AUXILIARY.
set -u

REPO=/workspace/repo
PY=/workspace/repo/.venv312/bin/python
OUT_LOG=/workspace/bc_a_full_round.log
SHARDS=/workspace/datasets_bc/DATASET-BC-DERIVED-UNION-A2
OUT_DIR=/workspace/bc_a_full_result
export PYTHONPATH=/workspace/repo:/workspace/repo/src:/workspace/repo/third_party/generals-bots

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
cd "${REPO}"
echo "=== BC-A-FULL round start $(stamp) commit $(git rev-parse --short HEAD)" > "$OUT_LOG"
"$PY" scripts/training/bc_a_train_full.py \
  --shard-dir "$SHARDS" \
  --out-dir "$OUT_DIR" \
  --epochs 40 --seed 20260831 >> "$OUT_LOG" 2>&1
code=$?
echo "=== BC-A-FULL round end $(stamp) exit=${code}" >> "$OUT_LOG"
exit "$code"
