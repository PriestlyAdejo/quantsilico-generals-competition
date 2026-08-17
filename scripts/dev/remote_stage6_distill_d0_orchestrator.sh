#!/usr/bin/env bash
# STAGE6_DISTILL_PPO_R1 STEP D0 remote orchestrator (predeclared:
# stage6_distill_ppo_r1_plan.yaml). Distills the TEACHER-R2 teacher dataset
# into the canonical transformer on GPU, applies the EV-0060-identical
# screening gate, and writes a warm-start checkpoint for D1.
set -u
cd /workspace/repo || exit 1
PY=/workspace/repo/.venv312/bin/python
mkdir -p /workspace/distill_ppo_r1
LOG=/workspace/distill_ppo_r1_round.log
echo "=== distill_d0 round start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
"$PY" scripts/training/distill_ppo_d0_train.py \
  --out-dir /workspace/distill_ppo_r1/DISTILL-S0-TRANSFORMER-BC >> "$LOG" 2>&1
code=$?
echo "distill_d0 exit=$code $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
echo "=== distill_d0 round end $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
exit $code
