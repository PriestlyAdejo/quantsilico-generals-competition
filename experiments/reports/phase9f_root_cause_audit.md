# Phase 9F root-cause audit

Created: 2026-08-04T17:35:09.922395+00:00

## Executive verdict

Phase 9E did not fail because CUDA or legality broke. It failed because the trainer
cannot credit long-horizon outcomes: GAE zero-bootstraps every rollout boundary,
each 512-step chunk starts a fresh environment, gamma=0.99 erases ~1200-turn terminals,
and recurrent state is not persisted across chunks. Reward shaping alone cannot fix this.

## Gates

- LONG_HORIZON_CREDIT_GATE: **FAIL** (0.99^1200 = 5.78e-06)
- CHUNK_CREDIT_CONTINUITY_GATE: **FAIL**
- PARTIAL_OBSERVABILITY_MEMORY_GATE: **FAIL**
- EFFECTIVE_VOLUME_GATE: **FAIL** (~10.2 full-game equivalents at 1200 turns)

## Immediate remediations (auto-execute)

1. Fix `_gae` truncation bootstrap.
2. Persist env + hidden state across updates and chunks.
3. Wire structured map memory into learned/hybrid act path.
4. Prefer hybrid expert+ranker + BC/DAgger before another full-game PPO.

