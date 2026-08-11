# Phase 9F terminal report

**Local time:** 2026-08-05T07:18:00+01:00  
**Deadline:** 2026-08-05T08:00:00+01:00  
**Branch:** `research/phase9f-autonomous-rebuild-v1`  
**Recovery tag:** `phase9f-mandatory-rl-resume-ab7c9c0`  
**Plan v2 sha256:** `cafe3f473e2d81727f6b89fcf0c3df7a4cf90db21dcf4faccc786173462e6d3c`

## Mandatory RL

**MANDATORY_RL_GATE: PASS**

| Arm | Updates | ~Transitions | Elapsed | Loss first→last | Entropy last |
| --- | --- | --- | --- | --- | --- |
| RL_CONTROL | 144 | 36864 | 9392s | 7.48→0.0006 | **0.0** |
| RL_CURRICULUM | 127 | 32512 | 9341s | 7.44→0.0008 | **0.0** |

- Architecture: `recurrent_cnn_v2` (`QS-P9F-CNN-RANKER-V1`)
- Init: BC checkpoint (val_action_acc ≈ 0.66)
- Trainer: persistent actors, synchronous policy freshness
- Episode resume: `PARTIAL_WITH_EPISODE_BOUNDARY_FALLBACK`
- Fragment `continuation_mask` nonzero for all recorded updates
- **Caution:** final entropy collapsed to 0 on both arms — treat as training-continuity / mandatory-RL evidence, not a competent deployable policy without further eval/repair

## Packages / roles

- Strict `dist/upload_ready/`: **empty** (no `OFFICIAL_UPLOAD_READY`)
- Portal ZIPs: `dist/windows_smoke_passed/` + `dist/legacy_mislabelled_upload_ready/`
- Portal remains **PROVISIONAL_BEST_OVERALL_AMONG_TESTED_CANDIDATES** among packaged bots
- Learned PPO checkpoints: research-only (`experiments/manifests/phase9f_research_candidates.json`)
- Linux: **BLOCKED_EXTERNAL** (parity not run)
- Finalist tournament: **PROVISIONAL_UNDER_SAMPLED** (0 paired comparisons; predeclared sample was 16)

## Explicit non-claims

- Not `CONFIRMED_BEST` / `PROMOTABLE`
- Not official upload ready
- RL overnight is evidence of mandatory attempt + episode-continuity repair, not a proven portal beater
