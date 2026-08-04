# Phase 9F generated execution plan (locked from audit)

**Plan hash source:** `experiments/manifests/phase9f_generated_plan.json`  
**Near-term architecture:** heuristic candidates + neural ranker (C)  
**Tonight outcome:** portal packages shipped; trainer GAE truncation repaired; full hybrid deferred to resume

## Verified root causes

1. **CRITICAL** — `_gae` zero-bootstrapped every rollout (`values+[0.0]`) → repaired to V(s_T) + done masks
2. **CRITICAL** — each 512-step pilot chunk called fresh `run_bounded_ppo` / fresh env (still open: persistent workers)
3. **HIGH** — gamma=0.99 → 0.99^1200 ≈ 6e-6 terminal credit
4. **HIGH** — recurrent hidden reset every update (repaired within-chunk persistence)
5. **HIGH** — ~10 full-game equivalents per 12288-step arm
6. **PASS as diagnosis** — reward-only curricula cannot invent attack/conversion from draws

## Automatic stages

| Stage | Status tonight | Next on resume |
| --- | --- | --- |
| 9F.0 preserve 9E | DONE | — |
| 9F.1 foundation audit | DONE | — |
| 9F.2 GAE truncation repair | DONE | add persistent env workers across chunks |
| 9F.3 structured memory in learned path | PENDING | implement |
| 9F.4 specialist teachers + scenarios | PENDING | scout/collect/castle/attack/convert/DT |
| 9F.5 teacher dataset + BC | PENDING | after teacher gates |
| 9F.6 DAgger / hybrid ranker | PENDING | after BC |
| 9F.7 constrained RL | BLOCKED until credit+memory+teacher PASS | — |
| 9F.8 package | DONE (portal as overall/fallback/deterministic) | replace when stronger candidate qualifies |

## Decision branches

- Chunk continuity FAIL → repair before PPO (in progress)
- Long-horizon FAIL → repair value targets before PPO
- No stronger neural → package portal; continue research independently
- PPO degrades teacher → revert (N/A tonight)

## Stop conditions tonight

- Upload-ready portal ZIPs present
- Evidence bundle present
- GAE repair committed
- Exact resume command printed
