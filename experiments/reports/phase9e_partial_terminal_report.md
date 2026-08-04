# Phase 9E terminal report — matched BASELINE vs CURRICULUM_ONLY

**Branch:** `research/rules-aware-phase9e-v1`  
**Start:** `b53d9e7` (Phase 9D preserved) → readiness at `4b86e92`  
**Pilot manifest:** `experiments/manifests/phase9e_matched_pilot.json`  
**Process:** completed spontaneously (exit 0); no interrupt required

## Verdict

Neither CNN nor graph curriculum beat its matched baseline. **No replication, confirmation, Phase 10, or learned package.**

| Arm | Env steps | Best monitoring W/D/L | Faults | Treatment |
| --- | --- | --- | --- | --- |
| CNN BASELINE | 12288 | 0/8/0 | 0 | — |
| CNN CURRICULUM | 12288 | 0/8/0 | 0 | NO_MEANINGFUL_IMPROVEMENT |
| GRAPH BASELINE | 12288 | 0/8/0 | 0 | — |
| GRAPH CURRICULUM | 12288 | 0/8/0 | 0 | NO_MEANINGFUL_IMPROVEMENT |

Step-zero equality: PASS. Device: `cuda:0`. Confirmation/holdout: sealed. Portal remains `heuristic_v2f_plus_planner_terminal_force`.

## Operator redirection

Redirect to Phase 9F autonomous audit → generated plan → repair → candidates → packaging.
Do not rerun discovery-gated reward-only PPO.
