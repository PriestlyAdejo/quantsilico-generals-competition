# Protocol + dashboard research integrity — terminal report

**Working branch:** `fix/protocol-and-dashboard-research-integrity`  
**Starting commit:** `c671bab`  
**Research generation:** `protocol_dashboard_integrity_2026-08-04`  
**Engine SHA:** `9e3b9d13cca51caa1bb07db48bb85c9e90ce0462`

## Verdict

Protocol root cause is **Class A** (validation adapter only). After the typed forward contract:

* CNN and graph compatibility ladders: **0 protocol faults**
* Corrected Stage 1 (4 seeds × both seats vs Expander): **0 faults**, all draws
* Corrected Stage 2 (24 games/arch vs Expander + Hunter + portal `terminal_fix`): **0 faults**; CNN 0W/24D/0L; graph 0W/23D/1L

Historical INITIAL evidence remains immutable: **0W/2D/0L + 200 faults**, `OVERNIGHT_READINESS_GATE=BLOCKED`.

Corrected overnight readiness: **BLOCKED** (zero genuine wins). Phase 10 / holdout / learned packaging **not entered**. Delivery integrity: **BLOCKED** (expected — not a package failure). Dashboard operability: **PARTIAL**.

## Gates

| Gate | Decision |
| --- | --- |
| PORTAL_ATTRIBUTION_GATE | RESOLVED → `heuristic_v2f_plus_planner_terminal_fix` |
| LEARNED_PROTOCOL_COMPATIBILITY (CNN) | PASS |
| LEARNED_PROTOCOL_COMPATIBILITY (graph) | PASS |
| PROTOCOL_ROOT_CAUSE_CLASSIFICATION | CLASS_A |
| RESEARCH_EVIDENCE_INTEGRITY_GATE | PASS |
| CORRECTED Stage 1 | COMPLETE |
| CORRECTED Stage 2 | COMPLETE |
| OVERNIGHT_READINESS_GATE (corrected) | BLOCKED |
| DASHBOARD_OPERABILITY_GATE | PARTIAL |
| DELIVERY_INTEGRITY_GATE | BLOCKED |

## What was fixed

* Shared `ModelForwardResult` / `adapt_forward_output` contract
* `CheckpointPolicy.act` no longer indexes dict outputs as tuples
* Experiments/Models stop inventing `0W/0D/0L` and zero params
* PFSP/PSRO no longer fill unplayed cells with `0.5`
* Qualification suite selector drives stage + WDL chart filter
* Match jobs can report `METADATA_ONLY` replays honestly
* Stage 1/2 diagnostic replays with frames+actions under `replays/private/protocol_integrity/`

## Honest next experiment

Policies are protocol-correct but competitively weak (draw-dominated at 100 turns). Focused next work should improve win rate against Expander under the repaired validation path — not overnight training on these INITIAL checkpoints.
