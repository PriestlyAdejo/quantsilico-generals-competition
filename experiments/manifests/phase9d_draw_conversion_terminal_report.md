# Phase 9D terminal report — dual-architecture draw-conversion

**Branch:** `research/draw-conversion-cnn-graph-v1`  
**Start:** `79f6f93` (protocol-integrity last_verified)  
**Research generation:** `phase9d_draw_conversion_2026-08-04`

## Verdict

Neither CNN nor graph conversion beat its matched control. **No matched-pair replication, no expanded confirmation, no Phase 10, no learned package.**

| Arm | Env steps | Best monitoring W/D/L | Faults | Treatment effect |
| --- | --- | --- | --- | --- |
| CNN CONTROL | 12288 | 0/8/0 | 0 | — |
| CNN CONVERSION | 12288 | 0/8/0 | 0 | NO_MEANINGFUL_IMPROVEMENT |
| GRAPH CONTROL | 12288 | 0/8/0 | 0 | — |
| GRAPH CONVERSION | 12288 | 0/8/0 | 0 | NO_MEANINGFUL_IMPROVEMENT |

Step-zero equality: PASS for both architectures.

## Gates

| Gate | Decision |
| --- | --- |
| CNN_OVERNIGHT_READINESS_GATE | BLOCKED |
| GRAPH_OVERNIGHT_READINESS_GATE | BLOCKED |
| Selected overnight candidate | none |
| Expanded confirmation | not opened (seeds remain sealed) |
| Promotion holdout | untouched |
| Learned package | NOT PACKAGED |

## Dominant blockers (both)

1. Zero genuine wins after 8× INITIAL env-step budget  
2. Contact intervention (`conversion_v1`) did not create measurable treatment effect vs control  
3. Draw-dominated score_rate 0.5 throughout monitoring

## Diagnosis / intervention (for next experiment)

- Dominant diagnosis: **CONTACT_FAILURE** (both)  
- Intervention tested: policy-visible contact shaping + train-vs-Expander curriculum  
- Honest attribution: extra training alone also failed to produce wins under control

## Exact next action

Design a stronger contact/exploration curriculum under a new versioned intervention (still minimal: one reward family + one curriculum), with denser contact opportunities or longer horizons — do **not** open confirmation seeds or overnight until a PROMISING matched treatment effect appears.
