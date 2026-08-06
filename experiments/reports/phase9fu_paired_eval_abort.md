# Phase 9FU paired evaluation abort

Created: 2026-08-05T15:56:00+00:00

## Case

`CASE_2` under `CURRENT_EVALUATOR_STATE_RULE` (re-resolved at execution):

- paired evaluator still alive (`scripts/phase9fu_paired_eval.py`);
- Hybrid BC had no candidate summary;
- no resumable checkpoint / partial JSON;
- no `phase9fu_v001_vs_challengers.json`.

## Action

Preserved logs and process metadata under
`experiments/aborted_runs/phase9fu_paired_eval_20260805/`, then terminated only the
paired-evaluator process tree.

## Classification

| Candidate | Result |
|-----------|--------|
| QS-P9FU-HEURISTIC-TACTICAL-V2 | Diagnostic from log: direct score_rate 0.5, suite_mean 0.671875 — not V002-eligible under protocol v1 (+0.05) |
| QS-P9FU-HYBRID-BC-V1 | `ABORTED_OPAQUE_NO_CHECKPOINT` / `NOT_EVALUATED_TO_COMPLETION` |

Do not invent Hybrid performance from the incomplete run.
