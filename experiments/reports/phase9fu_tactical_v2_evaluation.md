# QS-P9FU-HEURISTIC-TACTICAL-V2 evaluation

Created: 2026-08-05

## Status

`BEHAVIOURAL_EVAL_PENDING`

## Candidate

- Policy: `heuristic_v2f_tactical_attack_v2`
- Package ID: `QS-P9FU-HEURISTIC-TACTICAL-V2`
- Ablation: V2F planner + terminal + hunt + credible + persistent + dual + `use_attack_commitment`

## Implemented behaviour (code)

- Internal `AttackCommitmentState` (NONE / PREPARE / COMMIT / CONVERT / RETREAT)
- Soft-gate: does not extend `StrategicPhase`; PREPARE rewrites `phase_reason` and demotes approach proposals
- CASTLE→BUILD hunt filter fix (shared with hybrid path)
- Contextual castles allowed in PREPARE; BUILD stripped in COMMIT
- Off-route COLLECT stripped in COMMIT

## Pending

- Per-candidate behavioural gate vs frozen V001
- Paired evaluation (only if behavioural PASS)
- Do not invent win/draw/loss metrics here
