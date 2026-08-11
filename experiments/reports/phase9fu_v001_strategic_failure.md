# QS-PUBLIC-V001 strategic failure

Created: 2026-08-05T13:23:35.514855+00:00

## Classification

`TECHNICALLY_QUALIFIED_STRATEGICALLY_WEAK_BASELINE`

## Live package (actually uploaded)

- Path: `C:/Users/pries/Documents/Projects/quantsilico-generals-competition/submission/packages/heuristic_v2_preppo_8f7405fe9834161c_packaged.zip`
- SHA-256: `e1237f77dee469935fc3a60811b9a34522b83dd37bf4d76fa2555e6107a8edfa`
- Candidate: `heuristic_v2f_plus_planner_terminal_fix`
- Stable ID: `QS-P9F-PORTAL-V0`
- Upload: `2026-08-04T00:37:00+01:00` (2026-08-03T23:37:00Z)

## Code-grounded defects

- V2F sets `prefer_castles=False` and `castle_weight=0.85`
- Known enemy general immediately selects `GENERAL_HUNT` in `phase_controller_v2f.select_phase`
- Approach-enemy-general proposals use `hard_priority=93`
- Castle proposals use `hard_priority=26` and are gated (builds&lt;2, turn&lt;600, phase exclusions)
- Collection receives a toward-enemy bonus when the general is known
- No attack-readiness gate before commit

Observed premature hunting / weak preparation is the expected result of this hierarchy.

## Related non-live repackage

- `dist/submission_recommended/qs_p9fu_heuristic_v1_packaged.zip` SHA `898b37b104545fa6217877dd2db2af7c6e8810f41b4ba1f79cc8530b798d558e`
- Same underlying policy; Windows smoke only — **not** the live portal upload
