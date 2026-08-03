# Manual upload instructions — heuristic_v2 pre-PPO

Do **not** upload automatically from CI or the dashboard.

## Preconditions

1. Windows clean-package validation: PASS
2. Linux parity report: PASS (`experiments/manifests/linux_parity_report_preppo.json`)
3. Package report status: `UPLOAD_READY`
4. `PRE_PPO_SUBMISSION_GATE`: PASS (see `experiments/manifests/phase_9q_pre_ppo_submission_gate.json`)
5. Engine submodule pin matches the competition evaluation pin
6. Existing `heuristic_v1_packaged.zip` left untouched

## Package location

- ZIP: `submission/packages/heuristic_v2_preppo_8f7405fe9834161c_packaged.zip`
- Report: `submission/packages/heuristic_v2_preppo_8f7405fe9834161c_packaged.report.json`
- SHA-256: `e1237f77dee469935fc3a60811b9a34522b83dd37bf4d76fa2555e6107a8edfa`
- Candidate: `heuristic_v2f_plus_planner_terminal_fix`
- Config hash: `8f7405fe9834161c`

## Operator steps

1. Verify SHA-256 in the package report matches the ZIP on disk.
2. Confirm `run.sh` is at the ZIP root (no nested folder).
3. Confirm Linux parity artefact (`linux_parity_report_preppo.json`) shows `passed: true`.
4. Upload **manually** via the competition portal using your own credentials.
5. Do not store competition credentials in this repository.
6. After portal accept, fill `submission/UPLOAD_RECORD_heuristic_v2_preppo.md`.

## Safety

- Do not overwrite `heuristic_v1_packaged.zip`.
- Do not mark `SUBMITTED` until the operator completes portal upload.
- Do not start PPO until `SUBMITTED` is recorded.
