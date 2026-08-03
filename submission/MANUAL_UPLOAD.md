# Manual upload instructions — heuristic_v1

Do **not** upload automatically from CI or the dashboard.

## Preconditions

1. Windows clean-package validation: PASS
2. Linux parity report: PASS (`experiments/manifests/linux_parity_report.json`)
3. Package report status: `UPLOAD_READY`
4. Engine submodule pin matches the competition evaluation pin

## Package location

- ZIP: `submission/packages/heuristic_v1_packaged.zip`
- Report: `submission/packages/heuristic_v1_packaged.report.json`

## Operator steps

1. Verify SHA-256 in the package report matches the ZIP on disk.
2. Confirm `run.sh` is at the ZIP root (no nested folder).
3. Confirm Linux parity artefact from GitHub Actions or local Docker run.
4. Upload **manually** via the competition portal using your own credentials.
5. Do not store competition credentials in this repository.

## Safety

- Champion remains `heuristic_v1` until a learned challenger passes the full promotion gate.
- A Windows-only `PACKAGED` artefact must not be labelled `UPLOAD_READY`.
