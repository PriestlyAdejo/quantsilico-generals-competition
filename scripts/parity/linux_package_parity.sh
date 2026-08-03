#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PACKAGE_ZIP="${1:?usage: linux_package_parity.sh package.zip [report.json]}"
REPORT_OUT="${2:-$ROOT/experiments/manifests/linux_parity_report.json}"
exec python -u "$ROOT/scripts/parity/linux_package_parity.py" "$PACKAGE_ZIP" --report "$REPORT_OUT"
