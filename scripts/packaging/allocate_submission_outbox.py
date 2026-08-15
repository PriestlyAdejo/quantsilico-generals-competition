"""Allocate a canonical submission outbox slot from a promoted package ZIP.

Usage:
  python scripts/packaging/allocate_submission_outbox.py \
      --candidate-id QS-PUBLIC-V001 \
      --source-zip submission/packages/QS-PUBLIC-V001/<hash>/package.zip \
      [--date 2026-08-15]

Stage 4B packaging lane: dependency-safe, never uploads, never touches
live training files.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from generals_bot.submission.outbox import allocate_outbox  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today UTC)")
    parser.add_argument("--root", default=None, type=Path, help="outbox root override")
    args = parser.parse_args()

    allocation = allocate_outbox(
        args.candidate_id,
        args.source_zip,
        date=args.date,
        root=args.root,
        provenance={"allocator": "scripts/packaging/allocate_submission_outbox.py"},
    )
    payload = asdict(allocation)
    payload["zip_path"] = str(allocation.zip_path)
    payload["manifest_path"] = str(allocation.manifest_path)
    payload["sidecar_path"] = str(allocation.sidecar_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
