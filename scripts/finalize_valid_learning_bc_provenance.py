#!/usr/bin/env python3
"""Attach immutable runtime/opponent identities to a completed BC dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from collect_valid_learning_bc import collection_source_hashes, file_sha256

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_manifest", type=Path)
    args = parser.parse_args()
    report = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    dataset = Path(report["dataset"])
    if report.get("status") != "PASS" or not dataset.is_file():
        raise RuntimeError("only a completed PASS dataset can be finalized")
    if file_sha256(dataset) != report["dataset_sha256"]:
        raise RuntimeError("dataset hash changed before provenance finalization")
    report["source_hashes"] = collection_source_hashes()
    report["provenance_finalized"] = True
    atomic_json(args.dataset_manifest, report)
    atomic_json(
        ROOT / "experiments" / "manifests" / "valid_learning_bc_dataset.json",
        report,
    )
    print(json.dumps(report["source_hashes"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
