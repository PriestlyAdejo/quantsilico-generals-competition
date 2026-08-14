"""Semantic-state hashing for Marathon baseline checkpoints (EXECUTION_PLAN §6.3).

Computes ``SEMANTIC_STATE_SHA256`` for each npz artefact of a checkpoint:
a canonical digest over deterministically ordered array keys where each entry
contributes key name, dtype, shape, and canonical contiguous little-endian
value bytes. Container/serialization metadata (zip headers, compression,
array order) does not affect the digest, so re-serialization alone cannot
imply learner-state drift.

File hashes (``FILE_SHA256``) are reported alongside for the separate
file-identity evidence class.

Usage:
    python scripts/analysis/semantic_state_hash.py <checkpoint_dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def semantic_state_sha256(npz_path: Path) -> dict[str, str]:
    """Return per-array and whole-state semantic digests for one npz artefact."""
    data = np.load(npz_path, allow_pickle=False)
    digest = hashlib.sha256()
    per_array: dict[str, str] = {}
    for key in sorted(data.keys()):
        arr = np.ascontiguousarray(data[key])
        entry = hashlib.sha256()
        entry.update(key.encode("utf-8"))
        entry.update(b"|")
        entry.update(str(arr.dtype).encode("utf-8"))
        entry.update(b"|")
        entry.update(json.dumps(list(arr.shape)).encode("utf-8"))
        entry.update(b"|")
        entry.update(arr.tobytes(order="C"))
        array_digest = entry.hexdigest()
        per_array[key] = array_digest
        digest.update(array_digest.encode("utf-8"))
    return {"state": digest.hexdigest(), "arrays": per_array}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    args = parser.parse_args()

    ckpt = args.checkpoint_dir.resolve()
    if not ckpt.is_dir():
        print(f"checkpoint directory not found: {ckpt}", file=sys.stderr)
        return 2

    report: dict[str, dict[str, object]] = {}
    for npz in sorted(ckpt.glob("*.npz")):
        file_hash = hashlib.sha256(npz.read_bytes()).hexdigest()
        semantic = semantic_state_sha256(npz)
        report[npz.name] = {
            "FILE_SHA256": file_hash,
            "SEMANTIC_STATE_SHA256": semantic["state"],
            "ARRAY_COUNT": len(semantic["arrays"]),
        }

    output = {
        "schema_version": 1,
        "checkpoint": str(ckpt),
        "artefacts": report,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
