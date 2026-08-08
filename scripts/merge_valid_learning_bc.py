#!/usr/bin/env python3
"""Merge independently hashed whole-game BC corpora without split leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from collect_valid_learning_bc import file_sha256

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / "quantsilico-runtime" / "cloud_valid_learning_recovery_v1"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_verified(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads(path.with_name("dataset_manifest.json").read_text(encoding="utf-8"))
    actual = file_sha256(path)
    if manifest.get("status") != "PASS" or actual != manifest.get("dataset_sha256"):
        raise RuntimeError(f"input dataset verification failed: {path}")
    raw = np.load(path, allow_pickle=False)
    return manifest, {name: raw[name] for name in raw.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--validation-existing-split", type=Path, nargs="*", default=[]
    )
    args = parser.parse_args()
    sources: list[dict] = []
    arrays: dict[str, list[np.ndarray]] = {}
    roles: list[np.ndarray] = []
    groups = (
        ("train", args.train, None),
        ("validation", args.validation, None),
        ("validation", args.validation_existing_split, "validation"),
    )
    for role, paths, existing_split in groups:
        for path in paths:
            manifest, data = load_verified(path)
            if existing_split is not None:
                selected = data["split"] == existing_split
                data = {name: value[selected] for name, value in data.items()}
            count = len(data["teacher_action"])
            sources.append(
                {
                    "role": role,
                    "dataset": str(path),
                    "dataset_sha256": manifest["dataset_sha256"],
                    "samples": count,
                    "source_split_filter": existing_split,
                }
            )
            for name, value in data.items():
                if name != "split":
                    arrays.setdefault(name, []).append(value)
            roles.append(np.full((count,), role, dtype="U10"))
    if not roles or not any(source["role"] == "validation" for source in sources):
        raise RuntimeError("at least one validation corpus or split is required")
    merged = {name: np.concatenate(values, axis=0) for name, values in arrays.items()}
    merged["split"] = np.concatenate(roles)
    identities = merged["sample_id"].astype(str)
    _, first = np.unique(identities, return_index=True)
    keep = np.sort(first)
    duplicates_removed = len(identities) - len(keep)
    merged = {name: value[keep] for name, value in merged.items()}

    train_games = set(merged["game_id"][merged["split"] == "train"].astype(str))
    validation_games = set(
        merged["game_id"][merged["split"] == "validation"].astype(str)
    )
    if train_games & validation_games:
        raise RuntimeError("complete-game split leakage detected")
    actions = merged["teacher_action"].astype(np.int64)
    if not np.all(merged["legal_mask"][np.arange(len(actions)), actions]):
        raise RuntimeError("merged corpus contains illegal teacher target")
    if not np.isfinite(merged["spatial"]).all() or not np.isfinite(
        merged["global_vec"]
    ).all():
        raise RuntimeError("merged corpus contains nonfinite tensors")
    if set(merged["opponent"].astype(str)) != {
        "pass",
        "legal_random",
        "official_expander",
        "official_hunter",
    } or set(merged["seat"].astype(int)) != {0, 1}:
        raise RuntimeError("merged all-four-opponent/both-seat coverage failed")

    source_identity = hashlib.sha256(
        json.dumps(sources, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output = RUNTIME / "bc" / f"dataset_merged_{source_identity[:16]}"
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "teacher_states.npz"
    np.savez_compressed(dataset, **merged)
    dataset_sha = file_sha256(dataset)
    split = merged["split"]
    report = {
        "schema_version": 1,
        "kind": "VALID_LEARNING_BC_DATASET_MERGED",
        "status": "PASS",
        "dataset_id": f"BC_VALID_LEARNING_{dataset_sha[:16]}",
        "dataset": str(dataset),
        "dataset_sha256": dataset_sha,
        "unique_recurrent_states": len(merged["teacher_action"]),
        "duplicates_removed": duplicates_removed,
        "sources": sources,
        "coverage": {
            "opponents": sorted(set(merged["opponent"].astype(str))),
            "seats": sorted(set(map(int, merged["seat"]))),
            "train_samples": int(np.count_nonzero(split == "train")),
            "validation_samples": int(np.count_nonzero(split == "validation")),
            "train_games": len(train_games),
            "validation_games": len(validation_games),
            "train_validation_game_overlap": 0,
            "legal_teacher_target_fraction": 1.0,
            "finite_tensor_fraction": 1.0,
        },
        "written_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(output / "dataset_manifest.json", report)
    atomic_json(ROOT / "experiments/manifests/valid_learning_bc_dataset_merged.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
