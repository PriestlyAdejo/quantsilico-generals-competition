"""Versioned, reload-verified checkpoints for valid-learning recovery."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from generals_bot.competition_native_jax.competition_env_jax import ObsMemoryJax
from train.competition_native_jax.rollout_curriculum_jax import CurriculumCarry
from train.competition_native_jax.train_jax import load_tree, save_tree


class PersistedCurriculumCarry(NamedTuple):
    states: Any
    mem0: ObsMemoryJax
    mem1: ObsMemoryJax
    key: Any
    pool_cursor: Any
    learner_seat: Any
    episode_id: Any


ARTIFACTS = (
    "raw.npz",
    "ema.npz",
    "opt_state.npz",
    "rollout_carry.npz",
    "frozen_opponent.npz",
    "meta.json",
)
FINAL_FILES = frozenset((*ARTIFACTS, "manifest.json", "COMPLETE"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def persisted_carry(carry: CurriculumCarry) -> PersistedCurriculumCarry:
    return PersistedCurriculumCarry(
        carry.states,
        carry.mem0,
        carry.mem1,
        carry.key,
        carry.pool_cursor,
        carry.learner_seat,
        carry.episode_id,
    )


def restore_carry(
    saved: PersistedCurriculumCarry,
    *,
    params: dict,
    pool: Any,
    frozen_opponent_params: dict,
) -> CurriculumCarry:
    return CurriculumCarry(
        saved.states,
        saved.mem0,
        saved.mem1,
        saved.key,
        params,
        pool,
        saved.pool_cursor,
        saved.learner_seat,
        saved.episode_id,
        frozen_opponent_params,
    )


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_checkpoint(
    path: Path,
    *,
    params_like: dict,
    opt_state_like: Any,
    carry_like: PersistedCurriculumCarry,
) -> dict[str, Any]:
    names = frozenset(item.name for item in path.iterdir() if item.is_file())
    if names != FINAL_FILES:
        raise RuntimeError(f"checkpoint artifact set mismatch: {sorted(names)}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    complete = json.loads((path / "COMPLETE").read_text(encoding="utf-8"))
    if complete.get("status") != "COMPLETE":
        raise RuntimeError("checkpoint COMPLETE status invalid")
    for field in ("update", "transitions", "programme_transitions"):
        if complete.get(field) != meta.get(field):
            raise RuntimeError(f"checkpoint COMPLETE/meta mismatch: {field}")
    expected = manifest.get("artifacts") or {}
    if set(expected) != set(ARTIFACTS):
        raise RuntimeError("checkpoint manifest artifact keys mismatch")
    for name in ARTIFACTS:
        artifact = path / name
        record = expected[name]
        if artifact.stat().st_size != int(record["size"]):
            raise RuntimeError(f"checkpoint size mismatch: {name}")
        if sha256_file(artifact) != record["sha256"]:
            raise RuntimeError(f"checkpoint SHA mismatch: {name}")
    load_tree(path / "raw.npz", params_like)
    load_tree(path / "ema.npz", params_like)
    load_tree(path / "opt_state.npz", opt_state_like)
    load_tree(path / "rollout_carry.npz", carry_like)
    load_tree(path / "frozen_opponent.npz", params_like)
    return {
        "status": "PASS",
        "path": str(path),
        "update": meta["update"],
        "transitions": meta["transitions"],
        "artifact_count": len(ARTIFACTS),
    }


def save_checkpoint(
    root: Path,
    *,
    tag: str,
    params: dict,
    ema: dict,
    opt_state: Any,
    carry: CurriculumCarry,
    meta: dict[str, Any],
) -> Path:
    update = int(meta["update"])
    transitions = int(meta["transitions"])
    name = f"ckpt_{tag}_u{update}_t{transitions}"
    destination = root / name
    staging = root / f".{name}.tmp-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    if destination.exists() or staging.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint generation: {name}")
    staging.mkdir()
    saved_carry = persisted_carry(carry)
    save_tree(staging / "raw.npz", params)
    save_tree(staging / "ema.npz", ema)
    save_tree(staging / "opt_state.npz", opt_state)
    save_tree(staging / "rollout_carry.npz", saved_carry)
    save_tree(staging / "frozen_opponent.npz", carry.frozen_opponent_params)
    metadata = {
        **meta,
        "checkpoint_schema_version": 2,
        "rollout_carry_included": True,
        "written_at": datetime.now(UTC).isoformat(),
    }
    (staging / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for artifact in ARTIFACTS:
        _fsync_file(staging / artifact)
    manifest = {
        "schema_version": 2,
        "artifacts": {
            artifact: {
                "size": (staging / artifact).stat().st_size,
                "sha256": sha256_file(staging / artifact),
            }
            for artifact in ARTIFACTS
        },
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _fsync_file(staging / "manifest.json")
    complete = {
        "status": "COMPLETE",
        "update": metadata["update"],
        "transitions": metadata["transitions"],
        "programme_transitions": metadata["programme_transitions"],
        "written_at": datetime.now(UTC).isoformat(),
    }
    (staging / "COMPLETE").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _fsync_file(staging / "COMPLETE")
    _fsync_directory(staging)
    os.replace(staging, destination)
    _fsync_directory(root)
    verify_checkpoint(
        destination,
        params_like=params,
        opt_state_like=opt_state,
        carry_like=saved_carry,
    )
    return destination
