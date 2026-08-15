"""Registry unit tests (Stage 3 minimum canonical registry)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from generals_bot.marathon_registry import (
    SCHEMA_VERSION,
    Registry,
    RegistryError,
    canonical_id,
)

LINEAGE = {"NAME": "competition_native_jax_v1", "IMPLEMENTATION_FINGERPRINT": "a" * 16}
HASH = "a" * 64


def _experiment(registry: Registry, name: str = "exp-a", **overrides) -> str:
    record = {
        "KIND": "experiment",
        "ID": canonical_id("experiment", name, "material-1"),
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "NAME": name,
        "PPO_SEMANTICS": "UNCHANGED",
        "LINEAGE": LINEAGE,
        "CONFIG_IDENTITY": {"config": "configs/training/x.yaml", "hash": HASH},
        "SEEDS": [1, 2],
        "EVIDENCE_LINKS": ["EV-0001"],
    }
    record.update(overrides)
    return registry.add(record)


def _run(registry: Registry, experiment_id: str, name: str = "run-a") -> str:
    return registry.add(
        {
            "KIND": "run",
            "ID": canonical_id("run", name, "material-1"),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "EXPERIMENT_ID": experiment_id,
            "COMMAND": "python train.py",
            "BUDGET": {"transitions": 1000},
            "STOP_REASON": "BUDGET_REACHED",
            "ENVIRONMENT": {"python": "3.12.10"},
            "ARTEFACT_LOCATIONS": ["var/run-a"],
        }
    )


def _checkpoint(registry: Registry, run_id: str, name: str = "ckpt-a") -> str:
    return registry.add(
        {
            "KIND": "checkpoint",
            "ID": canonical_id("checkpoint", name, "material-1"),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "RUN_ID": run_id,
            "ARTEFACT_HASHES": {"raw.npz": HASH},
            "LINEAGE": LINEAGE,
            "TRANSITIONS": 1000,
            "ARTEFACT_LOCATIONS": ["models/ckpt-a"],
        }
    )


def _candidate(registry: Registry, checkpoint_id: str, name: str = "cand-a") -> str:
    return registry.add(
        {
            "KIND": "candidate",
            "ID": canonical_id("candidate", name, "material-1"),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "CHECKPOINT_ID": checkpoint_id,
            "PPO_SEMANTICS": "EVAL_ONLY",
            "EVIDENCE_LINKS": ["EV-0019"],
        }
    )


def test_canonical_ids_are_stable_and_kind_checked() -> None:
    first = canonical_id("checkpoint", "baseline-v0", "material")
    assert first == canonical_id("checkpoint", "baseline-v0", "material")
    assert first.startswith("checkpoint#baseline-v0#")
    assert canonical_id("checkpoint", "baseline-v0", "other") != first
    with pytest.raises(RegistryError):
        canonical_id("wizard", "x", "m")
    with pytest.raises(RegistryError):
        canonical_id("checkpoint", "9bad", "m")


def test_full_chain_persists_and_is_discoverable(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry")
    exp = _experiment(registry)
    run = _run(registry, exp)
    ckpt = _checkpoint(registry, run)
    cand = _candidate(registry, ckpt)
    evaln = registry.add(
        {
            "KIND": "evaluation",
            "ID": canonical_id("evaluation", "eval-a", "material-1"),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "CANDIDATE_ID": cand,
            "EVALUATOR_IDENTITY": "marathon_paired_evaluator_v1",
            "EVAL_PROTOCOL": "SEAT_SWAPPED_PAIRS_ANYTIME_VALID_CS",
            "RESULTS_LOCATION": "experiments/marathon/paired_eval_runs/x",
            "EVIDENCE_LINKS": ["EV-0019"],
        }
    )
    registry.add(
        {
            "KIND": "opponent_reference",
            "ID": canonical_id("opponent_reference", "legal-random", "material-1"),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "NAME": "legal_random",
            "SOURCE_IDENTITY": {"path": "baselines/legal_random/main.py"},
            "ARTEFACT_LOCATIONS": ["baselines/legal_random"],
        }
    )
    assert registry.list_ids("checkpoint") == [ckpt]
    assert registry.get(evaln)["CANDIDATE_ID"] == cand
    assert json.loads(
        (tmp_path / "registry/records" / f"{ckpt.replace('#', '__')}.json").read_text()
    )["TRANSITIONS"] == 1000


def test_silent_overwrite_refused(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry")
    exp = _experiment(registry)
    with pytest.raises(RegistryError, match="silent overwrite"):
        _experiment(registry)
    assert registry.list_ids("experiment") == [exp]


def test_ppo_semantics_enforced(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry")
    with pytest.raises(RegistryError, match="PPO_SEMANTICS"):
        _experiment(registry, PPO_SEMANTICS="AMBIGUOUS")
    with pytest.raises(RegistryError, match="missing fields"):
        registry.add(
            {
                "KIND": "experiment",
                "ID": canonical_id("experiment", "exp-b", "m"),
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "NAME": "exp-b",
                "LINEAGE": LINEAGE,
                "CONFIG_IDENTITY": {},
                "SEEDS": [],
                "EVIDENCE_LINKS": [],
            }
        )


def test_unresolved_hashes_and_lineage_rejected(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry")
    exp = _experiment(registry)
    run = _run(registry, exp)
    with pytest.raises(RegistryError, match="ARTEFACT_HASHES"):
        registry.add(
            {
                "KIND": "checkpoint",
                "ID": canonical_id("checkpoint", "bad-hash", "m"),
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "RUN_ID": run,
                "ARTEFACT_HASHES": {"raw.npz": "not-a-hash"},
                "LINEAGE": LINEAGE,
                "TRANSITIONS": 1,
                "ARTEFACT_LOCATIONS": [],
            }
        )
    with pytest.raises(RegistryError, match="LINEAGE"):
        registry.add(
            {
                "KIND": "checkpoint",
                "ID": canonical_id("checkpoint", "bad-lineage", "m"),
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "RUN_ID": run,
                "ARTEFACT_HASHES": {"raw.npz": HASH},
                "LINEAGE": {"FINGERPRINT_ONLY": True},
                "TRANSITIONS": 1,
                "ARTEFACT_LOCATIONS": [],
            }
        )


def test_dangling_cross_references_rejected(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry")
    ghost = "experiment#ghost#000000000000"
    with pytest.raises(RegistryError, match="EXPERIMENT_ID unresolved"):
        _run(registry, ghost, name="orphan-run")
    with pytest.raises(RegistryError, match="CHECKPOINT_ID unresolved"):
        _candidate(registry, "checkpoint#ghost#000000000000", name="orphan-cand")


def test_evaluation_requires_evaluator_identity(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry")
    exp = _experiment(registry)
    run = _run(registry, exp)
    ckpt = _checkpoint(registry, run)
    cand = _candidate(registry, ckpt)
    with pytest.raises(RegistryError, match="EVALUATOR_IDENTITY"):
        registry.add(
            {
                "KIND": "evaluation",
                "ID": canonical_id("evaluation", "eval-b", "m"),
                "SCHEMA_VERSION": SCHEMA_VERSION,
                "CANDIDATE_ID": cand,
                "EVALUATOR_IDENTITY": "   ",
                "EVAL_PROTOCOL": "SEAT_SWAPPED_PAIRS_ANYTIME_VALID_CS",
                "RESULTS_LOCATION": "x",
                "EVIDENCE_LINKS": [],
            }
        )


def test_schema_version_enforced(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry")
    with pytest.raises(RegistryError, match="SCHEMA_VERSION"):
        _experiment(registry, SCHEMA_VERSION="0.0.1", name="old-schema")
