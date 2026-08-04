"""Versioned training telemetry field catalogue (producers must emit these)."""

from __future__ import annotations

TELEMETRY_SCHEMA_VERSION = 1

# Fields that future DEVELOPMENT runs should emit when measured.
PPO_UPDATE_FIELDS = (
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "grad_norm",
    "learning_rate",
    "advantage_mean",
    "advantage_std",
)

BC_METRIC_FIELDS = (
    "train_loss",
    "train_action_acc",
    "val_action_acc",
    "val_legal_action_rate",
)

MISSING = "NOT RECORDED"


def annotate_history(history: list[dict] | None, *, producer: str) -> dict:
    """Wrap existing history without inventing values."""
    if not history:
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "producer": producer,
            "points": [],
            "missing": list(PPO_UPDATE_FIELDS),
            "note": MISSING,
        }
    present: set[str] = set()
    for row in history:
        present.update(k for k, v in row.items() if v is not None)
    missing = [f for f in PPO_UPDATE_FIELDS if f not in present]
    return {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "producer": producer,
        "points": history,
        "missing": missing,
        "note": None if not missing else f"{MISSING}: {', '.join(missing)} — extend PPO trainer to emit these fields",
    }
