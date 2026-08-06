"""Control modules — default OFF; telemetry must not change actions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TelemetryState:
    records: list[dict] = field(default_factory=list)

    def record(self, **kwargs: object) -> None:
        self.records.append(dict(kwargs))


def control_off(logits: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Identity controller."""
    return logits, mask


def apply_static_risk(
    logits: np.ndarray,
    mask: np.ndarray,
    *,
    risk_bias: float = 0.0,
    pass_boost: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded STATIC_RISK: optional PASS boost; does not invent legality."""
    out = logits.copy()
    out = out - float(risk_bias)
    out[0] = out[0] + float(pass_boost)
    return out, mask


def passive_telemetry_noninterference(
    logits: np.ndarray,
    mask: np.ndarray,
    telemetry: TelemetryState,
) -> tuple[np.ndarray, np.ndarray]:
    """Record stats without mutating logits/mask."""
    telemetry.record(legal_count=int(mask.sum()), logit_mean=float(logits[mask].mean()) if mask.any() else 0.0)
    return logits, mask
