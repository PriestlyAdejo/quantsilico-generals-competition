"""Typed learned-policy forward result shared by PPO, validation, Arena, Env Lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


REQUIRED_KEYS = ("logits", "value", "hidden")


@dataclass(frozen=True)
class ModelForwardResult:
    """Canonical interpretation of ``forward_tensors`` output."""

    logits: Tensor
    value: Tensor
    hidden: Tensor
    cell_memory: Tensor | None = None
    raw: Mapping[str, Any] | None = None

    def require_finite(self) -> None:
        if not torch.isfinite(self.logits).all():
            raise ValueError("NONFINITE_LOGITS: model logits contain NaN/Inf")
        if not torch.isfinite(self.value).all():
            raise ValueError("NONFINITE_VALUE: model value contains NaN/Inf")
        if not torch.isfinite(self.hidden).all():
            raise ValueError("NONFINITE_HIDDEN: model hidden state contains NaN/Inf")


class MalformedModelOutputError(ValueError):
    """Structured protocol fault: model output cannot be interpreted."""


def adapt_forward_output(out: Any) -> ModelForwardResult:
    """Adapt model forward output into a typed result.

    Accepts only mapping/dict-like outputs with required keys.
    Tuple/list numeric indexing is rejected — that path caused the INITIAL
    validation protocol-fault storm (KeyError → PASS fallback per turn).
    """
    if isinstance(out, ModelForwardResult):
        return out
    if isinstance(out, (tuple, list)):
        raise MalformedModelOutputError(
            "MODEL_SHAPE_ERROR: forward_tensors returned a sequence; "
            "callers must not index model output as a tuple. Expected dict with "
            f"keys {REQUIRED_KEYS}."
        )
    if not isinstance(out, Mapping):
        raise MalformedModelOutputError(
            f"MODEL_SHAPE_ERROR: forward_tensors returned {type(out).__name__}; expected Mapping"
        )
    missing = [k for k in REQUIRED_KEYS if k not in out]
    if missing:
        raise MalformedModelOutputError(
            f"MODEL_SHAPE_ERROR: forward output missing required keys {missing}; "
            f"present={sorted(out.keys())}"
        )
    cell = out.get("cell_memory")
    result = ModelForwardResult(
        logits=out["logits"],
        value=out["value"],
        hidden=out["hidden"],
        cell_memory=cell if isinstance(cell, Tensor) else None,
        raw=out,
    )
    result.require_finite()
    return result
