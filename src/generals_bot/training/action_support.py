"""Action-support hashing and replay checks for PPO likelihood ratios."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

SUPPORT_KIND_FULL_LEGAL_MASK = "FULL_ACTION_SPACE_LEGAL_MASK"
MAX_EPISODE_TURN = 1200


def support_hash_from_mask(mask: np.ndarray | torch.Tensor) -> str:
    """Stable short hash of a boolean action-support mask."""
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().to(dtype=torch.bool, device="cpu").numpy()
    else:
        arr = np.asarray(mask, dtype=bool)
    packed = np.packbits(arr.reshape(-1))
    return hashlib.sha256(packed.tobytes()).hexdigest()[:16]


def masks_equal(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> bool:
    if isinstance(a, torch.Tensor):
        a = a.detach().to(dtype=torch.bool, device="cpu").numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().to(dtype=torch.bool, device="cpu").numpy()
    return bool(np.array_equal(np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)))
