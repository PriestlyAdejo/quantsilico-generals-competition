"""Patchify / unpatchify utilities for 21x21 boards with 3x3 patches."""

from __future__ import annotations

import numpy as np

from generals_bot.competition_native_jax.constants import MAX_HW, NUM_PATCHES, PATCH, PATCH_GRID


def patch_coords(patch_index: int) -> tuple[int, int]:
    if not (0 <= patch_index < NUM_PATCHES):
        raise ValueError(patch_index)
    pr, pc = divmod(patch_index, PATCH_GRID)
    return pr, pc


def cell_to_patch_local(row: int, col: int) -> tuple[int, int, int, int]:
    """Return (patch_r, patch_c, local_r, local_c)."""
    if not (0 <= row < MAX_HW and 0 <= col < MAX_HW):
        raise ValueError((row, col))
    pr, lr = divmod(row, PATCH)
    pc, lc = divmod(col, PATCH)
    return pr, pc, lr, lc


def patch_local_to_cell(pr: int, pc: int, lr: int, lc: int) -> tuple[int, int]:
    return pr * PATCH + lr, pc * PATCH + lc


def all_cell_mappings() -> list[tuple[int, int, int, int, int, int]]:
    """List (row, col, pr, pc, lr, lc) for every cell — bijective check helper."""
    out: list[tuple[int, int, int, int, int, int]] = []
    for r in range(MAX_HW):
        for c in range(MAX_HW):
            pr, pc, lr, lc = cell_to_patch_local(r, c)
            out.append((r, c, pr, pc, lr, lc))
    return out


def unpatchify_move_logits(patch_move: np.ndarray) -> np.ndarray:
    """patch_move: [49, 3, 3, 8] -> [21, 21, 8]."""
    assert patch_move.shape == (NUM_PATCHES, PATCH, PATCH, 8)
    out = np.zeros((MAX_HW, MAX_HW, 8), dtype=patch_move.dtype)
    for p in range(NUM_PATCHES):
        pr, pc = patch_coords(p)
        r0, c0 = pr * PATCH, pc * PATCH
        out[r0 : r0 + PATCH, c0 : c0 + PATCH, :] = patch_move[p]
    return out


def unpatchify_build_logits(patch_build: np.ndarray) -> np.ndarray:
    """patch_build: [49, 3, 3] -> [21, 21]."""
    assert patch_build.shape == (NUM_PATCHES, PATCH, PATCH)
    out = np.zeros((MAX_HW, MAX_HW), dtype=patch_build.dtype)
    for p in range(NUM_PATCHES):
        pr, pc = patch_coords(p)
        r0, c0 = pr * PATCH, pc * PATCH
        out[r0 : r0 + PATCH, c0 : c0 + PATCH] = patch_build[p]
    return out


def pack_flat_logits(move: np.ndarray, build: np.ndarray, pass_logit: float) -> np.ndarray:
    """Pack [21,21,8], [21,21], pass into ACTION_DIM vector."""
    from generals_bot.competition_native_jax.action_codec import build_index, move_index
    from generals_bot.competition_native_jax.constants import ACTION_DIM, PASS_INDEX

    logits = np.full((ACTION_DIM,), -1e9, dtype=np.float64)
    logits[PASS_INDEX] = float(pass_logit)
    for r in range(MAX_HW):
        for c in range(MAX_HW):
            for d in range(4):
                for s in range(2):
                    logits[move_index(r, c, d, s)] = float(move[r, c, d * 2 + s])
            logits[build_index(r, c)] = float(build[r, c])
    return logits
