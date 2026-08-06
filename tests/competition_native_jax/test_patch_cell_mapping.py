"""Patch-to-cell mapping gate tests."""

from __future__ import annotations

import numpy as np

from generals_bot.competition_native_jax.constants import MAX_HW, NUM_PATCHES, PATCH
from generals_bot.competition_native_jax.patchify import (
    all_cell_mappings,
    cell_to_patch_local,
    patch_local_to_cell,
    unpatchify_build_logits,
    unpatchify_move_logits,
)


def test_bijective_mapping() -> None:
    mappings = all_cell_mappings()
    assert len(mappings) == MAX_HW * MAX_HW
    cells = {(r, c) for r, c, *_ in mappings}
    assert len(cells) == MAX_HW * MAX_HW
    for r, c, pr, pc, lr, lc in mappings:
        assert patch_local_to_cell(pr, pc, lr, lc) == (r, c)
        assert cell_to_patch_local(r, c) == (pr, pc, lr, lc)


def test_unpatchify_single_hot() -> None:
    patch_move = np.zeros((NUM_PATCHES, PATCH, PATCH, 8), dtype=np.float32)
    # Set patch 0 local (1,2) direction slot 3
    patch_move[0, 1, 2, 3] = 1.0
    move = unpatchify_move_logits(patch_move)
    assert move[1, 2, 3] == 1.0
    assert move.sum() == 1.0

    patch_build = np.zeros((NUM_PATCHES, PATCH, PATCH), dtype=np.float32)
    patch_build[5, 0, 1] = 2.0
    build = unpatchify_build_logits(patch_build)
    pr, pc = divmod(5, 7)
    r, c = pr * 3 + 0, pc * 3 + 1
    assert build[r, c] == 2.0
