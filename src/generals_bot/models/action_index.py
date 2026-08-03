"""Flat action-index mappings for padded MAX_HW boards."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from generals_bot.action import KIND_BUILD, KIND_MOVE, KIND_PASS, Action
from generals_bot.models.observation_encoder import MAX_HW

NUM_DIRECTIONS = 4
NUM_SPLITS = 2
PASS_INDEX = 0
BUILD_OFFSET = 1
BUILD_COUNT = MAX_HW * MAX_HW
MOVE_OFFSET = BUILD_OFFSET + BUILD_COUNT
MOVE_COUNT = MAX_HW * MAX_HW * NUM_DIRECTIONS * NUM_SPLITS
ACTION_DIM = MOVE_OFFSET + MOVE_COUNT


def build_index(row: int, col: int) -> int:
    return BUILD_OFFSET + row * MAX_HW + col


def move_index(row: int, col: int, direction: int, split: int) -> int:
    cell = row * MAX_HW + col
    return MOVE_OFFSET + ((cell * NUM_DIRECTIONS + direction) * NUM_SPLITS + split)


def action_to_index(action: Action) -> int:
    if action.kind == KIND_PASS:
        return PASS_INDEX
    if action.kind == KIND_BUILD:
        return build_index(action.row, action.col)
    if action.kind == KIND_MOVE:
        return move_index(action.row, action.col, action.direction, action.split)
    raise ValueError(f"unknown action kind {action.kind}")


def index_to_action(index: int) -> Action:
    if index == PASS_INDEX:
        return Action(kind=KIND_PASS)
    if BUILD_OFFSET <= index < MOVE_OFFSET:
        flat = index - BUILD_OFFSET
        row, col = divmod(flat, MAX_HW)
        return Action(kind=KIND_BUILD, row=row, col=col)
    if MOVE_OFFSET <= index < ACTION_DIM:
        flat = index - MOVE_OFFSET
        split = flat % NUM_SPLITS
        flat //= NUM_SPLITS
        direction = flat % NUM_DIRECTIONS
        flat //= NUM_DIRECTIONS
        row, col = divmod(flat, MAX_HW)
        return Action(kind=KIND_MOVE, row=row, col=col, direction=direction, split=split)
    raise ValueError(f"action index out of range: {index}")


@lru_cache(maxsize=1)
def direction_deltas() -> np.ndarray:
    # Matches protocol DIRECTIONS order: N, E, S, W typically — verify against protocol.
    from generals_bot.protocol import DIRECTIONS

    return np.asarray(DIRECTIONS, dtype=np.int32)
