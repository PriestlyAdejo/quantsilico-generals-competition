"""Canonical full-support action codec: PASS + 9 actions per padded cell."""

from __future__ import annotations

from generals_bot.action import KIND_BUILD, KIND_MOVE, KIND_PASS, Action
from generals_bot.competition_native_jax.constants import (
    ACTION_DIM,
    ACTIONS_PER_CELL,
    MAX_HW,
    MOVES_PER_CELL,
    NUM_DIRECTIONS,
    NUM_SPLITS,
    PASS_INDEX,
)


def cell_base(row: int, col: int) -> int:
    return 1 + ACTIONS_PER_CELL * (row * MAX_HW + col)


def move_index(row: int, col: int, direction: int, split: int) -> int:
    if not (0 <= direction < NUM_DIRECTIONS and 0 <= split < NUM_SPLITS):
        raise ValueError("direction/split out of range")
    local = direction * NUM_SPLITS + split
    return cell_base(row, col) + local


def build_index(row: int, col: int) -> int:
    return cell_base(row, col) + MOVES_PER_CELL


def action_to_index(action: Action) -> int:
    if action.kind == KIND_PASS:
        return PASS_INDEX
    if action.kind == KIND_BUILD:
        return build_index(action.row, action.col)
    if action.kind == KIND_MOVE:
        return move_index(action.row, action.col, action.direction, action.split)
    raise ValueError(f"unknown kind {action.kind}")


def index_to_action(index: int) -> Action:
    if index == PASS_INDEX:
        return Action(kind=KIND_PASS)
    if not (0 <= index < ACTION_DIM):
        raise ValueError(f"index out of range: {index}")
    flat = index - 1
    cell, local = divmod(flat, ACTIONS_PER_CELL)
    row, col = divmod(cell, MAX_HW)
    if local == MOVES_PER_CELL:
        return Action(kind=KIND_BUILD, row=row, col=col)
    direction, split = divmod(local, NUM_SPLITS)
    return Action(kind=KIND_MOVE, row=row, col=col, direction=direction, split=split)
