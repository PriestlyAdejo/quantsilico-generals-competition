"""Competition-engine oracle for elite replay action semantics.

AUTHORITY (operator amendment 2026-08-16, EV-0042): action validity,
observation semantics and replay interpretation are defined by the PINNED
competition engine (third_party/generals-bots @ its submodule SHA), then the
official generals.bot competition rules/docs, then QuantSilico parity
contracts - NEVER generic present-day generals.io behaviour.

This module provides:

1. TRUE_COMPETITION_STATE reconstruction: rebuild the pinned engine's
   GameState from a full-state replay tick (plus static terrain). Growth
   phase (time) is supplied by the caller; alignment evidence decides it.
   The full state is an ENGINE ORACLE / offline-analysis source only - it
   must never become policy input (LEGAL_PLAYER_OBSERVATION is the separate
   replay_legal_pov pipeline).

2. Engine-exact action classification, mirroring the pinned predicates:
   - MOVE validity (game.py _execute_move): src in bounds, dest in bounds,
     src owned, army_to_move > 0 (a-1 whole / a//2 half, clamped to a-1),
     dest passable. Invalid-but-well-formed => ENGINE_SILENT_PASS (no-op).
   - BUILD validity (modifiers/build_castles.py _apply_one): in bounds,
     owned, plain (not general/castle), army >= dynamic cost
     (35 + sum over own structures max(0, 14 - 2*manhattan)), alive.
     Invalid build is consumed as a pass; builds resolve before moves.
   - PASS always executes.
   These are NOT runner faults. Runner faults (malformed/late/missing
   replies) and process forfeits are NOT reconstructable from replay
   payloads - audits must say so explicitly, never fabricate them.

3. Protocol encoding of extracted events into the five-integer competition
   action (kind row col dir split); events without a coherent encoding are
   UNCLASSIFIABLE, never silently coerced.

Replays record absolute player indices (0/1, -1 unowned); perspective
relative (1=ME/2=OPPONENT) encoding is the protocol layer's job - audits
here stay absolute and seat tests cover both players. Real board dims are
tracked separately from any 21x21 training padding: coordinates outside the
real H/W are OUT_OF_BOUNDS, never "mountain in padding".
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from generals.core.game import DIRECTIONS, GameState

ENGINE_SUBMODULE_SHA = "9e3b9d13cca51caa1bb07db48bb85c9e90ce0462"  # competition-engine-2026-15
RULESET = "competition"

# Classification categories (competition semantics; see module docstring).
PROTOCOL_VALID = "PROTOCOL_VALID"
PROTOCOL_UNCLASSIFIABLE = "PROTOCOL_UNCLASSIFIABLE"
ENGINE_EXECUTED = "ENGINE_EXECUTED"
ENGINE_SILENT_PASS = "ENGINE_SILENT_PASS"

MOVE_REASONS = (
    "src_out_of_bounds",
    "dest_out_of_bounds",
    "src_not_owned",
    "insufficient_source_army",
    "dest_mountain",
)
BUILD_REASONS = (
    "out_of_bounds",
    "not_owned",
    "not_plain",
    "insufficient_castle_price",
    "game_over",
)

# Build-cost constants: MUST stay in lockstep with the pinned engine's
# generals/modifiers/build_castles.py (differential tests enforce parity).
BASE_COST = 35
PROXIMITY_PENALTY = 14
PROXIMITY_DECAY = 2


@dataclass
class TrueCompetitionState:
    """Full engine state from a replay tick. Never feeds policy inputs."""

    engine_state: GameState
    real_h: int
    real_w: int
    time: int


@dataclass
class ActionClassification:
    protocol_status: str  # PROTOCOL_VALID | PROTOCOL_UNCLASSIFIABLE
    protocol_action: tuple[int, int, int, int, int] | None
    kind: str  # MOVE | BUILD | PASS
    engine_outcome: str  # ENGINE_EXECUTED | ENGINE_SILENT_PASS
    rejection_reasons: list[str]
    split: int | None = None  # move split bit actually encoded (0/1)


def _as_bool_grid(cells: set[tuple[int, int]], h: int, w: int, pad_to: int = 21) -> np.ndarray:
    grid = np.zeros((pad_to, pad_to), dtype=bool)
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = True
    return grid


def state_from_tick(
    tick: dict,
    *,
    dims: tuple[int, int],
    mountains: set[tuple[int, int]],
    castles: set[tuple[int, int]],
    generals: dict[int, tuple[int, int]],
    time: int,
    winner: int = -1,
    pad_to: int = 21,
) -> TrueCompetitionState:
    """Reconstruct the pinned engine GameState from one replay tick.

    The engine pads to 21x21 with mountains; real dims are recorded so that
    out-of-real-bounds coordinates classify as OUT_OF_BOUNDS, not mountain.
    """
    h, w = dims
    armies = np.zeros((pad_to, pad_to), dtype=np.int32)
    ownership = np.zeros((2, pad_to, pad_to), dtype=bool)
    ownership_neutral = np.zeros((pad_to, pad_to), dtype=bool)
    rows = tick["owners"]
    army_rows = tick["armies"]
    for r in range(h):
        for c in range(w):
            owner = rows[r][c]
            armies[r, c] = int(army_rows[r][c])
            if owner == 0:
                ownership[0, r, c] = True
            elif owner == 1:
                ownership[1, r, c] = True
            else:
                ownership_neutral[r, c] = True
    mountains_grid = _as_bool_grid(mountains, h, w, pad_to)
    for r in range(h, pad_to):
        mountains_grid[r, :] = True
    mountains_grid[:, w:] = True
    generals_grid = _as_bool_grid(set(generals.values()), h, w, pad_to)
    castles_grid = _as_bool_grid(castles, h, w, pad_to)
    general_positions = np.array(
        [list(generals.get(0, (-1, -1))), list(generals.get(1, (-1, -1)))], dtype=np.int32
    )
    engine_state = GameState(
        armies=jnp.asarray(armies),
        ownership=jnp.asarray(ownership),
        ownership_neutral=jnp.asarray(ownership_neutral),
        generals=jnp.asarray(generals_grid),
        castles=jnp.asarray(castles_grid),
        mountains=jnp.asarray(mountains_grid),
        passable=jnp.asarray(~mountains_grid),
        general_positions=jnp.asarray(general_positions),
        time=jnp.int32(time),
        winner=jnp.int32(winner),
        pool_idx=jnp.int32(0),
    )
    return TrueCompetitionState(engine_state=engine_state, real_h=h, real_w=w, time=time)


def move_validity(
    state: TrueCompetitionState, player: int, src: tuple[int, int], dst: tuple[int, int], split: int
) -> tuple[bool, list[str]]:
    """Mirror of pinned game.py _execute_move validity for one seat."""
    eng = state.engine_state
    reasons: list[str] = []
    sr, sc = src
    if not (0 <= sr < state.real_h and 0 <= sc < state.real_w):
        reasons.append("src_out_of_bounds")
        return False, reasons
    dr, dc = dst
    dest_oob = not (0 <= dr < state.real_h and 0 <= dc < state.real_w)
    if dest_oob:
        reasons.append("dest_out_of_bounds")
    owns = bool(eng.ownership[player, sr, sc])
    if not owns:
        reasons.append("src_not_owned")
    army = int(eng.armies[sr, sc])
    amount = army // 2 if split == 1 else army - 1
    amount = max(0, min(amount, army - 1))
    if amount <= 0:
        reasons.append("insufficient_source_army")
    # engine AND: out-of-bounds dests are padding mountains, but the audit
    # must say OUT_OF_BOUNDS first - never "mountain" for training padding
    if not dest_oob and bool(eng.mountains[dr, dc]):
        reasons.append("dest_mountain")
    return not reasons, reasons


def build_cost(state: TrueCompetitionState, player: int, cell: tuple[int, int]) -> int:
    """Dynamic castle price mirroring build_castles.build_cost_grid."""
    eng = state.engine_state
    r, c = cell
    cost = BASE_COST
    structures = (np.asarray(eng.castles) | np.asarray(eng.generals)) & np.asarray(
        eng.ownership[player]
    )
    for sr in range(state.real_h):
        for sc in range(state.real_w):
            if structures[sr, sc]:
                surcharge = PROXIMITY_PENALTY - PROXIMITY_DECAY * (abs(sr - r) + abs(sc - c))
                if surcharge > 0:
                    cost += surcharge
    return cost


def build_validity(
    state: TrueCompetitionState, player: int, cell: tuple[int, int]
) -> tuple[bool, list[str]]:
    """Mirror of pinned build_castles._apply_one validity for one seat."""
    eng = state.engine_state
    reasons: list[str] = []
    r, c = cell
    if not (0 <= r < state.real_h and 0 <= c < state.real_w):
        return False, ["out_of_bounds"]
    if not bool(eng.ownership[player, r, c]):
        reasons.append("not_owned")
    if bool(eng.generals[r, c]) or bool(eng.castles[r, c]):
        reasons.append("not_plain")
    if int(eng.armies[r, c]) < build_cost(state, player, cell):
        reasons.append("insufficient_castle_price")
    if int(eng.winner) >= 0:
        reasons.append("game_over")
    return not reasons, reasons


def direction_from(src: tuple[int, int], dst: tuple[int, int]) -> int | None:
    delta = (dst[0] - src[0], dst[1] - src[1])
    for idx in range(4):
        if (int(DIRECTIONS[idx, 0]), int(DIRECTIONS[idx, 1])) == delta:
            return idx
    return None


def classify_move(
    state: TrueCompetitionState,
    player: int,
    src: tuple[int, int],
    dst: tuple[int, int],
    split: int = 0,
) -> ActionClassification:
    direction = direction_from(src, dst)
    if direction is None:
        return ActionClassification(
            protocol_status=PROTOCOL_UNCLASSIFIABLE,
            protocol_action=None,
            kind="MOVE",
            engine_outcome=ENGINE_SILENT_PASS,
            rejection_reasons=["no_protocol_encoding"],
            split=None,
        )
    ok, reasons = move_validity(state, player, src, dst, split)
    return ActionClassification(
        protocol_status=PROTOCOL_VALID,
        protocol_action=(0, src[0], src[1], direction, split),
        kind="MOVE",
        engine_outcome=ENGINE_EXECUTED if ok else ENGINE_SILENT_PASS,
        rejection_reasons=reasons,
        split=split,
    )


def classify_build(
    state: TrueCompetitionState, player: int, cell: tuple[int, int]
) -> ActionClassification:
    ok, reasons = build_validity(state, player, cell)
    return ActionClassification(
        protocol_status=PROTOCOL_VALID,
        protocol_action=(2, cell[0], cell[1], 0, 0),
        kind="BUILD",
        engine_outcome=ENGINE_EXECUTED if ok else ENGINE_SILENT_PASS,
        rejection_reasons=reasons,
    )


def classify_pass() -> ActionClassification:
    return ActionClassification(
        protocol_status=PROTOCOL_VALID,
        protocol_action=(1, 0, 0, 0, 0),
        kind="PASS",
        engine_outcome=ENGINE_EXECUTED,
        rejection_reasons=[],
    )
