"""Device-resident static action / board geometry for competition JAX."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from generals_bot.competition_native_jax.constants import ACTION_DIM, MAX_HW, PASS_INDEX


class StaticGeometry(NamedTuple):
    """Immutable lookup tensors; build once, reuse across rollouts."""

    action_source_cell: jnp.ndarray  # [ACTION_DIM] int32
    action_dest_cell: jnp.ndarray  # [ACTION_DIM] int32
    action_mode: jnp.ndarray  # [ACTION_DIM] int32  0=pass,1=move,2=build
    action_direction: jnp.ndarray  # [ACTION_DIM] int32
    action_split: jnp.ndarray  # [ACTION_DIM] int32
    action_build_cell: jnp.ndarray  # [ACTION_DIM] int32
    row_of_cell: jnp.ndarray  # [MAX_HW*MAX_HW]
    col_of_cell: jnp.ndarray
    neighbour_cell: jnp.ndarray  # [MAX_HW*MAX_HW, 4] dest cell or -1
    manhattan: jnp.ndarray  # [MAX_HW*MAX_HW, MAX_HW*MAX_HW] int16
    playable: jnp.ndarray  # [4, MAX_HW, MAX_HW] bool for sizes 18..21
    pass_index: int
    build_local: int


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # N S W E matching engine


def build_static_geometry() -> StaticGeometry:
    n = MAX_HW * MAX_HW
    cells = np.arange(n, dtype=np.int32)
    rows = cells // MAX_HW
    cols = cells % MAX_HW

    neigh = np.full((n, 4), -1, dtype=np.int32)
    for d, (dr, dc) in enumerate(_DIRS):
        nr = rows + dr
        nc = cols + dc
        valid = (nr >= 0) & (nr < MAX_HW) & (nc >= 0) & (nc < MAX_HW)
        dest = nr * MAX_HW + nc
        neigh[:, d] = np.where(valid, dest, -1)

    r0 = rows[:, None]
    c0 = cols[:, None]
    manhattan = (np.abs(r0 - rows[None, :]) + np.abs(c0 - cols[None, :])).astype(np.int16)

    playable = np.zeros((4, MAX_HW, MAX_HW), dtype=bool)
    for i, s in enumerate(range(18, 22)):
        playable[i, :s, :s] = True

    src = np.zeros((ACTION_DIM,), dtype=np.int32)
    dest = np.zeros((ACTION_DIM,), dtype=np.int32)
    mode = np.zeros((ACTION_DIM,), dtype=np.int32)
    direction = np.zeros((ACTION_DIM,), dtype=np.int32)
    split = np.zeros((ACTION_DIM,), dtype=np.int32)
    build_cell = np.zeros((ACTION_DIM,), dtype=np.int32)

    mode[PASS_INDEX] = 0

    # Vectorised non-PASS actions: idx = 1 + cell*9 + local, local in 0..8
    cell_grid = np.repeat(cells, 9)
    local = np.tile(np.arange(9, dtype=np.int32), n)
    idxs = 1 + cell_grid * 9 + local
    is_build = local == 8
    d = local // 2
    sp = local % 2
    d = np.where(is_build, 0, d)
    sp = np.where(is_build, 0, sp)

    src[idxs] = cell_grid
    mode[idxs] = np.where(is_build, 2, 1)
    direction[idxs] = d
    split[idxs] = sp
    build_cell[idxs] = np.where(is_build, cell_grid, 0)
    nd = neigh[cell_grid, np.clip(d, 0, 3)]
    dest[idxs] = np.where(is_build, cell_grid, np.where(nd >= 0, nd, cell_grid))

    return StaticGeometry(
        action_source_cell=jnp.asarray(src),
        action_dest_cell=jnp.asarray(dest),
        action_mode=jnp.asarray(mode),
        action_direction=jnp.asarray(direction),
        action_split=jnp.asarray(split),
        action_build_cell=jnp.asarray(build_cell),
        row_of_cell=jnp.asarray(rows),
        col_of_cell=jnp.asarray(cols),
        neighbour_cell=jnp.asarray(neigh),
        manhattan=jnp.asarray(manhattan),
        playable=jnp.asarray(playable),
        pass_index=PASS_INDEX,
        build_local=8,
    )


STATIC_GEOMETRY = None  # lazy; avoid import-time device arrays closed over by jit


def get_static_geometry() -> StaticGeometry:
    global STATIC_GEOMETRY
    if STATIC_GEOMETRY is None:
        STATIC_GEOMETRY = build_static_geometry()
    return STATIC_GEOMETRY


def index_to_engine_action_static(idx: jnp.ndarray, geo: StaticGeometry | None = None) -> jnp.ndarray:
    """Scalar action index -> [5] using static geometry (exact codec parity). Host/tests only."""
    if geo is None:
        geo = get_static_geometry()
    is_pass = idx == geo.pass_index
    cell = geo.action_source_cell[idx]
    row = geo.row_of_cell[cell]
    col = geo.col_of_cell[cell]
    is_build = geo.action_mode[idx] == 2
    direction = geo.action_direction[idx]
    split = geo.action_split[idx]
    kind = jnp.where(is_pass, 1, jnp.where(is_build, 2, 0))
    return jnp.stack(
        [
            kind.astype(jnp.int32),
            jnp.where(is_pass, 0, row).astype(jnp.int32),
            jnp.where(is_pass, 0, col).astype(jnp.int32),
            jnp.where(is_pass | is_build, 0, direction).astype(jnp.int32),
            jnp.where(is_pass | is_build, 0, split).astype(jnp.int32),
        ]
    )
