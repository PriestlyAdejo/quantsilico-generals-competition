"""Pure-JAX transformer policy/value (canonical training model)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from generals_bot.competition_native_jax.constants import (
    ACTION_DIM,
    EMB_DIM,
    HL_GAUSS_BINS,
    MAX_HW,
    N_HEADS,
    N_LAYERS,
    NUM_PATCHES,
    PASS_INDEX,
    PATCH,
    PATCH_GRID,
)
from generals_bot.competition_native_jax.obs_memory import N_GLOBAL, N_SPATIAL


def _glorot(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    fan_in, fan_out = shape[0], shape[-1]
    lim = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


def init_params(
    key: jax.Array,
    spatial_planes: int = N_SPATIAL,
    global_dim: int = N_GLOBAL,
) -> dict[str, Any]:
    """spatial_planes: canonical 8; STAGE5 T2 (k1 history) uses 16; OBS-V2 14.

    global_dim: canonical 8; OBS-V2 uses 12.
    """
    keys = jax.random.split(key, 8 + 4 * N_LAYERS)
    params: dict[str, Any] = {
        "patch_proj": _glorot(keys[0], (spatial_planes * PATCH * PATCH, EMB_DIM)),
        "cls": _glorot(keys[1], (EMB_DIM,)),
        "pos": _glorot(keys[2], (NUM_PATCHES + 1, EMB_DIM)),
        "global_proj": _glorot(keys[3], (global_dim, EMB_DIM)),
        "move_head": _glorot(keys[4], (EMB_DIM, PATCH * PATCH * 8)),
        "build_head": _glorot(keys[5], (EMB_DIM, PATCH * PATCH)),
        "pass_head": _glorot(keys[6], (EMB_DIM,)),
        "value_head": _glorot(keys[7], (EMB_DIM, HL_GAUSS_BINS)),
        "layers": [],
    }
    ki = 8
    for _ in range(N_LAYERS):
        params["layers"].append(
            {
                "attn_w": _glorot(keys[ki], (EMB_DIM, 3 * EMB_DIM)),
                "attn_out": _glorot(keys[ki + 1], (EMB_DIM, EMB_DIM)),
                "ff_w1": _glorot(keys[ki + 2], (EMB_DIM, 4 * EMB_DIM)),
                "ff_w2": _glorot(keys[ki + 3], (4 * EMB_DIM, EMB_DIM)),
            }
        )
        ki += 4
    return params


def _layernorm(x: jax.Array, eps: float = 1e-5) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def _mha(x: jax.Array, attn_w: jax.Array, attn_out: jax.Array) -> jax.Array:
    t, d = x.shape
    qkv = x @ attn_w
    q, k, v = jnp.split(qkv, 3, axis=-1)
    head_dim = d // N_HEADS
    qh = q.reshape(t, N_HEADS, head_dim).transpose(1, 0, 2)
    kh = k.reshape(t, N_HEADS, head_dim).transpose(1, 0, 2)
    vh = v.reshape(t, N_HEADS, head_dim).transpose(1, 0, 2)
    att = jax.nn.softmax((qh @ kh.transpose(0, 2, 1)) / jnp.sqrt(head_dim), axis=-1)
    out = (att @ vh).transpose(1, 0, 2).reshape(t, d)
    return out @ attn_out


def unpatchify_move(patch_move: jax.Array) -> jax.Array:
    """[49,3,3,8] -> [21,21,8] via reshape (exact patch-grid layout)."""
    x = patch_move.reshape(PATCH_GRID, PATCH_GRID, PATCH, PATCH, 8)
    return x.transpose(0, 2, 1, 3, 4).reshape(MAX_HW, MAX_HW, 8)


def unpatchify_build(patch_build: jax.Array) -> jax.Array:
    """[49,3,3] -> [21,21] via reshape (exact patch-grid layout)."""
    x = patch_build.reshape(PATCH_GRID, PATCH_GRID, PATCH, PATCH)
    return x.transpose(0, 2, 1, 3).reshape(MAX_HW, MAX_HW)


def pack_flat_logits(move: jax.Array, build: jax.Array, pass_logit: jax.Array) -> jax.Array:
    """Pack to ACTION_DIM using PASS + 9*cell layout (vectorised scatter)."""
    logits = jnp.full((ACTION_DIM,), -1e9, dtype=jnp.float32)
    logits = logits.at[PASS_INDEX].set(pass_logit.astype(jnp.float32))
    flat_move = move.reshape(MAX_HW * MAX_HW, 8)
    flat_build = build.reshape(MAX_HW * MAX_HW)
    cells = jnp.arange(MAX_HW * MAX_HW)
    base = 1 + cells * 9
    move_idx = (base[:, None] + jnp.arange(8)[None, :]).reshape(-1)
    logits = logits.at[move_idx].set(flat_move.reshape(-1))
    logits = logits.at[base + 8].set(flat_build)
    return logits


def _extract_patch_tokens(spatial: jax.Array, patch_proj: jax.Array) -> jax.Array:
    """spatial [C,21,21] -> tokens [NUM_PATCHES, EMB_DIM] without Python patch loop."""
    c = spatial.shape[0]
    patches = spatial.reshape(c, PATCH_GRID, PATCH, PATCH_GRID, PATCH)
    patches = patches.transpose(1, 3, 0, 2, 4).reshape(NUM_PATCHES, c * PATCH * PATCH)
    return patches @ patch_proj


def forward(params: dict, spatial: jax.Array, global_vec: jax.Array) -> dict[str, jax.Array]:
    """spatial [C,21,21], global [G] -> flat_logits + value_logits."""
    tokens_arr = _extract_patch_tokens(spatial, params["patch_proj"])
    cls = params["cls"] + (global_vec @ params["global_proj"])
    x = jnp.concatenate([cls[None, :], tokens_arr], axis=0) + params["pos"]
    for layer in params["layers"]:
        x = _layernorm(x + _mha(x, layer["attn_w"], layer["attn_out"]))
        h = jax.nn.relu(x @ layer["ff_w1"])
        x = _layernorm(x + h @ layer["ff_w2"])
    cls_out = x[0]
    patch_out = x[1:]
    patch_move = (patch_out @ params["move_head"]).reshape(NUM_PATCHES, PATCH, PATCH, 8)
    patch_build = (patch_out @ params["build_head"]).reshape(NUM_PATCHES, PATCH, PATCH)
    move = unpatchify_move(patch_move)
    build = unpatchify_build(patch_build)
    pass_logit = cls_out @ params["pass_head"]
    flat = pack_flat_logits(move, build, pass_logit)
    value_logits = cls_out @ params["value_head"]
    return {"flat_logits": flat, "value_logits": value_logits, "move": move, "build": build}


forward_jit = jax.jit(forward)


def forward_batch(params: dict, spatial: jax.Array, global_vec: jax.Array) -> dict[str, jax.Array]:
    """Native compiled batched forward.

    Inputs:
      spatial: [B, C, 21, 21]
      global_vec: [B, G]
    Outputs: flat_logits [B, 3970], value_logits [B, bins], ...

    Implemented via vmap over the single-sample kernel; callers must not
    invoke ``forward`` once per environment from Python.
    """
    return _forward_batch(params, spatial, global_vec)


_forward_batch = jax.jit(jax.vmap(forward, in_axes=(None, 0, 0)))
