"""Student JAX transformer (emb96/d2/h4) — NOT in learner hash (separate from transformer_jax)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from generals_bot.competition_native_jax.constants import (
    ACTION_DIM,
    HL_GAUSS_BINS,
    MAX_HW,
    NUM_PATCHES,
    PASS_INDEX,
    PATCH,
    PATCH_GRID,
)
from generals_bot.competition_native_jax.obs_memory import N_GLOBAL, N_SPATIAL

STUDENT_EMB = 96
STUDENT_LAYERS = 2
STUDENT_HEADS = 4


def _glorot(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    fan_in, fan_out = shape[0], shape[-1]
    lim = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


def init_student_params(
    key: jax.Array,
    *,
    emb: int = STUDENT_EMB,
    layers: int = STUDENT_LAYERS,
    heads: int = STUDENT_HEADS,
) -> dict[str, Any]:
    if emb % heads != 0:
        raise ValueError(f"emb {emb} not divisible by heads {heads}")
    keys = jax.random.split(key, 8 + 4 * layers)
    # Keep integer meta under "meta" so optax/grad never sees int leaves.
    params: dict[str, Any] = {
        "meta": {"emb": emb, "layers_n": layers, "heads": heads},
        "patch_proj": _glorot(keys[0], (N_SPATIAL * PATCH * PATCH, emb)),
        "cls": _glorot(keys[1], (emb,)),
        "pos": _glorot(keys[2], (NUM_PATCHES + 1, emb)),
        "global_proj": _glorot(keys[3], (N_GLOBAL, emb)),
        "move_head": _glorot(keys[4], (emb, PATCH * PATCH * 8)),
        "build_head": _glorot(keys[5], (emb, PATCH * PATCH)),
        "pass_head": _glorot(keys[6], (emb,)),
        "value_head": _glorot(keys[7], (emb, HL_GAUSS_BINS)),
        "layers": [],
    }
    ki = 8
    for _ in range(layers):
        params["layers"].append(
            {
                "attn_w": _glorot(keys[ki], (emb, 3 * emb)),
                "attn_out": _glorot(keys[ki + 1], (emb, emb)),
                "ff_w1": _glorot(keys[ki + 2], (emb, 4 * emb)),
                "ff_w2": _glorot(keys[ki + 3], (4 * emb, emb)),
            }
        )
        ki += 4
    return params


def _layernorm(x: jax.Array, eps: float = 1e-5) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def _mha(x: jax.Array, attn_w: jax.Array, attn_out: jax.Array, n_heads: int) -> jax.Array:
    t, d = x.shape
    qkv = x @ attn_w
    q, k, v = jnp.split(qkv, 3, axis=-1)
    head_dim = d // n_heads
    qh = q.reshape(t, n_heads, head_dim).transpose(1, 0, 2)
    kh = k.reshape(t, n_heads, head_dim).transpose(1, 0, 2)
    vh = v.reshape(t, n_heads, head_dim).transpose(1, 0, 2)
    att = jax.nn.softmax((qh @ kh.transpose(0, 2, 1)) / jnp.sqrt(head_dim), axis=-1)
    out = (att @ vh).transpose(1, 0, 2).reshape(t, d)
    return out @ attn_out


def _unpatchify_move(patch_move: jax.Array) -> jax.Array:
    x = patch_move.reshape(PATCH_GRID, PATCH_GRID, PATCH, PATCH, 8)
    return x.transpose(0, 2, 1, 3, 4).reshape(MAX_HW, MAX_HW, 8)


def _unpatchify_build(patch_build: jax.Array) -> jax.Array:
    x = patch_build.reshape(PATCH_GRID, PATCH_GRID, PATCH, PATCH)
    return x.transpose(0, 2, 1, 3).reshape(MAX_HW, MAX_HW)


def _pack_flat_logits(move: jax.Array, build: jax.Array, pass_logit: jax.Array) -> jax.Array:
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
    c = spatial.shape[0]
    patches = spatial.reshape(c, PATCH_GRID, PATCH, PATCH_GRID, PATCH)
    patches = patches.transpose(1, 3, 0, 2, 4).reshape(NUM_PATCHES, c * PATCH * PATCH)
    return patches @ patch_proj


def forward_student(
    params: dict,
    spatial: jax.Array,
    global_vec: jax.Array,
    *,
    n_heads: int = STUDENT_HEADS,
) -> dict[str, jax.Array]:
    # n_heads is a Python int (static); do not read from params under jit.
    tokens_arr = _extract_patch_tokens(spatial, params["patch_proj"])
    cls = params["cls"] + (global_vec @ params["global_proj"])
    x = jnp.concatenate([cls[None, :], tokens_arr], axis=0) + params["pos"]
    for layer in params["layers"]:
        x = _layernorm(x + _mha(x, layer["attn_w"], layer["attn_out"], n_heads))
        h = jax.nn.relu(x @ layer["ff_w1"])
        x = _layernorm(x + h @ layer["ff_w2"])
    cls_out = x[0]
    patch_out = x[1:]
    patch_move = (patch_out @ params["move_head"]).reshape(NUM_PATCHES, PATCH, PATCH, 8)
    patch_build = (patch_out @ params["build_head"]).reshape(NUM_PATCHES, PATCH, PATCH)
    move = _unpatchify_move(patch_move)
    build = _unpatchify_build(patch_build)
    pass_logit = cls_out @ params["pass_head"]
    flat = _pack_flat_logits(move, build, pass_logit)
    value_logits = cls_out @ params["value_head"]
    return {"flat_logits": flat, "value_logits": value_logits}


def forward_student_batch(params: dict, spatial_b: jax.Array, global_b: jax.Array) -> dict[str, jax.Array]:
    # Strip non-array meta before vmap/jit so int leaves never enter the transform.
    train = {k: v for k, v in params.items() if k != "meta"}
    return _forward_student_batch_jit(train, spatial_b, global_b)


_forward_student_batch_jit = jax.jit(jax.vmap(forward_student, in_axes=(None, 0, 0)))


def student_params_to_numpy_weights(params: dict) -> "TransformerWeights":
    """Export JAX student params into deployable NumPy TransformerWeights."""
    from generals_bot.competition_native_jax.transformer import TransformerWeights
    import numpy as np

    layers = params["layers"]
    return TransformerWeights(
        patch_proj=np.asarray(params["patch_proj"], dtype=np.float32),
        cls=np.asarray(params["cls"], dtype=np.float32),
        pos=np.asarray(params["pos"], dtype=np.float32),
        global_proj=np.asarray(params["global_proj"], dtype=np.float32),
        attn_w=[np.asarray(L["attn_w"], dtype=np.float32) for L in layers],
        attn_out=[np.asarray(L["attn_out"], dtype=np.float32) for L in layers],
        ff_w1=[np.asarray(L["ff_w1"], dtype=np.float32) for L in layers],
        ff_w2=[np.asarray(L["ff_w2"], dtype=np.float32) for L in layers],
        move_head=np.asarray(params["move_head"], dtype=np.float32),
        build_head=np.asarray(params["build_head"], dtype=np.float32),
        pass_head=np.asarray(params["pass_head"], dtype=np.float32),
        value_head=np.asarray(params["value_head"], dtype=np.float32),
    )
