"""NumPy transformer policy/value for training prototypes and CPU deployment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from generals_bot.competition_native_jax.constants import (
    EMB_DIM,
    HL_GAUSS_BINS,
    MAX_HW,
    N_HEADS,
    N_LAYERS,
    NUM_PATCHES,
    PATCH,
    PATCH_GRID,
)
from generals_bot.competition_native_jax.obs_memory import N_GLOBAL, N_SPATIAL
from generals_bot.competition_native_jax.patchify import (
    pack_flat_logits,
    unpatchify_build_logits,
    unpatchify_move_logits,
)


def _xavier(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    fan_in = shape[0] if len(shape) >= 2 else shape[-1]
    fan_out = shape[-1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


@dataclass
class TransformerWeights:
    patch_proj: np.ndarray  # [C*9, D]
    cls: np.ndarray  # [D]
    pos: np.ndarray  # [50, D] 49+cls
    global_proj: np.ndarray  # [G, D]
    attn_w: list[np.ndarray]  # each [D, 3D]
    attn_out: list[np.ndarray]  # [D, D]
    ff_w1: list[np.ndarray]
    ff_w2: list[np.ndarray]
    move_head: np.ndarray  # [D, 9*8] wait: per patch 3*3*8 = 72
    build_head: np.ndarray  # [D, 9]
    pass_head: np.ndarray  # [D]
    value_head: np.ndarray  # [D, bins]


def init_weights(seed: int = 0) -> TransformerWeights:
    rng = np.random.default_rng(seed)
    d = EMB_DIM
    w = TransformerWeights(
        patch_proj=_xavier((N_SPATIAL * PATCH * PATCH, d), rng),
        cls=_xavier((d,), rng),
        pos=_xavier((NUM_PATCHES + 1, d), rng),
        global_proj=_xavier((N_GLOBAL, d), rng),
        attn_w=[],
        attn_out=[],
        ff_w1=[],
        ff_w2=[],
        move_head=_xavier((d, PATCH * PATCH * 8), rng),
        build_head=_xavier((d, PATCH * PATCH), rng),
        pass_head=_xavier((d,), rng),
        value_head=_xavier((d, HL_GAUSS_BINS), rng),
    )
    for _ in range(N_LAYERS):
        w.attn_w.append(_xavier((d, 3 * d), rng))
        w.attn_out.append(_xavier((d, d), rng))
        w.ff_w1.append(_xavier((d, 4 * d), rng))
        w.ff_w2.append(_xavier((4 * d, d), rng))
    return w


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def _mha(x: np.ndarray, w_qkv: np.ndarray, w_out: np.ndarray) -> np.ndarray:
    # x: [T, D]
    t, d = x.shape
    qkv = x @ w_qkv
    q, k, v = np.split(qkv, 3, axis=-1)
    head_dim = d // N_HEADS
    def split_heads(tns: np.ndarray) -> np.ndarray:
        return tns.reshape(t, N_HEADS, head_dim).transpose(1, 0, 2)

    qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
    att = _softmax((qh @ kh.transpose(0, 2, 1)) / np.sqrt(head_dim), axis=-1)
    out = (att @ vh).transpose(1, 0, 2).reshape(t, d)
    return out @ w_out


def forward(
    spatial: np.ndarray,
    global_vec: np.ndarray,
    weights: TransformerWeights,
) -> dict[str, np.ndarray]:
    """spatial [C,21,21], global [G] -> logits + value bins.

    Embedding width and depth are taken from ``weights`` so student shapes can be
    benchmarked without changing locked teacher defaults in constants.py.
    """
    emb = int(weights.patch_proj.shape[1])
    n_layers = len(weights.attn_w)
    # Patchify
    tokens = np.zeros((NUM_PATCHES, emb), dtype=np.float32)
    for p in range(NUM_PATCHES):
        pr, pc = divmod(p, PATCH_GRID)
        r0, c0 = pr * PATCH, pc * PATCH
        patch = spatial[:, r0 : r0 + PATCH, c0 : c0 + PATCH].reshape(-1)
        tokens[p] = patch @ weights.patch_proj
    cls = weights.cls + (global_vec @ weights.global_proj)
    x = np.concatenate([cls[None, :], tokens], axis=0) + weights.pos
    for i in range(n_layers):
        x = _layernorm(x + _mha(x, weights.attn_w[i], weights.attn_out[i]))
        h = np.maximum(0, x @ weights.ff_w1[i])
        x = _layernorm(x + h @ weights.ff_w2[i])
    cls_out = x[0]
    patch_out = x[1:]
    patch_move = (patch_out @ weights.move_head).reshape(NUM_PATCHES, PATCH, PATCH, 8)
    patch_build = (patch_out @ weights.build_head).reshape(NUM_PATCHES, PATCH, PATCH)
    move = unpatchify_move_logits(patch_move)
    build = unpatchify_build_logits(patch_build)
    pass_logit = float(cls_out @ weights.pass_head)
    flat = pack_flat_logits(move, build, pass_logit)
    value_logits = cls_out @ weights.value_head
    return {
        "flat_logits": flat,
        "move_logits": move,
        "build_logits": build,
        "pass_logit": np.asarray([pass_logit]),
        "value_logits": value_logits,
    }


def weights_to_dict(w: TransformerWeights) -> dict[str, np.ndarray]:
    d = {
        "patch_proj": w.patch_proj,
        "cls": w.cls,
        "pos": w.pos,
        "global_proj": w.global_proj,
        "move_head": w.move_head,
        "build_head": w.build_head,
        "pass_head": w.pass_head,
        "value_head": w.value_head,
    }
    for i in range(len(w.attn_w)):
        d[f"attn_w_{i}"] = w.attn_w[i]
        d[f"attn_out_{i}"] = w.attn_out[i]
        d[f"ff_w1_{i}"] = w.ff_w1[i]
        d[f"ff_w2_{i}"] = w.ff_w2[i]
    return d


def weights_from_dict(d: dict[str, np.ndarray]) -> TransformerWeights:
    layer_ids = sorted(
        int(k.split("_")[-1]) for k in d if k.startswith("attn_w_")
    )
    w = TransformerWeights(
        patch_proj=d["patch_proj"],
        cls=d["cls"],
        pos=d["pos"],
        global_proj=d["global_proj"],
        attn_w=[d[f"attn_w_{i}"] for i in layer_ids],
        attn_out=[d[f"attn_out_{i}"] for i in layer_ids],
        ff_w1=[d[f"ff_w1_{i}"] for i in layer_ids],
        ff_w2=[d[f"ff_w2_{i}"] for i in layer_ids],
        move_head=d["move_head"],
        build_head=d["build_head"],
        pass_head=d["pass_head"],
        value_head=d["value_head"],
    )
    return w
