"""Load CNJ ema.npz → TransformerWeights without importing JAX (CPU-safe under GPU PPO)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from generals_bot.competition_native_jax.transformer import TransformerWeights


def load_ema_numpy(ckpt_dir: Path) -> TransformerWeights:
    path = Path(ckpt_dir) / "ema.npz"
    data = np.load(path, allow_pickle=False)
    keys = list(data.files)

    def top(name: str) -> np.ndarray:
        cands = [k for k in keys if f"DictKey(key='{name}')" in k and "layers" not in k]
        if not cands:
            cands = [k for k in keys if f"'{name}'" in k and "layers" not in k]
        if not cands:
            raise KeyError(name)
        cands.sort(key=len)
        return np.asarray(data[cands[0]], dtype=np.float32)

    def layer_arr(i: int, name: str) -> np.ndarray:
        needle = f"SequenceKey(idx={i})"
        cands = [
            k
            for k in keys
            if "layers" in k and needle in k and f"DictKey(key='{name}')" in k
        ]
        if not cands:
            # fallback ordered list
            cands = sorted(k for k in keys if "layers" in k and f"DictKey(key='{name}')" in k)
            if i >= len(cands):
                raise KeyError((i, name))
            return np.asarray(data[cands[i]], dtype=np.float32)
        return np.asarray(data[cands[0]], dtype=np.float32)

    n_layers = len([k for k in keys if "attn_w" in k and "layers" in k])
    if n_layers <= 0:
        raise RuntimeError(f"no layers in {path}")

    return TransformerWeights(
        patch_proj=top("patch_proj"),
        cls=top("cls"),
        pos=top("pos"),
        global_proj=top("global_proj"),
        attn_w=[layer_arr(i, "attn_w") for i in range(n_layers)],
        attn_out=[layer_arr(i, "attn_out") for i in range(n_layers)],
        ff_w1=[layer_arr(i, "ff_w1") for i in range(n_layers)],
        ff_w2=[layer_arr(i, "ff_w2") for i in range(n_layers)],
        move_head=top("move_head"),
        build_head=top("build_head"),
        pass_head=top("pass_head"),
        value_head=top("value_head"),
    )


if __name__ == "__main__":
    import sys

    w = load_ema_numpy(Path(sys.argv[1]))
    print("ok", w.patch_proj.shape, len(w.attn_w), flush=True)
