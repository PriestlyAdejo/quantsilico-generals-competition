"""Bounded Stage 1 structural probe: baseline checkpoint vs current CNJ templates.

Compares the npz key structure (names, shapes, dtypes) of the locked baseline
checkpoint artefacts against pytrees built from THIS repository's
competition_native_jax lineage (init_params / optimizer). This proves whether
the marathon branch's own code can structurally host the baseline weights,
independently of the unresolved writer-source identity (EV-0012).

Classification per artefact:
- EXACT_MATCH: keys, shapes, and dtypes all agree.
- KEY_MATCH_SHAPE_DIFF: same keys but some shapes/dtypes differ.
- KEY_MISMATCH: key sets differ (missing/extra reported).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from train.competition_native_jax.ppo_jax import make_optimizer  # noqa: E402

DEFAULT_CHECKPOINT = (
    Path.home()
    / "quantsilico-runtime"
    / "cloud_assisted_deadline_salvage_v1_final"
    / "ckpt_final_u482_t7593984"
)
LR = 3e-4


def template_signature(tree) -> dict[str, tuple[tuple[int, ...], str]]:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {
        str(key_path): (tuple(np.asarray(leaf).shape), str(np.asarray(leaf).dtype))
        for key_path, leaf in flat
    }


def npz_signature(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    data = np.load(path, allow_pickle=False)
    return {key: (tuple(data[key].shape), str(data[key].dtype)) for key in data.keys()}


def compare(template: dict, stored: dict) -> dict[str, object]:
    template_keys = set(template)
    stored_keys = set(stored)
    if template_keys != stored_keys:
        return {
            "status": "KEY_MISMATCH",
            "missing_from_checkpoint": sorted(template_keys - stored_keys)[:20],
            "extra_in_checkpoint": sorted(stored_keys - template_keys)[:20],
            "template_keys": len(template_keys),
            "stored_keys": len(stored_keys),
        }
    differences = {
        key: {"template": list(template[key]), "stored": list(stored[key])}
        for key in template_keys
        if template[key] != stored[key]
    }
    if differences:
        return {"status": "KEY_MATCH_SHAPE_DIFF", "differences": differences}
    return {"status": "EXACT_MATCH", "keys": len(template_keys)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    ckpt = args.checkpoint
    if not ckpt.is_dir():
        print(f"checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    key = jax.random.PRNGKey(0)
    params_like = init_params(key)
    optimizer = make_optimizer(LR)
    opt_like = optimizer.init(params_like)

    report = {
        "checkpoint": str(ckpt),
        "learner_template": "competition_native_jax@marathon-branch",
        "artefacts": {
            "raw.npz": compare(template_signature(params_like), npz_signature(ckpt / "raw.npz")),
            "ema.npz": compare(template_signature(params_like), npz_signature(ckpt / "ema.npz")),
            "frozen_opponent.npz": compare(
                template_signature(params_like), npz_signature(ckpt / "frozen_opponent.npz")
            ),
            "opt_state.npz": compare(
                template_signature(opt_like), npz_signature(ckpt / "opt_state.npz")
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
