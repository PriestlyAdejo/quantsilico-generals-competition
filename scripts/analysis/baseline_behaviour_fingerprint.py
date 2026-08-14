"""Stage 1 behavioural fingerprint: first-N-observation capsule for the baseline.

Loads the locked baseline checkpoint (raw and EMA) through this branch's
``load_tree`` and collects one small deterministic self-play batch on CPU.
Records the first N observations/actions/legal masks/rewards/dones as both
sampled arrays (npz) and SHA-256 digests plus scalar summaries (JSON), per
EXECUTION_PLAN §6.2 capsule requirements.

This is a fingerprint, not strength evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from train.competition_native_jax.rollout_selfplay_jax import collect_selfplay_batch  # noqa: E402
from train.competition_native_jax.train_jax import load_tree  # noqa: E402

DEFAULT_CHECKPOINT = (
    Path.home()
    / "quantsilico-runtime"
    / "cloud_assisted_deadline_salvage_v1_final"
    / "ckpt_final_u482_t7593984"
)
FIRST_N_STEPS = 2
FIRST_N_ENVS = 2


def digest(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array)).tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=Path("var/marathon_takeover"))
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pool-size", type=int, default=64)
    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    params_like = init_params(jax.random.PRNGKey(0))
    params = load_tree(args.checkpoint / "raw.npz", params_like)
    ema = load_tree(args.checkpoint / "ema.npz", params_like)

    batch = collect_selfplay_batch(
        params,
        num_envs=args.num_envs,
        rollout_len=args.rollout_len,
        seed=args.seed,
        reset_pool_size=args.pool_size,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    sample = {
        key: np.asarray(batch[key])[:FIRST_N_STEPS, :FIRST_N_ENVS]
        for key in (
            "spatial",
            "global",
            "mask",
            "actions",
            "old_logp",
            "values",
            "rewards",
            "dones",
        )
    }
    sample["ema_params_digest"] = np.frombuffer(
        bytes.fromhex(digest(jax.tree_util.tree_leaves(ema)[0])), dtype=np.uint8
    )
    np.savez_compressed(args.out / "baseline_behaviour_sample.npz", **sample)

    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "config": {
            "num_envs": args.num_envs,
            "rollout_len": args.rollout_len,
            "seed": args.seed,
            "reset_pool_size": args.pool_size,
            "device": str(jax.devices()),
        },
        "batch_shapes": {key: list(np.asarray(batch[key]).shape) for key in sample if key in batch},
        "first_n_digests": {key: digest(value) for key, value in sample.items() if key in batch},
        "scalar_summary": {
            "reward_sum": float(np.asarray(batch["rewards"]).sum()),
            "done_count": int(np.asarray(batch["dones"]).sum()),
            "action_hist": {
                str(action): int(count)
                for action, count in zip(
                    *np.unique(np.asarray(batch["actions"]), return_counts=True), strict=True
                )
            },
            "value_mean": float(np.asarray(batch["values"]).mean()),
            "logp_mean": float(np.asarray(batch["old_logp"]).mean()),
        },
    }
    report_path = args.out / "baseline_behaviour_fingerprint.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
