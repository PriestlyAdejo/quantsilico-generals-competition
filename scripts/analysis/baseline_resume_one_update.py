"""Stage 1 resume proof: one PPO update from the locked baseline state.

Loads params/EMA/opt_state from ckpt_final_u482_t7593984 through this
branch's loaders, collects one deterministic self-play batch, applies exactly
one full-batch PPO update using the RESTORED optimizer state, and updates EMA.
Demonstrates checkpoint/resume viability (EXECUTION_PLAN §6.5) on CPU.

Checks: all metrics finite, parameters actually changed, restored opt_state
compatible with one optimizer step. Records pre/post semantic fingerprints.
This is a resume proof, not strength evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from train.competition_native_jax.ema_jax import ema_update  # noqa: E402
from train.competition_native_jax.gae_jax import gae_advantages_batch_jit  # noqa: E402
from train.competition_native_jax.ppo_jax import make_optimizer, ppo_update  # noqa: E402
from train.competition_native_jax.rollout_selfplay_jax import collect_selfplay_batch  # noqa: E402
from train.competition_native_jax.train_jax import load_tree, save_tree  # noqa: E402

DEFAULT_CHECKPOINT = (
    Path.home()
    / "quantsilico-runtime"
    / "cloud_assisted_deadline_salvage_v1_final"
    / "ckpt_final_u482_t7593984"
)
LR = 3e-4


def digest(tree) -> str:
    digest_obj = hashlib.sha256()
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    for key_path, leaf in sorted(flat, key=lambda item: str(item[0])):
        arr = np.ascontiguousarray(np.asarray(leaf))
        digest_obj.update(str(key_path).encode("utf-8"))
        digest_obj.update(arr.tobytes(order="C"))
    return digest_obj.hexdigest()


def finite(tree) -> bool:
    return all(bool(np.isfinite(np.asarray(leaf)).all()) for leaf in jax.tree_util.tree_leaves(tree))


def flatten_batch(batch: dict) -> dict:
    t_steps, n_envs = batch["rewards"].shape
    flat = {
        "spatial": batch["spatial"].reshape(t_steps * n_envs, *batch["spatial"].shape[2:]),
        "global": batch["global"].reshape(t_steps * n_envs, *batch["global"].shape[2:]),
        "mask": batch["mask"].reshape(t_steps * n_envs, -1),
        "actions": batch["actions"].reshape(t_steps * n_envs),
        "old_logp": batch["old_logp"].reshape(t_steps * n_envs),
    }
    values = jnp.concatenate([batch["values"], batch["bootstrap_values"][None, :]], axis=0)
    advantages, returns = gae_advantages_batch_jit(batch["rewards"], values, batch["dones"])
    flat["advantages"] = advantages.reshape(t_steps * n_envs)
    flat["returns"] = returns.reshape(t_steps * n_envs)
    return flat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=Path("var/marathon_takeover/resume_step1"))
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pool-size", type=int, default=64)
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    params_like = init_params(jax.random.PRNGKey(0))
    optimizer = make_optimizer(LR)
    opt_like = optimizer.init(params_like)

    params = load_tree(args.checkpoint / "raw.npz", params_like)
    ema = load_tree(args.checkpoint / "ema.npz", params_like)
    opt_state = load_tree(args.checkpoint / "opt_state.npz", opt_like)
    pre = {
        "params": digest(params),
        "ema": digest(ema),
        "opt_state": digest(opt_state),
    }

    batch = collect_selfplay_batch(
        params,
        num_envs=args.num_envs,
        rollout_len=args.rollout_len,
        seed=args.seed,
        reset_pool_size=args.pool_size,
    )
    flat = flatten_batch(batch)
    new_params, new_opt_state, metrics = ppo_update(params, opt_state, optimizer, flat)
    new_ema = ema_update(ema, new_params)

    metrics_float = {key: float(value) for key, value in metrics.items()}
    checks = {
        "metrics_finite": all(math.isfinite(value) for value in metrics_float.values()),
        "params_finite_after": finite(new_params),
        "opt_state_finite_after": finite(new_opt_state),
        "params_changed": digest(new_params) != pre["params"],
        "ema_changed": digest(new_ema) != pre["ema"],
        "opt_state_changed": digest(new_opt_state) != pre["opt_state"],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    save_tree(args.out / "raw.npz", new_params)
    save_tree(args.out / "ema.npz", new_ema)
    save_tree(args.out / "opt_state.npz", new_opt_state)

    report = {
        "schema_version": 1,
        "kind": "MARATHON_BASELINE_RESUME_ONE_UPDATE",
        "checkpoint": str(args.checkpoint),
        "config": {
            "num_envs": args.num_envs,
            "rollout_len": args.rollout_len,
            "seed": args.seed,
            "reset_pool_size": args.pool_size,
            "lr": LR,
            "device": str(jax.devices()),
        },
        "pre_digests": pre,
        "post_digests": {
            "params": digest(new_params),
            "ema": digest(new_ema),
            "opt_state": digest(new_opt_state),
        },
        "metrics": metrics_float,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    (args.out / "resume_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
