"""SH-R1 screening runner: resume MARATHON_BASELINE_V0 and train one arm.

PPO_SEMANTICS: UNCHANGED (continues the baseline lineage without altering
action-selection semantics). Resumes raw/opt_state from the hash-verified
checkpoint (EV-0013/0015), trains a fixed transition budget at the arm's
geometry, and emits per-update telemetry plus an arm summary computed from
the PREDECLARED screening metrics in screening_round_1_plan.yaml. Integrity
stops only (round 1): any non-finite metric halts the arm as INTEGRITY_FAILURE.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from generals_bot.competition_native_jax.competition_env_jax import (  # noqa: E402
    build_competition_reset_pool,
)
from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from train.competition_native_jax.ema_jax import ema_update  # noqa: E402
from train.competition_native_jax.gae_jax import gae_advantages_batch_jit  # noqa: E402
from train.competition_native_jax.ppo_jax import make_optimizer, ppo_update  # noqa: E402
from train.competition_native_jax.rollout_selfplay_jax import (  # noqa: E402
    collect_selfplay_batch,
)
from train.competition_native_jax.train_jax import load_tree, save_tree  # noqa: E402

DEFAULT_CHECKPOINT = (
    Path.home()
    / "quantsilico-runtime"
    / "cloud_assisted_deadline_salvage_v1_final"
    / "ckpt_final_u482_t7593984"
)
LR = 3e-4
RESET_POOL_SIZE = 4096


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


def healthy(metrics: dict) -> bool:
    if not all(math.isfinite(float(v)) for v in metrics.values()):
        return False
    return 0.5 <= float(metrics.get("ratio", 0.0)) <= 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--rollout-len", type=int, required=True)
    parser.add_argument("--budget-transitions", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: experiments/marathon/screening_runs/<arm-id>",
    )
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument(
        "--min-generals-distance",
        type=int,
        default=17,
        help="spawn-distance curriculum knob (map generation only; PPO_SEMANTICS UNCHANGED)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or REPO / f"experiments/marathon/screening_runs/{args.arm_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = out_dir / "telemetry.jsonl"
    if telemetry_path.exists():
        print(f"refusing to overwrite existing telemetry: {telemetry_path}", file=sys.stderr)
        return 2

    params_like = init_params(jax.random.PRNGKey(0))
    optimizer = make_optimizer(LR)
    opt_like = optimizer.init(params_like)
    params = load_tree(args.checkpoint / "raw.npz", params_like)
    ema = load_tree(args.checkpoint / "ema.npz", params_like)
    opt_state = load_tree(args.checkpoint / "opt_state.npz", opt_like)

    per_update = args.num_envs * args.rollout_len
    n_updates = args.budget_transitions // per_update
    if args.max_updates is not None:
        n_updates = min(n_updates, args.max_updates)
    if n_updates < 1:
        print("budget smaller than one update", file=sys.stderr)
        return 2

    records = []
    stop_reason = "BUDGET_REACHED"
    collect_wall = 0.0
    update_wall = 0.0
    # Build the reset pool ONCE and reuse it across updates (ladder pattern,
    # EV-0029): reconstructing boards per collect dominated wall-time.
    # Environment initialisation only; PPO_SEMANTICS unchanged.
    pool_started = time.perf_counter()
    reset_pool = build_competition_reset_pool(
        jax.random.PRNGKey(args.seed),
        RESET_POOL_SIZE,
        min_generals_distance=args.min_generals_distance,
    )
    jax.block_until_ready(jax.tree_util.tree_leaves(reset_pool))
    pool_wall = time.perf_counter() - pool_started
    started = time.perf_counter()
    telemetry_file = telemetry_path.open("w", encoding="utf-8")
    try:
        for index in range(n_updates):
            collect_started = time.perf_counter()
            batch = collect_selfplay_batch(
                params,
                num_envs=args.num_envs,
                rollout_len=args.rollout_len,
                seed=args.seed + index,
                reset_pool_size=RESET_POOL_SIZE,
                pool=reset_pool,
            )
            jax.block_until_ready(jax.tree_util.tree_leaves(batch["actions"]))
            collect_wall += time.perf_counter() - collect_started
            flat = flatten_batch(batch)
            update_started = time.perf_counter()
            params, opt_state, metrics = ppo_update(params, opt_state, optimizer, flat)
            jax.block_until_ready(jax.tree_util.tree_leaves(params))
            update_wall += time.perf_counter() - update_started
            ema = ema_update(ema, params)
            record = {
                "update": index,
                "transitions": (index + 1) * per_update,
                "collect_s": collect_wall,
                "update_s": update_wall,
                **{key: float(value) for key, value in metrics.items()},
            }
            record["healthy"] = healthy(metrics)
            records.append(record)
            telemetry_file.write(json.dumps(record, sort_keys=True) + "\n")
            telemetry_file.flush()
            if not record["healthy"]:
                stop_reason = "INTEGRITY_FAILURE"
                break
    finally:
        telemetry_file.close()

    transitions = len(records) * per_update
    valid_share = sum(1 for r in records if r["healthy"]) / max(len(records), 1)
    finite = [r for r in records if r["healthy"]]
    first = records[0] if records else {}
    last = records[-1] if records else {}
    finite_first = next((r for r in records if r["healthy"]), {})
    finite_last = next((r for r in reversed(records) if r["healthy"]), {})
    elimination = []
    if stop_reason == "INTEGRITY_FAILURE":
        elimination.append("INTEGRITY_FAILURE_NON_FINITE_OR_RATIO")
    if valid_share < 0.9:
        elimination.append("VALID_LEARNING_SHARE_BELOW_0.9")
    if finite and float(finite_last.get("entropy", 1.0)) < 0.05:
        elimination.append("ENTROPY_COLLAPSE")
    if finite_first and finite_last and finite_last["vloss"] >= finite_first["vloss"]:
        elimination.append("NO_VLOSS_REDUCTION")
    summary = {
        "kind": "SH_R1_ARM_SUMMARY",
        "arm_id": args.arm_id,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "geometry": {"num_envs": args.num_envs, "rollout_len": args.rollout_len},
        "seed": args.seed,
        "min_generals_distance": args.min_generals_distance,
        "budget_transitions": args.budget_transitions,
        "actual_transitions": transitions,
        "updates": len(records),
        "stop_reason": stop_reason,
        "metrics": {
            "VLOSS_FIRST": finite_first.get("vloss"),
            "VLOSS_LAST": finite_last.get("vloss"),
            "VLOSS_REDUCTION_OVER_ROUND": (
                finite_first.get("vloss", 0.0) - finite_last.get("vloss", 0.0)
                if finite_first and finite_last
                else None
            ),
            "VALID_LEARNING_SHARE": valid_share,
            "ENTROPY_FIRST": finite_first.get("entropy"),
            "ENTROPY_LAST": finite_last.get("entropy"),
            "PG_FIRST": finite_first.get("pg"),
            "PG_LAST": finite_last.get("pg"),
            "RATIO_FIRST": first.get("ratio"),
            "RATIO_LAST": last.get("ratio"),
        },
        "throughput": {
            "reset_pool_build_s": pool_wall,
            "collect_tps": transitions / max(collect_wall, 1e-9),
            "end_to_end_tps": transitions / max(collect_wall + update_wall, 1e-9),
            "collect_wall_s": collect_wall,
            "update_wall_s": update_wall,
            "total_wall_s": time.perf_counter() - started,
        },
        "elimination": elimination,
        "eliminated": bool(elimination),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if records:
        save_tree(out_dir / "raw.npz", params)
        save_tree(out_dir / "ema.npz", ema)
        save_tree(out_dir / "opt_state.npz", opt_state)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not elimination and stop_reason == "BUDGET_REACHED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
