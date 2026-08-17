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
import hashlib
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from generals_bot.competition_native_jax.competition_env_jax import (  # noqa: E402
    build_competition_reset_pool,
)
from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from train.competition_native_jax.ema_jax import ema_update  # noqa: E402
from train.competition_native_jax.gae_jax import gae_advantages_batch_jit  # noqa: E402
from train.competition_native_jax.ppo_jax import make_optimizer, ppo_update  # noqa: E402
from train.competition_native_jax.reward_shaping_jax import set_active_shaping  # noqa: E402
from generals_bot.competition_native_jax.obs_v2_jax import (  # noqa: E402
    N_GLOBAL_V2,
    N_SPATIAL_V2,
)
from train.competition_native_jax.rollout_selfplay_jax import (  # noqa: E402
    collect_selfplay_batch,
    set_obs_version,
    set_opponent_mode,
)
from train.competition_native_jax.curriculum_eval_jax import (  # noqa: E402
    greedy_win_rate_vs_random,
)
from train.competition_native_jax.temporal_history_jax import (  # noqa: E402
    set_temporal_history_mode,
)
from train.competition_native_jax.top_advantage_jax import top_advantage_mask  # noqa: E402
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


def params_sha256(tree: dict) -> str:
    """STAGE6_OPPDIST_R1: hash opponent param leaves (never-updated invariant)."""
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        h.update(np.ascontiguousarray(np.asarray(leaf)).view(np.uint8))
    return h.hexdigest()


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
    parser.add_argument(
        "--top-advantage-fraction",
        type=float,
        default=1.0,
        help="TOPADV knob: keep PG signal on top fraction of |advantage| transitions "
        "(1.0 = identity/default; PPO_SEMANTICS UNCHANGED for serving)",
    )
    parser.add_argument(
        "--reward-shape",
        default="none",
        choices=["none", "kill_delta", "potential", "land_potential"],
        help="REWARD-SHAPING-R1 knob (EV-0044): bounded progress signal on non-terminal "
        "ticks (none = identity/control; PPO_SEMANTICS UNCHANGED for serving)",
    )
    parser.add_argument(
        "--reward-shape-beta",
        type=float,
        default=0.0,
        help="REWARD-SHAPING-R1 shaping coefficient (0.0 = identity)",
    )
    parser.add_argument(
        "--episode-carry",
        default="persistent",
        choices=["none", "persistent"],
        help="EPISODE_CONTINUITY knob (LEARNING-PATH-INTEGRITY audit, EV-0048/0049): "
        "persistent = CANONICAL regime (PERSISTENT_EPISODE_REGIME_V1, adopted by "
        "EPISODE_CONTINUITY_R1): thread the RolloutCarry across PPO updates so live "
        "games continue until true competition terminal/reset. none = historical "
        "EARLY_WINDOW_RESET_REGIME_V1 retained ONLY for exact reproduction of "
        "registered reset-regime runs. PPO_SEMANTICS UNCHANGED.",
    )
    parser.add_argument(
        "--temporal-history",
        default="off",
        choices=["off", "k1"],
        help="STAGE5 T2 knob (stage5_capacity_value_r1_plan.yaml): k1 appends the "
        "previous tick's LEGAL spatial observation as extra input planes "
        "(spatial 8 -> 16; shared layers warm-started from the checkpoint, "
        "patch_proj fresh - shape-forced). off = canonical 8-plane path. "
        "PPO_SEMANTICS UNCHANGED for serving.",
    )
    parser.add_argument(
        "--obs-version",
        default="v1",
        choices=["v1", "v2"],
        help="OBS_V2_R1 knob (obs_v2_r1_plan.yaml): v1 = canonical 8-plane/8-global "
        "observation (bit-identical control path). v2 = objective-aware 14-plane/"
        "12-global observation (legal scoreboard, cell-type identity, enemy memory; "
        "EV-0071); warm-started from the checkpoint via DECLARED shape surgery - "
        "patch_proj 72->126 and global_proj 8->12 keep the old rows in place, new "
        "rows deterministic (fixed seed), opt_state re-initialised, EMA re-"
        "initialised from new params. Incompatible with --temporal-history k1 and "
        "only predeclared with --opponent-mode self. PPO_SEMANTICS UNCHANGED.",
    )
    parser.add_argument(
        "--opponent-mode",
        default="self",
        choices=["self", "teacher_frozen"],
        help="STAGE6_OPPDIST_R1 knob (stage6_oppdist_r1_plan.yaml): self = canonical "
        "mirror self-play (bit-identical control path). teacher_frozen = seat 1 acts "
        "from --opponent-checkpoint params loaded ONCE and never updated, sampled "
        "stochastically at temperature 1.0 with its own obs/memory threading; PPO "
        "updates apply to seat-0 samples only. The opponent is environment dynamics; "
        "PPO_SEMANTICS UNCHANGED for the trained seat.",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        default=None,
        help="STAGE6_OPPDIST_R1: directory containing raw.npz of the frozen opponent "
        "(required when --opponent-mode teacher_frozen).",
    )
    parser.add_argument(
        "--schedules",
        default="none",
        choices=["none", "rc1"],
        help="RC_R1_BRIDGE knob (rc_r1_bridge_plan.yaml delta D4): rc1 applies the "
        "predeclared frozen power-law schedules - ent_coef_t = max(0.05/(t+1)^0.2, "
        "0.001) and lr_t = clip(4.5e-3/(t+1)^1.1, 5e-6, 1e-4) (reference power-law "
        "shape, amplitude matched to ~1e-5 terminal LR at 256 updates); opt_state is "
        "re-initialised under rc1 (schedule changes optimiser semantics; EMA still "
        "continues from checkpoint). none = canonical constant ent 0.01 / LR 3e-4 "
        "(control path). PPO_SEMANTICS UNCHANGED for serving.",
    )
    parser.add_argument(
        "--curriculum",
        default="none",
        choices=["none", "competence-spawn"],
        help="RC_R1_BRIDGE knob (delta D2): competence-spawn runs spawn-distance "
        "stages [8, 17] starting at 8; every 32 updates a greedy-vs-legal_random "
        "diagnostic eval (64 envs, 1200-turn cap) is run; win rate >= 0.6 advances "
        "the stage and regenerates the reset pool. none = fixed min_generals_distance "
        "(control path). PPO_SEMANTICS UNCHANGED for serving.",
    )
    parser.add_argument(
        "--accumulate-minibatches",
        type=int,
        default=None,
        help="ENGINEERING MEMORY KNOB (RC-R1 OOM repair): accumulate mean gradients "
        "over this many static shards with lax.scan, then apply exactly ONE optimiser "
        "step (ppo_update canonical semantics; parameters unchanged between shards). "
        "None = single full-batch grad (control path). No scientific-semantics change: "
        "same single-step mean-gradient update, different float-summation order.",
    )
    args = parser.parse_args()

    set_active_shaping(args.reward_shape, args.reward_shape_beta)
    set_temporal_history_mode(args.temporal_history)
    if args.obs_version == "v2":
        if args.temporal_history != "off":
            print(
                "obs-version v2 (14 planes) is incompatible with temporal-history k1 "
                "(which forces 16 planes)",
                file=sys.stderr,
            )
            return 2
        if args.opponent_mode != "self":
            print(
                "obs-version v2 is only predeclared with --opponent-mode self "
                "(frozen opponents are v1-shaped)",
                file=sys.stderr,
            )
            return 2
    set_obs_version(args.obs_version)
    if args.opponent_mode == "teacher_frozen":
        if args.temporal_history != "off":
            print(
                "teacher_frozen opponent requires the canonical 8-plane path "
                "(--temporal-history off)",
                file=sys.stderr,
            )
            return 2
        if args.opponent_checkpoint is None:
            print("--opponent-checkpoint is required with --opponent-mode teacher_frozen", file=sys.stderr)
            return 2
    set_opponent_mode(args.opponent_mode)
    opp_params = None
    opp_hash_before = None
    if args.opponent_mode == "teacher_frozen":
        opp_like = init_params(jax.random.PRNGKey(0))
        opp_params = load_tree(args.opponent_checkpoint / "raw.npz", opp_like)
        opp_hash_before = params_sha256(opp_params)

    out_dir = args.out_dir or REPO / f"experiments/marathon/screening_runs/{args.arm_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = out_dir / "telemetry.jsonl"
    if telemetry_path.exists():
        print(f"refusing to overwrite existing telemetry: {telemetry_path}", file=sys.stderr)
        return 2

    if args.schedules == "rc1":
        # RC_R1_BRIDGE D4 (frozen): lr_t = clip(4.5e-3/(t+1)^1.1, 5e-6, 1e-4).
        def rc1_lr(step):
            return jnp.clip(4.5e-3 / jnp.power(step + 1.0, 1.1), 5e-6, 1e-4)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.inject_hyperparams(optax.adam)(learning_rate=rc1_lr),
        )
    else:
        optimizer = make_optimizer(LR)
    if args.temporal_history == "k1":
        # STAGE5 T2: shared layers warm-started from the checkpoint; patch_proj
        # is shape-forced fresh (8 -> 16 input planes); opt/EMA re-initialised.
        params_like = init_params(jax.random.PRNGKey(0), spatial_planes=16)
        base_like = init_params(jax.random.PRNGKey(0))
        base_params = load_tree(args.checkpoint / "raw.npz", base_like)
        params = {**base_params, "patch_proj": params_like["patch_proj"]}
        opt_state = optimizer.init(params)
        ema = jax.tree_util.tree_map(jnp.asarray, params)
        obs_shape_surgery = None
    elif args.obs_version == "v2":
        # OBS_V2_R1 warm-start shape surgery (DECLARED, obs_v2_r1_plan.yaml
        # parity_mandate_frozen): v2 planes 0-7 / globals 0-7 are order-identical
        # to v1, so the old patch_proj rows (72) and global_proj rows (8) are
        # kept in place; new rows are deterministic (fixed seed 0 init); opt is
        # re-initialised and EMA re-initialised from the new params (T2 precedent).
        params_like = init_params(
            jax.random.PRNGKey(0),
            spatial_planes=N_SPATIAL_V2,
            global_dim=N_GLOBAL_V2,
        )
        base_like = init_params(jax.random.PRNGKey(0))
        base_params = load_tree(args.checkpoint / "raw.npz", base_like)
        params = {
            **base_params,
            "patch_proj": jnp.concatenate(
                [
                    base_params["patch_proj"],
                    params_like["patch_proj"][base_params["patch_proj"].shape[0] :],
                ],
                axis=0,
            ),
            "global_proj": jnp.concatenate(
                [
                    base_params["global_proj"],
                    params_like["global_proj"][base_params["global_proj"].shape[0] :],
                ],
                axis=0,
            ),
        }
        opt_state = optimizer.init(params)
        ema = jax.tree_util.tree_map(jnp.asarray, params)
        obs_shape_surgery = {
            "patch_proj": [list(base_params["patch_proj"].shape), list(params["patch_proj"].shape)],
            "global_proj": [list(base_params["global_proj"].shape), list(params["global_proj"].shape)],
            "old_rows_preserved": True,
            "new_rows_seed": 0,
            "opt_state": "fresh",
            "ema": "fresh_from_new_params",
        }
    elif args.schedules == "rc1":
        # RC_R1_BRIDGE: raw/EMA continue from the warm start; opt_state fresh
        # (schedule changes optimiser semantics - declared in the plan).
        params_like = init_params(jax.random.PRNGKey(0))
        params = load_tree(args.checkpoint / "raw.npz", params_like)
        ema = load_tree(args.checkpoint / "ema.npz", params_like)
        opt_state = optimizer.init(params)
        obs_shape_surgery = None
    else:
        params_like = init_params(jax.random.PRNGKey(0))
        opt_like = optimizer.init(params_like)
        params = load_tree(args.checkpoint / "raw.npz", params_like)
        ema = load_tree(args.checkpoint / "ema.npz", params_like)
        opt_state = load_tree(args.checkpoint / "opt_state.npz", opt_like)
        obs_shape_surgery = None

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
    rollout_carry = None  # EPISODE_CONTINUITY knob: threaded only when persistent
    # RC_R1_BRIDGE D2 (frozen): competence-spawn stages, start at the easy
    # (close-spawn) stage; advancement rule lives in the update loop below.
    curriculum_stage = None
    curriculum_stages = None
    curriculum_evals = []
    eff_min_distance = args.min_generals_distance
    if args.curriculum == "competence-spawn":
        curriculum_stages = [8, 17]
        curriculum_stage = 0
        eff_min_distance = curriculum_stages[0]
    # Build the reset pool ONCE and reuse it across updates (ladder pattern,
    # EV-0029): reconstructing boards per collect dominated wall-time.
    # Environment initialisation only; PPO_SEMANTICS unchanged.
    pool_started = time.perf_counter()
    reset_pool = build_competition_reset_pool(
        jax.random.PRNGKey(args.seed),
        RESET_POOL_SIZE,
        min_generals_distance=eff_min_distance,
    )
    jax.block_until_ready(jax.tree_util.tree_leaves(reset_pool))
    pool_wall = time.perf_counter() - pool_started
    started = time.perf_counter()
    telemetry_file = telemetry_path.open("w", encoding="utf-8")
    try:
        for index in range(n_updates):
            collect_started = time.perf_counter()
            if args.episode_carry == "persistent":
                batch, rollout_carry = collect_selfplay_batch(
                    params,
                    num_envs=args.num_envs,
                    rollout_len=args.rollout_len,
                    seed=args.seed,
                    reset_pool_size=RESET_POOL_SIZE,
                    pool=reset_pool,
                    carry=rollout_carry,
                    return_carry=True,
                    opp_params=opp_params,
                )
            else:
                batch = collect_selfplay_batch(
                    params,
                    num_envs=args.num_envs,
                    rollout_len=args.rollout_len,
                    seed=args.seed + index,
                    reset_pool_size=RESET_POOL_SIZE,
                    pool=reset_pool,
                    opp_params=opp_params,
                )
            jax.block_until_ready(jax.tree_util.tree_leaves(batch["actions"]))
            collect_wall += time.perf_counter() - collect_started
            flat = flatten_batch(batch)
            if args.top_advantage_fraction < 1.0:
                flat["advantages"] = top_advantage_mask(
                    flat["advantages"], args.top_advantage_fraction
                )
            update_started = time.perf_counter()
            if args.schedules == "rc1":
                # RC_R1_BRIDGE D4 (frozen): ent_coef_t = max(0.05/(t+1)^0.2, 0.001)
                ent_t = max(0.05 / ((index + 1) ** 0.2), 0.001)
            else:
                ent_t = 0.01
            params, opt_state, metrics = ppo_update(
                params, opt_state, optimizer, flat, ent_coef=ent_t,
                accumulate_minibatches=args.accumulate_minibatches,
            )
            jax.block_until_ready(jax.tree_util.tree_leaves(params))
            update_wall += time.perf_counter() - update_started
            ema = ema_update(ema, params)
            # RC_R1_BRIDGE D2 (frozen advancement rule): diagnostic greedy-vs-
            # legal_random eval every 32 updates; win rate >= 0.6 advances the
            # spawn-distance stage and regenerates the reset pool.
            curriculum_eval = None
            if args.curriculum == "competence-spawn" and (index + 1) % 32 == 0:
                curriculum_eval = greedy_win_rate_vs_random(
                    params, reset_pool, num_envs=64, horizon=1200,
                    seed=args.seed + index,
                )
                curriculum_eval["update"] = index
                curriculum_eval["stage_before"] = curriculum_stage
                advanced = (
                    curriculum_eval["win_rate_vs_decided"] >= 0.6
                    and curriculum_stage < len(curriculum_stages) - 1
                )
                if advanced:
                    curriculum_stage += 1
                    eff_min_distance = curriculum_stages[curriculum_stage]
                    reset_pool = build_competition_reset_pool(
                        jax.random.PRNGKey(args.seed),
                        RESET_POOL_SIZE,
                        min_generals_distance=eff_min_distance,
                    )
                    jax.block_until_ready(jax.tree_util.tree_leaves(reset_pool))
                    # Curriculum boundary: episodes restart from the new pool
                    # (declared; mirrors reference pool regeneration).
                    rollout_carry = None
                curriculum_eval["stage_after"] = curriculum_stage
                curriculum_eval["advanced"] = bool(advanced)
                curriculum_evals.append(curriculum_eval)
            record = {
                "update": index,
                "transitions": (index + 1) * per_update,
                "collect_s": collect_wall,
                "update_s": update_wall,
                "decisive_share": float(jnp.mean(batch["terminals"])),
                "opp_win_share": float(jnp.mean((batch["rewards1"] > 0.5) * batch["terminals"])),
                "ent_coef": ent_t,
                **({"curriculum": curriculum_eval} if curriculum_eval is not None else {}),
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
    opp_hash_after = None
    if opp_params is not None:
        final_opp = rollout_carry.opp_params if rollout_carry is not None else opp_params
        opp_hash_after = params_sha256(final_opp)
        if opp_hash_after != opp_hash_before:
            elimination.append("OPPONENT_PARAMS_MUTATED")
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
        "top_advantage_fraction": args.top_advantage_fraction,
        "reward_shape": args.reward_shape,
        "reward_shape_beta": args.reward_shape_beta,
        "episode_carry": args.episode_carry,
        "temporal_history": args.temporal_history,
        "obs_version": args.obs_version,
        "obs_shape_surgery": obs_shape_surgery,
        "opponent_mode": args.opponent_mode,
        "opponent_checkpoint": str(args.opponent_checkpoint) if args.opponent_checkpoint else None,
        "opponent_sha256_before": opp_hash_before,
        "opponent_sha256_after": opp_hash_after,
        "schedules": args.schedules,
        "curriculum": args.curriculum,
        "curriculum_stages": curriculum_stages,
        "curriculum_final_stage": curriculum_stage,
        "curriculum_evals": curriculum_evals,
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
            "DECISIVE_SHARE_FIRST": finite_first.get("decisive_share"),
            "DECISIVE_SHARE_LAST": finite_last.get("decisive_share"),
            "OPP_WIN_SHARE_FIRST": finite_first.get("opp_win_share"),
            "OPP_WIN_SHARE_LAST": finite_last.get("opp_win_share"),
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
