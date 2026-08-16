"""LEARNING-PATH-INTEGRITY AUDIT probe: episode continuity across PPO updates.

Proves empirically (not from function names) whether environment state
survives PPO update boundaries under the two collection semantics:

  A. NO-CARRY (current screening runner `run_sh_r1_arm.py` semantics):
     every update re-initialises the rollout carry -> all envs restart at
     turn 0 from the first num_envs pool boards.
  B. CARRY (canonical trainer `cloud_gpu_last_push.py` semantics):
     the RolloutCarry is threaded across updates -> live games continue,
     turn counters increase, only done envs reset (per-env auto-reset).

Diagnostic-only. PPO_SEMANTICS: EVAL_ONLY (no training, no parameter update).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "third_party" / "generals-bots")]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from train.competition_native_jax.rollout_selfplay_jax import (  # noqa: E402
    collect_selfplay_batch,
    initialise_rollout_carry,
)

NUM_ENVS = 4
ROLLOUT_LEN = 8
SEED = 20260816
N_UPDATES = 3


def state_hash(states, env: int) -> str:
    leaves = jax.tree_util.tree_leaves(states)
    h = hashlib.sha256()
    for leaf in leaves:
        h.update(bytes(memoryview(jnp.asarray(leaf[env]))))
    return h.hexdigest()[:16]


def main() -> None:
    params = init_params(jax.random.PRNGKey(0))
    report: dict = {"num_envs": NUM_ENVS, "rollout_len": ROLLOUT_LEN}

    # ---- A. no-carry semantics (screening runner) ----
    no_carry: list[dict] = []
    for update in range(N_UPDATES):
        carry = initialise_rollout_carry(
            params, num_envs=NUM_ENVS, seed=SEED + update, reset_pool_size=64
        )
        start_time = [int(t) for t in carry.states.time]
        start_hashes = [state_hash(carry.states, e) for e in range(NUM_ENVS)]
        batch, final = collect_selfplay_batch(
            params,
            num_envs=NUM_ENVS,
            rollout_len=ROLLOUT_LEN,
            seed=SEED + update,
            reset_pool_size=64,
            carry=carry,
            return_carry=True,
        )
        end_time = [int(t) for t in final.states.time]
        no_carry.append(
            {
                "update": update,
                "start_time": start_time,
                "end_time": end_time,
                "start_hash_env0": start_hashes[0],
                "dones_in_window": int(jnp.sum(batch["dones"])),
            }
        )
    report["A_no_carry_screening"] = no_carry
    report["A_resets_every_update"] = all(
        rec["start_time"] == [0] * NUM_ENVS for rec in no_carry
    )
    report["A_identical_start_states_across_updates_env0"] = len(
        {rec["start_hash_env0"] for rec in no_carry}
    ) == 1

    # ---- B. carry semantics (canonical trainer) ----
    with_carry: list[dict] = []
    carry = None
    prev_end = None
    for update in range(N_UPDATES):
        if carry is None:
            batch, carry = collect_selfplay_batch(
                params,
                num_envs=NUM_ENVS,
                rollout_len=ROLLOUT_LEN,
                seed=SEED,
                reset_pool_size=64,
                return_carry=True,
            )
            start_hashes = [state_hash(carry.states, e) for e in range(NUM_ENVS)]
        else:
            start_hashes = [state_hash(carry.states, e) for e in range(NUM_ENVS)]
            batch, carry = collect_selfplay_batch(
                params,
                num_envs=NUM_ENVS,
                rollout_len=ROLLOUT_LEN,
                seed=SEED,
                reset_pool_size=64,
                carry=carry,
                return_carry=True,
            )
        with_carry.append(
            {
                "update": update,
                "start_time": [int(t) for t in carry.states.time] if update else [0] * NUM_ENVS,
                "end_time": [int(t) for t in carry.states.time],
                "dones_in_window": int(jnp.sum(batch["dones"])),
            }
        )
        if prev_end is not None:
            with_carry[-1]["continuity_hash_match_env0"] = (
                prev_end[0] == start_hashes[0]
            )
        prev_end = [state_hash(carry.states, e) for e in range(NUM_ENVS)]
    report["B_carry_canonical"] = with_carry
    max_time = max(max(rec["end_time"]) for rec in with_carry)
    report["B_turn_survives_updates"] = max_time > ROLLOUT_LEN
    report["B_max_turn_reached"] = max_time

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
