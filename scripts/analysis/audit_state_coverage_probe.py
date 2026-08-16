"""LEARNING-PATH-INTEGRITY AUDIT probe: training-state turn coverage.

Diagnostic-only measurement (no PPO updates) of which competition turns the
collected samples actually occupy, under:

  A. NO-CARRY semantics (current screening runner): every collection restarts
     all envs at turn 0 -> coverage provably limited to turns [0, rollout_len).
  B. CARRY semantics (canonical trainer): games persist across collections ->
     coverage extends through mid/endgame turns, terminals become reachable.

Reports the audit's turn buckets plus terminal counters.
PPO_SEMANTICS: EVAL_ONLY.
"""

from __future__ import annotations

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
)

NUM_ENVS = 64
ROLLOUT_LEN = 64
SEED = 20260816
BUCKETS = [(0, 32), (32, 100), (100, 200), (200, 400), (400, 800), (800, 1000), (1000, 1200)]


def bucket_counts(times: list[int]) -> dict:
    counts = {f"{lo}-{hi - 1}": 0 for lo, hi in BUCKETS}
    for t in times:
        for lo, hi in BUCKETS:
            if lo <= t < hi:
                counts[f"{lo}-{hi - 1}"] += 1
                break
    return counts


def main() -> None:
    params = init_params(jax.random.PRNGKey(0))
    report = {"num_envs": NUM_ENVS, "rollout_len": ROLLOUT_LEN}

    # A. no-carry: every window restarts at turn 0 (proven by
    # audit_episode_continuity_probe.py), so coverage is exactly 0..63.
    batch = collect_selfplay_batch(
        params, num_envs=NUM_ENVS, rollout_len=ROLLOUT_LEN, seed=SEED, reset_pool_size=512
    )
    times_a = list(range(ROLLOUT_LEN)) * NUM_ENVS
    report["A_no_carry_screening"] = {
        "samples": batch["rewards"].size,
        "bucket_counts": bucket_counts(times_a),
        "max_turn": ROLLOUT_LEN - 1,
        "terminal_wins_or_losses": int(jnp.sum(batch["terminals"])),
        "truncations_1200": int(jnp.sum(batch["dones"] - batch["terminals"])),
    }

    # B. carry: attribute each env's fragment to its window-start turn.
    carry = None
    times_b: list[int] = []
    terminals = truncations = samples = 0
    windows = 6
    max_start = 0
    for _ in range(windows):
        start = [0] * NUM_ENVS if carry is None else [int(t) for t in carry.states.time]
        max_start = max(max_start, max(start))
        batch, carry = collect_selfplay_batch(
            params,
            num_envs=NUM_ENVS,
            rollout_len=ROLLOUT_LEN,
            seed=SEED,
            reset_pool_size=512,
            carry=carry,
            return_carry=True,
        )
        for env in range(NUM_ENVS):
            times_b.extend([start[env]] * ROLLOUT_LEN)
        terminals += int(jnp.sum(batch["terminals"]))
        truncations += int(jnp.sum(batch["dones"] - batch["terminals"]))
        samples += batch["rewards"].size
    report["B_carry_canonical"] = {
        "samples": samples,
        "windows": windows,
        "bucket_counts": bucket_counts(times_b),
        "max_window_start_turn": max_start,
        "terminal_wins_or_losses": terminals,
        "truncations_1200": truncations,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
