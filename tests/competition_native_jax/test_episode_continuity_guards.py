"""EPISODE_CONTINUITY regression guards (audit amendment section 30, EV-0049).

Structural tests preventing recurrence of EARLY_WINDOW_RESET_REGIME_V1 in the
screening path:

  1. live non-terminal envs survive an auto-reset call (only done envs reset);
  2. the screening runner's default regime is PERSISTENT (the adopted
     canonical regime; a silent flip back to reset-per-update fails here);
  3. GAE bootstraps at non-terminal fragment boundaries (done=0 keeps the
     next value) while terminal masks fire only at true episode ends.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import jax.numpy as jnp

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "third_party" / "generals-bots")]

from generals_bot.competition_native_jax.competition_env_jax import (  # noqa: E402
    auto_reset_from_pool,
)
from train.competition_native_jax.gae_jax import gae_advantages  # noqa: E402


def test_only_done_envs_reset_live_envs_survive():
    # 3 envs, 1x1 boards: army value encodes identity
    states = jnp.asarray([10.0, 20.0, 30.0]).reshape(-1, 1, 1)
    pool = jnp.asarray([99.0, 98.0, 97.0]).reshape(-1, 1, 1)
    terminated = jnp.asarray([False, True, False])  # only env 1 done
    truncated = jnp.zeros(3, dtype=bool)
    cursor0 = jnp.zeros(3, dtype=jnp.int32)
    new_states, cursor = auto_reset_from_pool(states, terminated, truncated, pool, cursor0)
    out = [float(new_states[i, 0, 0]) for i in range(3)]
    assert out == [10.0, 99.0, 30.0]  # live envs untouched, done env replaced
    assert int(cursor[1]) == 1 and int(cursor[0]) == 0 and int(cursor[2]) == 0


def test_truncation_also_resets_only_that_env():
    states = jnp.asarray([5.0, 6.0]).reshape(-1, 1, 1)
    pool = jnp.asarray([77.0, 76.0]).reshape(-1, 1, 1)
    new_states, _ = auto_reset_from_pool(
        states, jnp.zeros(2, dtype=bool), jnp.asarray([False, True]),
        pool, jnp.zeros(2, dtype=jnp.int32)
    )
    assert float(new_states[0, 0, 0]) == 5.0
    assert float(new_states[1, 0, 0]) == 77.0  # cursor 0 -> pool[0]


def test_screening_runner_default_regime_is_persistent():
    source = (REPO / "scripts/training/run_sh_r1_arm.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = None
    for node in ast.walk(tree):
        is_add_arg = isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
        if not is_add_arg or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == "--episode-carry":
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    found = kw.value.value
    assert found == "persistent", (
        f"screening runner default regime drifted to {found!r}; "
        "PERSISTENT_EPISODE_REGIME_V1 is canonical (EV-0049)"
    )


def test_fragment_boundary_bootstraps_terminal_masks_only_true_ends():
    # 3-step fragment; done ONLY at the last step. With done=0 at steps 0-1
    # the next value must enter the TD target (bootstrap), and the terminal
    # mask must zero the bootstrap only at the true end.
    rewards = jnp.asarray([0.0, 0.0, 1.0])
    values = jnp.asarray([0.1, 0.2, 0.3, 0.9])  # last = bootstrap at boundary
    dones = jnp.asarray([0.0, 0.0, 1.0])
    adv, returns = gae_advantages(rewards, values, dones, gamma=1.0, lam=0.9)
    # final step: delta = 1 + 0 (masked) - 0.3
    assert abs(float(adv[2]) - 0.7) < 1e-6
    # step 1 (non-terminal): next value 0.3 participates (bootstrap alive)
    delta1 = 0.0 + 0.3 - 0.2
    assert abs(float(adv[1]) - (delta1 + 0.9 * 0.7)) < 1e-6
    # if the boundary were wrongly treated as terminal, adv[1] would equal
    # delta-with-zero-bootstrap (0.1) - guard the distinction
    assert abs(float(adv[1]) - (-0.1)) > 0.1
    assert bool(jnp.all(jnp.isfinite(returns)))
