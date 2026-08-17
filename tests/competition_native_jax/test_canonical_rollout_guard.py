"""Static + behavioural guard for the ONE-canonical-rollout invariant.

Amendment section 13 / REPOSITORY_CANONICALISATION_PLAN_V1: the screening
runner previously bypassed the already-correct RolloutCarry and reintroduced
reset-per-update training (EV-0048). This guard prevents a second rollout
implementation from silently appearing:

1. training entrypoints (scripts/training/) must collect via
   collect_selfplay_batch and must never import the raw environment step
   kernel for a private rollout loop;
2. the raw step kernel's importers stay inside the canonical allow-list;
3. persistent carry threading actually advances episode state across
   successive collects (the EV-0048 regression).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from train.competition_native_jax.rollout_selfplay_jax import (  # noqa: E402
    collect_selfplay_batch,
    initialise_rollout_carry,
)

ALLOWED_STEP_KERNEL_IMPORTERS = {
    "src/generals_bot/competition_native_jax/competition_env_jax.py",  # definition
    "train/competition_native_jax/rollout_selfplay_jax.py",  # canonical rollout
    "train/competition_native_jax/curriculum_eval_jax.py",  # diagnostic eval
    "train/competition_native_jax/matched_benchmarks_v4_2.py",  # legacy bench (superseded, kept)
    "tests/competition_native_jax/test_long_horizon_engine_parity.py",  # engine parity
}


def test_training_entrypoints_use_canonical_collector():
    scripts = list((REPO / "scripts" / "training").glob("*.py"))
    assert scripts, "scripts/training entrypoints missing"
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert "step_batch_jax" not in text, (
            f"{path.name} imports the raw step kernel - private rollout loops "
            "are forbidden (EV-0048); use collect_selfplay_batch"
        )
    runner = (REPO / "scripts/training/run_sh_r1_arm.py").read_text(encoding="utf-8")
    assert "collect_selfplay_batch" in runner


def test_step_kernel_importers_within_allowlist():
    offenders = []
    self_rel = Path(__file__).resolve().relative_to(REPO).as_posix()
    for path in REPO.rglob("*.py"):
        rel = path.relative_to(REPO).as_posix()
        if rel == self_rel:
            continue
        if "third_party" in rel or "__pycache__" in rel or ".venv" in rel:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if re.search(r"\bstep_batch_jax\b", text) and rel not in ALLOWED_STEP_KERNEL_IMPORTERS:
            offenders.append(rel)
    assert not offenders, (
        f"step_batch_jax imported outside the canonical allow-list: {offenders}; "
        "training rollouts must go through rollout_selfplay_jax"
    )


def test_persistent_carry_advances_episode_state():
    params = init_params(jax.random.PRNGKey(0))
    carry = initialise_rollout_carry(params, num_envs=2, seed=1, reset_pool_size=8)
    batch, carry2 = collect_selfplay_batch(
        params, num_envs=2, rollout_len=4, seed=1, reset_pool_size=8,
        carry=carry, return_carry=True,
    )
    assert batch["rewards"].shape == (4, 2)
    # Episode state advanced - the carry was threaded, not rebuilt per update.
    mem_before = jnp.asarray(carry.mem0.seen_own).sum()
    mem_after = jnp.asarray(carry2.mem0.seen_own).sum()
    states_changed = not bool(
        jnp.array_equal(
            jax.tree_util.tree_leaves(carry.states)[0],
            jax.tree_util.tree_leaves(carry2.states)[0],
        )
    )
    assert states_changed or mem_after != mem_before, (
        "carry threading produced no episode advancement - reset-per-update "
        "regression (EV-0048)"
    )
    # A third collect reusing carry2 must accept it (persistent regime chain).
    _batch3, _carry3 = collect_selfplay_batch(
        params, num_envs=2, rollout_len=4, seed=1, reset_pool_size=8,
        carry=carry2, return_carry=True,
    )
