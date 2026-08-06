"""Hybrid package-runtime confirmation vs source factory (deployment confirmation)."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from generals import GeneralsEnv
from generals.core import game
import jax.numpy as jnp

from generals_bot.action import PASS_ACTION
from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.observation import GameContext
from generals_bot.policies.base import TraceLevel
from generals_bot.policies.hybrid_bc_ranker import HybridBcRankerPolicy, HybridConfidenceConfig
from generals_bot.training.bridge_benchmark import extract_numpy_boards
from generals_bot.training.collect_bc import _action_to_jax, _observation_from_arrays

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "submission/packages/QS-P9FU-HYBRID-BC-V1/5152a08eb774cf0e/package.zip"
BC = REPO / "experiments/phase9f_cnn_ranker_v1/checkpoints/bc/model.json"
FIXTURE_SEEDS = [101, 102, 103]
MAX_TURNS = 40


def _rollout(policy, seed: int) -> list[tuple]:
    env = GeneralsEnv(mode="competition")
    transition = make_transition(env)
    get_obs = game.get_observation
    state = make_board(env, seed)
    h, w = (int(d) for d in state.armies.shape)
    st = policy.initial_state(GameContext(0, h, w))
    actions = []
    for turn_i in range(MAX_TURNS):
        eng = get_obs(state, 0)
        t, o, a, _, m = extract_numpy_boards(eng, h, w)
        obs = _observation_from_arrays(t, o, a, m)
        d = policy.act(obs, st, deterministic=True, trace=TraceLevel.NONE, deadline=None)
        st = d.new_state
        actions.append(d.action.as_tuple())
        # opponent always PASS for identical trajectory of focal policy inputs
        state, info = transition(
            state, jnp.stack([_action_to_jax(d.action), _action_to_jax(PASS_ACTION)])
        )
        if bool(info.is_done):
            break
    return actions


def main() -> int:
    cfg = HybridConfidenceConfig()
    src = HybridBcRankerPolicy(checkpoint_json=BC, device="cpu", confidence=cfg)
    staging = Path(tempfile.mkdtemp(prefix="hybrid_pkg_confirm_"))
    divergences = []
    try:
        with zipfile.ZipFile(PKG, "r") as zf:
            zf.extractall(staging)
        sys.path.insert(0, str(staging))
        # Import packaged module under an alias path — package already on path.
        from generals_bot.policies.hybrid_bc_ranker import (  # type: ignore
            HybridBcRankerPolicy as PkgHybrid,
        )

        # Prefer package-relative model if present
        model_candidates = list(staging.rglob("model.json"))
        ckpt = model_candidates[0] if model_candidates else BC
        pkg_pol = PkgHybrid(checkpoint_json=ckpt, device="cpu", confidence=cfg)
        comparisons = []
        for seed in FIXTURE_SEEDS:
            a_src = _rollout(src, seed)
            a_pkg = _rollout(pkg_pol, seed)
            equal = a_src == a_pkg
            comparisons.append(
                {
                    "seed": seed,
                    "turns": min(len(a_src), len(a_pkg)),
                    "exact_action_equality": equal,
                }
            )
            if not equal:
                divergences.append({"seed": seed, "src": a_src[:5], "pkg": a_pkg[:5]})
        status = "PASS" if not divergences else "PACKAGE_RUNTIME_DIVERGENCE"
        out = {
            "schema_version": 1,
            "kind": "HYBRID_PACKAGE_RUNTIME_CONFIRMATION",
            "status": status,
            "package": str(PKG.as_posix()),
            "sha256": "5152a08eb774cf0e29167e9469422834b0a6e40392a6035ccc0f830d50674b9f",
            "fixture_seeds": FIXTURE_SEEDS,
            "max_turns": MAX_TURNS,
            "comparisons": comparisons,
            "divergences": divergences,
            "note": "Deployment confirmation only; does not modify Stage 3 competitive scores.",
        }
    finally:
        # remove staging from path
        if str(staging) in sys.path:
            sys.path.remove(str(staging))
        shutil.rmtree(staging, ignore_errors=True)

    path = REPO / "experiments/manifests/phase9fu_hybrid_package_runtime_confirmation.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "path": str(path)}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
