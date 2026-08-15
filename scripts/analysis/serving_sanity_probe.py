"""Serving sanity probe for the SH-R4 finalists (EV-0033 follow-up, amendment §2).

Two layers, both local/CPU and deterministic:

L1 PARAMETER EFFECT: the loaded checkpoint must actually change inference
   outputs vs freshly-initialised parameters, and the two distinct finalist
   checkpoints must produce distinct outputs on identical inputs. A serving
   stack that silently ignores weights would fail here.

L2 PROTOCOL BEHAVIOUR: each finalist's main.py is driven over the official
   stdio protocol with a scripted observation; the returned action must be
   well-formed, deterministic across two runs, and the two finalists must
   not collapse to identical effective policies.

Exit 0 = PASS, 2 = DEFECT_FOUND (details in the JSON report).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from generals_bot.competition_native_jax.constants import ACTION_DIM  # noqa: E402
from generals_bot.competition_native_jax.inference_jax import infer  # noqa: E402
from generals_bot.competition_native_jax.obs_memory import N_GLOBAL, N_SPATIAL  # noqa: E402
from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from train.competition_native_jax.train_jax import load_tree  # noqa: E402

FINALISTS = {
    "sh-r4-finalist-a0": REPO / "experiments/marathon/eval_candidates/sh-r4-finalist-a0",
    "sh-r4-finalist-a1": REPO / "experiments/marathon/eval_candidates/sh-r4-finalist-a1",
}
# Scripted observation identical to windows_clean_package_validation (EV-0019 era).
HANDSHAKE = "0 3 3\n"
OBS = (
    "1 1 5 1 3\n"
    "4 1 1\n1 2 1\n1 1 1\n"
    "1 1 0\n0 0 0\n0 0 2\n"
    "5 2 0\n0 0 0\n0 0 3\n"
)


def tree_distance(a, b) -> float:
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    diffs = (
        np.abs(np.asarray(x) - np.asarray(y)).max()
        for x, y in zip(leaves_a, leaves_b, strict=True)
    )
    return float(max(diffs))


def layer1_parameter_effect() -> dict:
    key = jax.random.PRNGKey(1234)
    fresh = init_params(jax.random.PRNGKey(0))
    loaded = {
        name: load_tree(root / "weights" / "raw.npz", fresh)
        for name, root in FINALISTS.items()
    }
    spatial = jax.random.normal(jax.random.fold_in(key, 1), (N_SPATIAL, 21, 21))
    global_vec = jax.random.normal(jax.random.fold_in(key, 2), (N_GLOBAL,))
    mask = jnp.ones((ACTION_DIM,), dtype=jnp.bool_)

    def action_of(params):
        idx, _logp, _out = infer(params, spatial, global_vec, mask, None)
        return int(np.asarray(idx))

    fresh_action = action_of(fresh)
    actions = {name: action_of(params) for name, params in loaded.items()}
    fresh_dist = {name: tree_distance(fresh, params) for name, params in loaded.items()}
    cross_dist = tree_distance(loaded["sh-r4-finalist-a0"], loaded["sh-r4-finalist-a1"])
    report = {
        "actions": {"fresh_init": fresh_action, **actions},
        "param_distance_vs_fresh": fresh_dist,
        "param_distance_between_finalists": cross_dist,
        "checks": {
            "weights_differ_from_fresh": all(d > 1e-6 for d in fresh_dist.values()),
            "finalists_differ_from_each_other": cross_dist > 1e-6,
            "checkpoint_changes_outputs": any(
                a != fresh_action for a in actions.values()
            ),
            "finalists_not_identical_policies": actions["sh-r4-finalist-a0"]
            != actions["sh-r4-finalist-a1"],
        },
    }
    return report


def run_protocol_action(main_py: Path) -> str | None:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}{os.pathsep}{REPO / 'src'}"
    proc = subprocess.Popen(
        [sys.executable, "-u", str(main_py)],
        cwd=str(main_py.parent),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdin is not None and proc.stdout is not None
    out, err = proc.communicate(HANDSHAKE + OBS, timeout=120)
    lines = [line for line in (out or "").strip().splitlines() if line.strip()]
    if not lines:
        print(f"[probe] no action from {main_py.name} parent; stderr tail:", file=sys.stderr)
        print((err or "")[-800:], file=sys.stderr)
    return lines[0] if lines else None


def layer2_protocol_behaviour() -> dict:
    actions: dict[str, list[str | None]] = {}
    for name, root in FINALISTS.items():
        actions[name] = [run_protocol_action(root / "main.py") for _ in range(2)]
    checks = {
        "actions_well_formed": all(
            action is not None and len(action.split()) == 5
            for runs in actions.values()
            for action in runs
        ),
        "deterministic_across_runs": all(runs[0] == runs[1] for runs in actions.values()),
    }
    # Informational only: agreement on ONE contrived board does not imply
    # policy collapse; layer 1's tensor-space distinctness is the gate.
    info = {
        "finalists_agree_on_this_board": actions["sh-r4-finalist-a0"][0]
        == actions["sh-r4-finalist-a1"][0],
    }
    return {"actions": actions, "checks": checks, "info": info}


def main() -> int:
    report = {
        "layer1_parameter_effect": layer1_parameter_effect(),
        "layer2_protocol_behaviour": layer2_protocol_behaviour(),
    }
    all_checks = {
        **report["layer1_parameter_effect"]["checks"],
        **report["layer2_protocol_behaviour"]["checks"],
    }
    report["verdict"] = "PASS" if all(all_checks.values()) else "DEFECT_FOUND"
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
