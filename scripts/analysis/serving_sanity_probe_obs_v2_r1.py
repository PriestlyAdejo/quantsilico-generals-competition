"""Pre-game serving sanity probe for the OBS_V2_R1 gameplay arbiter.

EV-0034 precedent + rc_r1/t2/distill probe pattern. Two layers, local/CPU,
deterministic:

L1 PARAMETER EFFECT (OBS-V2 14-plane/12-global path): each loaded OBS-V2
   terminal checkpoint must change inference outputs vs the DECLARED
   warm-start shape-surgery template (MARATHON_BASELINE_V0 rows preserved +
   deterministic new rows, exactly as run_sh_r1_arm.py builds them - proves
   training changed the weights), and the two checkpoints must produce
   distinct outputs on identical inputs.
L2 PROTOCOL BEHAVIOUR: each candidate main.py is driven over the official
   stdio protocol twice; actions must be well-formed 5-field competition
   actions and deterministic across runs.

Requires the terminal checkpoints fetched to
experiments/marathon/obs_v2_r1/<ARM>/raw.npz. Must PASS before any gameplay
games are counted. Exit 0 = PASS, 2 = DEFECT.
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

from generals_bot.competition_native_jax.constants import ACTION_DIM, MAX_HW  # noqa: E402
from generals_bot.competition_native_jax.inference_jax import infer  # noqa: E402
from generals_bot.competition_native_jax.obs_v2_jax import (  # noqa: E402
    N_GLOBAL_V2,
    N_SPATIAL_V2,
)
from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from train.competition_native_jax.train_jax import load_tree  # noqa: E402

CANDIDATES = {
    "obs_v2_s1": REPO / "baselines/obs_v2_s1/main.py",
    "obs_v2_s2": REPO / "baselines/obs_v2_s2/main.py",
}
CHECKPOINTS = {
    "obs_v2_s1": REPO / "experiments/marathon/obs_v2_r1/OBS-V2-R1-S1/raw.npz",
    "obs_v2_s2": REPO / "experiments/marathon/obs_v2_r1/OBS-V2-R1-S2/raw.npz",
}
BASELINE_NPZ = (
    Path.home()
    / "quantsilico-runtime/cloud_assisted_deadline_salvage_v1_final"
    / "ckpt_final_u482_t7593984/raw.npz"
)

HANDSHAKE = "0 3 3\n"
OBS = (
    "1 1 5 1 3\n"
    "4 1 1\n1 2 1\n1 1 1\n"
    "1 1 0\n0 0 0\n0 0 2\n"
    "5 2 0\n0 0 0\n0 0 3\n"
)


def surgery_template() -> dict:
    """Reproduce the runner's DECLARED OBS-V2 warm-start shape surgery."""
    v2_like = init_params(
        jax.random.PRNGKey(0),
        spatial_planes=N_SPATIAL_V2,
        global_dim=N_GLOBAL_V2,
    )
    base = load_tree(BASELINE_NPZ, init_params(jax.random.PRNGKey(0)))
    return {
        **base,
        "patch_proj": jnp.concatenate(
            [base["patch_proj"], v2_like["patch_proj"][base["patch_proj"].shape[0]:]],
            axis=0,
        ),
        "global_proj": jnp.concatenate(
            [base["global_proj"], v2_like["global_proj"][base["global_proj"].shape[0]:]],
            axis=0,
        ),
    }


def logits_for(params: dict) -> np.ndarray:
    spatial = np.zeros((N_SPATIAL_V2, MAX_HW, MAX_HW), dtype=np.float32)
    spatial[0, 10, 10] = 1.0
    spatial[1, 10, 11] = 1.0
    global_vec = np.zeros(N_GLOBAL_V2, dtype=np.float32)
    mask = np.ones(ACTION_DIM, dtype=np.float32)
    _idx, _logp, out = infer(params, jnp.asarray(spatial), jnp.asarray(global_vec),
                             jnp.asarray(mask), None)
    return np.asarray(out["flat_logits"], dtype=np.float64).ravel()


def one_action(main_py: Path) -> str | None:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{REPO}{os.pathsep}{REPO / 'src'}{os.pathsep}{REPO / 'third_party' / 'generals-bots'}"
    )
    proc = subprocess.Popen(
        [sys.executable, "-u", str(main_py)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(main_py.parent),
    )
    out, err = proc.communicate(HANDSHAKE + OBS, timeout=300)
    lines = [line for line in (out or "").strip().splitlines() if line.strip()]
    if not lines:
        print(f"[probe] no action from {main_py.parent.name}; stderr tail:", file=sys.stderr)
        print((err or "")[-800:], file=sys.stderr)
    return lines[0] if lines else None


def main() -> int:
    report: dict = {"l1": {}, "l2": {}}
    ok = True

    template = init_params(
        jax.random.PRNGKey(0),
        spatial_planes=N_SPATIAL_V2,
        global_dim=N_GLOBAL_V2,
    )
    surgery = logits_for(surgery_template())
    loaded = {}
    for name, npz in CHECKPOINTS.items():
        if not npz.is_file():
            print(f"[probe] missing checkpoint: {npz}", file=sys.stderr)
            return 2
        params = load_tree(npz, template)
        loaded[name] = logits_for(params)
        vs_surgery = float(np.max(np.abs(loaded[name] - surgery)))
        report["l1"][name] = {
            "max_abs_diff_vs_surgery_warm_start": vs_surgery,
            "weights_load_and_matter": vs_surgery > 1e-6,
        }
        ok = ok and vs_surgery > 1e-6
    distinct = float(np.max(np.abs(loaded["obs_v2_s1"] - loaded["obs_v2_s2"])))
    report["l1"]["checkpoints_distinct"] = {"max_abs_diff": distinct,
                                            "distinct": distinct > 1e-6}
    ok = ok and distinct > 1e-6

    for name, path in CANDIDATES.items():
        runs = [one_action(path) for _ in range(2)]
        well_formed = all(
            action is not None and len(action.split()) == 5 for action in runs
        )
        deterministic = runs[0] == runs[1]
        report["l2"][name] = {"actions": runs, "well_formed": well_formed,
                              "deterministic": deterministic}
        ok = ok and well_formed and deterministic

    report["verdict"] = "PASS" if ok else "DEFECT_FOUND"
    print(json.dumps(report, indent=1))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
