"""Pre-game serving sanity probe for the STAGE5 SCALE-R1 gameplay arbiter.

EV-0034 precedent + rwb1/t2 probe pattern. Two layers, local/CPU,
deterministic:

L1 PARAMETER EFFECT (canonical 8-plane path): each loaded terminal
   checkpoint must change inference outputs vs freshly-initialised
   parameters, and the two checkpoints must produce distinct outputs on
   identical inputs.
L2 PROTOCOL BEHAVIOUR: each candidate main.py is driven over the official
   stdio protocol twice; actions must be well-formed 5-field competition
   actions and deterministic across runs.

Must PASS before any gameplay games are counted. Exit 0 = PASS, 2 = DEFECT.
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
from generals_bot.competition_native_jax.obs_memory import N_GLOBAL, N_SPATIAL  # noqa: E402
from generals_bot.competition_native_jax.transformer_jax import init_params  # noqa: E402
from train.competition_native_jax.train_jax import load_tree  # noqa: E402

CANDIDATES = {
    "scale_b0_8m_s3": REPO / "baselines/scale_a0_8m_s1/main.py",
    "scale_b0_8m_s4": REPO / "baselines/scale_a0_8m_s2/main.py",
}
CHECKPOINTS = {
    "scale_b0_8m_s3": REPO
    / "experiments/marathon/screening_runs/STAGE5-SCALE-R2/SCALE-B0-8M-S3/raw.npz",
    "scale_b0_8m_s4": REPO
    / "experiments/marathon/screening_runs/STAGE5-SCALE-R2/SCALE-B0-8M-S4/raw.npz",
    "scale_b1_16m_s1": REPO
    / "experiments/marathon/screening_runs/STAGE5-SCALE-R2/SCALE-B1-16M-S1/raw.npz",
}

HANDSHAKE = "0 3 3\n"
OBS = (
    "1 1 5 1 3\n"
    "4 1 1\n1 2 1\n1 1 1\n"
    "1 1 0\n0 0 0\n0 0 2\n"
    "5 2 0\n0 0 0\n0 0 3\n"
)


def logits_for(params: dict) -> np.ndarray:
    spatial = np.zeros((N_SPATIAL, MAX_HW, MAX_HW), dtype=np.float32)
    spatial[0, 10, 10] = 1.0
    spatial[1, 10, 11] = 1.0
    global_vec = np.zeros(N_GLOBAL, dtype=np.float32)
    mask = np.ones(ACTION_DIM, dtype=np.float32)
    _idx, _logp, out = infer(params, jnp.asarray(spatial), jnp.asarray(global_vec),
                             jnp.asarray(mask), None)
    return np.asarray(out["flat_logits"], dtype=np.float64).ravel()


def one_action(main_py: Path) -> str | None:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}{os.pathsep}{REPO / 'src'}"
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

    template = init_params(jax.random.PRNGKey(0))
    fresh = logits_for(template)
    loaded = {}
    for name, npz in CHECKPOINTS.items():
        params = load_tree(npz, template)
        loaded[name] = logits_for(params)
        differs = float(np.max(np.abs(loaded[name] - fresh)))
        report["l1"][name] = {"max_abs_diff_vs_fresh": differs,
                              "weights_load_and_matter": differs > 1e-6}
        ok = ok and differs > 1e-6
    distinct = min(
        float(np.max(np.abs(loaded["scale_b0_8m_s3"] - loaded["scale_b0_8m_s4"]))),
        float(np.max(np.abs(loaded["scale_b0_8m_s3"] - loaded["scale_b1_16m_s1"]))),
        float(np.max(np.abs(loaded["scale_b0_8m_s4"] - loaded["scale_b1_16m_s1"]))),
    )
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
