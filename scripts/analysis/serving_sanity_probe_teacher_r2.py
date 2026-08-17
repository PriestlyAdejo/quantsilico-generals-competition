"""Pre-game serving sanity probe for the STAGE5 TEACHER-R2 gameplay arbiter.

EV-0034 precedent + scale/rwb1/t2 probe pattern. Two layers, local/CPU,
deterministic:

L1 PARAMETER EFFECT (canonical legal-obs path): the loaded teacher BC
   checkpoint must change inference outputs vs freshly-initialised
   parameters (small_policy over a fixed spatial/global input).
L2 PROTOCOL BEHAVIOUR: the candidate main.py is driven over the official
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
for entry in (REPO, REPO / "src", REPO / "third_party" / "generals-bots"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from baselines.teacher_r2_serving_common import load_params  # noqa: E402
from scripts.training.bc_a_train_pilot import ACTION_DIM, init_params, small_policy  # noqa: E402

CANDIDATE = REPO / "baselines/teacher_r2_bc_s1/main.py"

HANDSHAKE = "0 3 3\n"
OBS = (
    "1 1 5 1 3\n"
    "4 1 1\n1 2 1\n1 1 1\n"
    "1 1 0\n0 0 0\n0 0 2\n"
    "5 2 0\n0 0 0\n0 0 3\n"
)


def logits_for(params: dict) -> np.ndarray:
    spatial = np.zeros((1, 8, 21, 21), dtype=np.float32)
    spatial[0, 0, 10, 10] = 1.0
    spatial[0, 1, 10, 11] = 1.0
    global_vec = np.zeros((1, 8), dtype=np.float32)
    logits = small_policy(params, jnp.asarray(spatial), jnp.asarray(global_vec))
    return np.asarray(logits, dtype=np.float64).ravel()


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

    fresh = logits_for(init_params(jax.random.PRNGKey(1)))
    loaded = logits_for(load_params())
    assert loaded.shape == (ACTION_DIM,), loaded.shape
    differs = float(np.max(np.abs(loaded - fresh)))
    report["l1"]["teacher_r2_bc_s1"] = {"max_abs_diff_vs_fresh": differs,
                                        "weights_load_and_matter": differs > 1e-6}
    ok = ok and differs > 1e-6

    runs = [one_action(CANDIDATE) for _ in range(2)]
    well_formed = all(action is not None and len(action.split()) == 5 for action in runs)
    deterministic = runs[0] == runs[1]
    report["l2"]["teacher_r2_bc_s1"] = {"actions": runs, "well_formed": well_formed,
                                        "deterministic": deterministic}
    ok = ok and well_formed and deterministic

    report["verdict"] = "PASS" if ok else "DEFECT_FOUND"
    print(json.dumps(report, indent=1))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
