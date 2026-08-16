"""Pre-game serving sanity probe for the RWB1 gameplay arbiter (EV-0034 precedent).

Drives each candidate main.py over the official stdio protocol twice and
checks: actions are well-formed 5-field competition actions and deterministic
across runs. Must PASS before any gameplay games are counted.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CANDIDATES = {
    "rwb1_control_s1": REPO / "baselines/rwb1_a0_control_s1/main.py",
    "rwb1_control_s2": REPO / "baselines/rwb1_a0_control_s2/main.py",
    "rwb1_land_s1": REPO / "baselines/rwb1_a1_land_s1/main.py",
    "rwb1_land_s2": REPO / "baselines/rwb1_a1_land_s2/main.py",
}

# Identical protocol payload to the EV-0034 probe (scripts/analysis/serving_sanity_probe.py)
HANDSHAKE = "0 3 3\n"
OBS = (
    "1 1 5 1 3\n"
    "4 1 1\n1 2 1\n1 1 1\n"
    "1 1 0\n0 0 0\n0 0 2\n"
    "5 2 0\n0 0 0\n0 0 3\n"
)


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
    out, err = proc.communicate(HANDSHAKE + OBS, timeout=180)
    lines = [line for line in (out or "").strip().splitlines() if line.strip()]
    if not lines:
        print(f"[probe] no action from {main_py.parent.name}; stderr tail:", file=sys.stderr)
        print((err or "")[-800:], file=sys.stderr)
    return lines[0] if lines else None


def main() -> int:
    report = {}
    ok = True
    for name, path in CANDIDATES.items():
        runs = [one_action(path) for _ in range(2)]
        well_formed = all(
            action is not None and len(action.split()) == 5 for action in runs
        )
        deterministic = runs[0] == runs[1]
        report[name] = {"actions": runs, "well_formed": well_formed,
                        "deterministic": deterministic}
        ok = ok and well_formed and deterministic
    report["verdict"] = "PASS" if ok else "DEFECT_FOUND"
    print(json.dumps(report, indent=1))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
