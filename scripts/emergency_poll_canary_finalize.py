"""Poll canary completion then write dashboard/selection artefacts (no bash)."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    canary = ROOT / "experiments/manifests/emergency_bootstrap_canary.json"
    for i in range(70):
        if canary.exists():
            print("CANARY_READY", flush=True)
            break
        # still running?
        try:
            out = subprocess.check_output(
                ["wsl.exe", "-d", "Ubuntu", "--", "pgrep", "-f", "emergency_bootstrap_canary.py"],
                text=True,
            )
            if not out.strip():
                print("CANARY_PROC_GONE", flush=True)
                break
        except subprocess.CalledProcessError:
            print("CANARY_PROC_GONE", flush=True)
            break
        print(f"waiting_canary_{i}", flush=True)
        time.sleep(30)
    else:
        print("CANARY_TIMEOUT", flush=True)

    scripts = [
        "scripts/emergency_write_dashboard_gate.py",
        "scripts/emergency_final_selection_gate.py",
        "scripts/emergency_patch_programme_state.py",
    ]
    for s in scripts:
        cmd = [
            "wsl.exe",
            "-d",
            "Ubuntu",
            "--",
            "bash",
            "-lc",
            f"cd /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition && "
            f"source ~/.venvs/quantsilico-jax-gpu/bin/activate && "
            f"export PYTHONPATH=src:. && export CUDA_VISIBLE_DEVICES='' && export JAX_PLATFORMS=cpu && "
            f"python {s}",
        ]
        print("RUN", s, flush=True)
        subprocess.check_call(cmd)
    print("FINISH_WAIT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
