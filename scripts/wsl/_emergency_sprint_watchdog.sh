#!/usr/bin/env bash
# Sprint watchdog: enforce experiments_stop / qualification / final selection.
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export PYTHONUNBUFFERED=1

LOG="experiments/logs/owned_jobs/emergency_watchdog.out.log"
exec >> "${LOG}" 2>&1
echo "WATCHDOG_START $(date -u -Iseconds)"

python - <<'PY'
import json, os, signal, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(".").resolve()
deadlines = json.loads(Path("experiments/manifests/emergency_deadlines.json").read_text())
stop_at = datetime.fromisoformat(deadlines["experiments_stop_at"])
qual_at = datetime.fromisoformat(deadlines["qualification_only_at"])
end_at = datetime.fromisoformat(deadlines["sprint_end_at"])
mono_stop = float(deadlines.get("experiments_stop_monotonic_deadline_s") or 0)

def now():
    return datetime.now(timezone.utc)

print("deadlines", deadlines["experiments_stop_at"], deadlines["qualification_only_at"])

# Wait until experiments stop
while True:
    if now() >= stop_at:
        break
    if mono_stop and time.monotonic() >= mono_stop:
        break
    # heartbeat
    Path("experiments/logs/owned_jobs/emergency_watchdog.heartbeat").write_text(now().isoformat()+"\n")
    time.sleep(30)

print("EXPERIMENTS_STOP", now().isoformat())
pid_path = Path("experiments/logs/owned_jobs/emergency_ppo.pid")
if pid_path.exists():
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, signal.SIGINT)
        print("SIGINT", pid)
        # wait up to 10 min for clean exit
        for _ in range(120):
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(5)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
                print("SIGKILL", pid)
            except OSError:
                pass
    except Exception as e:
        print("stop_error", e)

# touch STOP_REQUEST as well
(Path.home() / "quantsilico-runtime/emergency_rolling_v1/training/STOP_REQUEST").write_text("1\n")
Path("experiments/competition_native_jax/emergency_rolling_v1/STOP_REQUEST").parent.mkdir(parents=True, exist_ok=True)
Path("experiments/competition_native_jax/emergency_rolling_v1/STOP_REQUEST").write_text("1\n")

# Controls: skip unless learned package + >3h remain (at stop time, remaining to end is 2h → skip)
ctrl = {
  "schema_version": 1,
  "kind": "EMERGENCY_CONTROLS_DECISION",
  "status": "SKIP_CONTROLS",
  "reason": "At experiments_stop_at remaining package reserve is 2h; controls require >3h remaining.",
  "updated_at": now().isoformat(),
}
Path("experiments/manifests/emergency_controls_decision.json").write_text(json.dumps(ctrl, indent=2)+"\n")

# Wait for qualification-only window
while now() < qual_at:
    time.sleep(15)

print("QUALIFICATION_ONLY", now().isoformat())
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
import subprocess
subprocess.check_call(["python", "scripts/emergency_final_selection_gate.py"])

# Clear owned processes
owned = {
  "schema_version": 1,
  "kind": "COMPETITION_NATIVE_JAX_OWNED_PROCESSES",
  "updated_at": now().isoformat(),
  "jobs": [],
  "note": "Cleared by emergency watchdog at hard stop",
}
Path("experiments/manifests/competition_native_jax_owned_processes.json").write_text(json.dumps(owned, indent=2)+"\n")

prog_path = Path("experiments/manifests/emergency_rolling_programme_state.json")
prog = json.loads(prog_path.read_text())
prog["status"] = "HARD_STOP_COMPLETE"
prog["hard_stop_at"] = now().isoformat()
prog["updated_at"] = now().isoformat()
tmp = prog_path.with_suffix(".tmp")
tmp.write_text(json.dumps(prog, indent=2)+"\n")
tmp.replace(prog_path)
print("HARD_STOP_COMPLETE", now().isoformat())
PY
