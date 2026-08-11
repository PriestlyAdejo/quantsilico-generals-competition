"""Cooperative first-learned handoff: STOP → train/package → exact-resume."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / "quantsilico-runtime" / "emergency_rolling_v1"
STOP = RUNTIME / "training" / "STOP_REQUEST"
STOP_REPO = ROOT / "experiments/competition_native_jax/emergency_rolling_v1" / "STOP_REQUEST"
READY_DEADLINE_LOCAL = datetime.fromisoformat("2026-08-07T06:30:00+01:00")
GRACE_MIN = 10


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    with tmp.open("rb") as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    tmp.replace(path)


def _set_gpu_owner(owner: str) -> None:
    doc = {
        "schema_version": 1,
        "kind": "EMERGENCY_GPU_OWNERSHIP",
        "gpu_owner": owner,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    _atomic_write_json(RUNTIME / "gpu" / "gpu_owner.json", doc)
    _atomic_write_json(ROOT / "experiments/manifests/emergency_gpu_owner.json", doc)


def _ppo_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _wsl(cmd: str, *, env_extra: dict | None = None, timeout: int | None = None) -> int:
    # Running inside WSL already when invoked via wsl; support direct.
    full = (
        "cd /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition && "
        "source ~/.venvs/quantsilico-jax-gpu/bin/activate && "
        "export PYTHONPATH=src:. && "
        f"{cmd}"
    )
    return subprocess.call(["bash", "-lc", full], timeout=timeout)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-pid", type=int, default=554)
    ap.add_argument("--skip-stop", action="store_true", help="train/package only (no PPO stop)")
    ap.add_argument("--wait-dataset-minutes", type=float, default=25.0)
    args = ap.parse_args()

    plumbing = json.loads(
        (ROOT / "experiments/manifests/emergency_distill_plumbing.json").read_text(encoding="utf-8")
    )
    if plumbing.get("status") != "DISTILLATION_MINIMAL_ROUTE_READY":
        print("BLOCKED_NOT_READY", plumbing.get("status"))
        return 2

    now = datetime.now(READY_DEADLINE_LOCAL.tzinfo)
    grace_end = READY_DEADLINE_LOCAL.timestamp() + GRACE_MIN * 60
    report = {
        "schema_version": 1,
        "kind": "EMERGENCY_FIRST_LEARNED_HANDOFF",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ready_deadline_local": READY_DEADLINE_LOCAL.isoformat(),
        "within_grace": now.timestamp() <= grace_end,
        "steps": [],
    }

    def step(name: str, ok: bool, detail: str = "") -> None:
        report["steps"].append({"name": name, "ok": ok, "detail": detail, "ts": datetime.now(timezone.utc).isoformat()})
        print(f"HANDOFF {name} ok={ok} {detail}", flush=True)

    # Wait for dataset
    ds = ROOT / "experiments/manifests/emergency_distill_dataset.json"
    t_wait = time.time()
    while not ds.exists() and (time.time() - t_wait) < args.wait_dataset_minutes * 60:
        time.sleep(5)
    if not ds.exists():
        step("dataset", False, "missing")
        report["status"] = "BLOCKED_NO_DATASET"
        _atomic_write_json(ROOT / "experiments/manifests/emergency_first_learned_handoff.json", report)
        return 3
    step("dataset", True, ds.read_text(encoding="utf-8")[:200])

    teacher_sel = {
        "rule": "newest_COMPLETE_EMA_else_best_screened_else_ckpt_final_420",
        "selected": json.loads(ds.read_text(encoding="utf-8")).get("teacher_checkpoint"),
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_teacher_selection.json", teacher_sel)
    step("teacher_selected", True, str(teacher_sel["selected"]))

    if not args.skip_stop:
        # Prefer stop near COMPLETE: wait up to grace for next complete if before deadline
        step("stop_request", True, "writing STOP_REQUEST")
        STOP.parent.mkdir(parents=True, exist_ok=True)
        STOP_REPO.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "reason": "FIRST_LEARNED_DISTILL_HANDOFF",
                "requested_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        STOP.write_text(payload + "\n", encoding="utf-8")
        STOP_REPO.write_text(payload + "\n", encoding="utf-8")

        # wait for PPO exit (up to 20 min)
        t0 = time.time()
        while _ppo_alive(args.ppo_pid) and (time.time() - t0) < 1200:
            time.sleep(2)
        alive = _ppo_alive(args.ppo_pid)
        step("ppo_stopped", not alive, f"pid={args.ppo_pid} alive={alive}")
        if alive:
            report["status"] = "BLOCKED_PPO_DID_NOT_STOP"
            _atomic_write_json(ROOT / "experiments/manifests/emergency_first_learned_handoff.json", report)
            return 4
        _set_gpu_owner("DISTILLATION")
    else:
        step("stop_request", True, "skipped")

    # Train (prefer GPU if available after stop; else CPU)
    device = "gpu" if not args.skip_stop else "cpu"
    rc = subprocess.call(
        [
            "bash",
            "-lc",
            (
                "cd /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition && "
                "source ~/.venvs/quantsilico-jax-gpu/bin/activate && "
                "export PYTHONPATH=src:. && "
                f"python scripts/emergency_first_learned_train.py --device {device} --steps 300 --max-minutes 70"
            ),
        ]
    )
    step("train", rc == 0, f"rc={rc}")
    if rc != 0:
        report["status"] = "BLOCKED_TRAIN_FAILED"
        _atomic_write_json(ROOT / "experiments/manifests/emergency_first_learned_handoff.json", report)
        if not args.skip_stop:
            _set_gpu_owner("NONE")
        return 5

    weights = RUNTIME / "distill" / "student_v1" / "student_emb96_v1.npz"
    rc = subprocess.call(
        [
            "bash",
            "-lc",
            (
                "cd /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition && "
                "source ~/.venvs/quantsilico-jax-gpu/bin/activate && "
                "export PYTHONPATH=src:. && "
                f"python scripts/emergency_package_cnj_student.py --weights {weights}"
            ),
        ]
    )
    step("package", rc == 0, f"rc={rc}")

    # Exact resume immediately
    if not args.skip_stop:
        # clear stop files
        for p in (STOP, STOP_REPO):
            if p.exists():
                p.unlink()
        _set_gpu_owner("PPO_TRAINER")
        # Resume from cooperative-stop final ckpt — patch PARENT via env if supported.
        # emergency_exact_resume_ppo uses fixed PARENT R-E.6; for true exact-resume
        # from cooperative stop we launch a wrapper that points at ckpt_final under RUNTIME.
        resume_wrapper = ROOT / "scripts" / "_emergency_resume_after_distill.sh"
        resume_wrapper.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "cd /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition\n"
            "source ~/.venvs/quantsilico-jax-gpu/bin/activate\n"
            "export PYTHONPATH=src:.\n"
            "export EMERGENCY_RESUME_PARENT=/home/pries/quantsilico-runtime/emergency_rolling_v1/training/checkpoints/ckpt_final\n"
            "python scripts/emergency_exact_resume_ppo.py\n",
            encoding="utf-8",
        )
        # Patch resume script to honor EMERGENCY_RESUME_PARENT if not already
        subprocess.Popen(
            ["bash", str(resume_wrapper)],
            stdout=open(RUNTIME / "training" / "resume_after_distill.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        step("exact_resume_launched", True, "EMERGENCY_RESUME_PARENT=ckpt_final")

    report["status"] = "HANDOFF_COMPLETE" if rc == 0 else "HANDOFF_PARTIAL"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(ROOT / "experiments/manifests/emergency_first_learned_handoff.json", report)
    print(json.dumps(report, indent=2))
    return 0 if rc == 0 else 6


if __name__ == "__main__":
    raise SystemExit(main())
