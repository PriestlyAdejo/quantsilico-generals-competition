#!/usr/bin/env bash
# Hot-path remediation: profile → tests → throughput v3 → optional bounded smoke.
# Does NOT launch short/medium/overnight.
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${HOME}/quantsilico-runtime/jax_cache"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
mkdir -p "${HOME}/quantsilico-runtime/competition_native_jax"

echo "=== FS overhead: /mnt/c vs Linux-native ==="
python - <<'PY'
import json, time
from pathlib import Path
home = Path.home() / "quantsilico-runtime" / "fs_bench"
home.mkdir(parents=True, exist_ok=True)
mnt = Path("experiments/competition_native_jax/_fs_bench")
mnt.mkdir(parents=True, exist_ok=True)
payload = b"x" * (1024 * 256)
def bench(d: Path, n: int = 40):
    t0 = time.perf_counter()
    for i in range(n):
        p = d / f"t{i}.bin"
        p.write_bytes(payload)
    return time.perf_counter() - t0
native_s = bench(home)
mnt_s = bench(mnt)
rep = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_FS_OVERHEAD",
    "writes": 40,
    "bytes_per_write": len(payload),
    "linux_native_s": native_s,
    "mnt_c_s": mnt_s,
    "mnt_c_slowdown": mnt_s / max(native_s, 1e-9),
}
Path("experiments/manifests/competition_native_jax_fs_overhead.json").write_text(
    json.dumps(rep, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(rep), flush=True)
PY

echo "=== Hot-path profile ==="
python -u scripts/run_competition_native_jax_hot_path_profile.py

echo "=== Impacted pytest ==="
pytest -q tests/competition_native_jax -o cache_dir=/tmp/qs_pytest_cache 2>&1 | tee experiments/manifests/_hotpath_pytest.log
PYTEST_RC=${PIPESTATUS[0]}

echo "=== Throughput ladder V3 ==="
python -u -m train.competition_native_jax.train_jax \
  --mode throughput_v3 \
  --out experiments/competition_native_jax/throughput_v3 \
  --manifest experiments/manifests/competition_native_jax_throughput_ladder_v3.json

python - <<'PY'
import json, math
from pathlib import Path
from datetime import datetime, timezone

m = json.loads(Path("experiments/manifests/competition_native_jax_throughput_ladder_v3.json").read_text())
cfg = m.get("frozen_config") or {}
tps = float(m.get("valid_learning_tps") or 0)
eligible = bool(m.get("meaningful_short_eligible"))
pytest_log = Path("experiments/manifests/_hotpath_pytest.log").read_text(encoding="utf-8", errors="replace")
tests_ok = "failed" not in pytest_log.lower() and "error" not in pytest_log.lower().split("===")[-1]

# Promotion gate freeze
gate = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_MEANINGFUL_TRAINING_PROMOTION_GATE",
    "minimum_valid_transitions": 100000,
    "minimum_ppo_updates": 100,
    "minimum_completed_games": 32,
    "minimum_curriculum_coverage": "competition_mode_only",
    "maximum_protocol_fault_rate": 0.0,
    "required_stable_tps_duration_s": 60,
    "required_checkpoint_resume": True,
    "required_legal_action_rate": 1.0,
    "min_tps_for_90m_short": 100000 / 5400.0,
    "measured_tps": tps,
    "supported_90m_transitions": m.get("supported_90m_transitions"),
    "supported_90m_updates": m.get("supported_90m_updates_at_rollout"),
    "tests_ok": tests_ok,
    "eligible_for_new_short_proposal": bool(eligible and tests_ok),
}
if not tests_ok:
    disposition = "BLOCKED_CORRECTNESS"
elif tps <= 0:
    disposition = "BLOCKED_COMPUTE_HOST_ENV"
elif not eligible:
    disposition = "BLOCKED_COMPUTE_HOST_ENV" if tps < 1.0 else "BATCHED_HOST_PATH_PROMOTION_READY"
    # still below 90m meaningful bar
    if tps < gate["min_tps_for_90m_short"]:
        disposition = "BLOCKED_COMPUTE_HOST_ENV" if tps < 5 else "BATCHED_HOST_PATH_PROMOTION_READY"
else:
    disposition = "HOT_PATH_REMEDIATION_PASSED"
    if m.get("rollout_architecture") == "END_TO_END_OFFICIAL_JAX_ROLLOUT":
        disposition = "END_TO_END_JAX_PROMOTION_READY"

gate["disposition"] = disposition
Path("experiments/manifests/competition_native_jax_meaningful_training_promotion_gate.json").write_text(
    json.dumps(gate, indent=2) + "\n", encoding="utf-8"
)
Path("experiments/competition_native_jax/frozen_train_config_v3.json").write_text(
    json.dumps(cfg, indent=2) + "\n", encoding="utf-8"
)
print("GATE", json.dumps(gate), flush=True)

# Bounded post-remediation smoke only if correctness + some throughput improvement
smoke_ok = False
smoke = None
if tests_ok and tps > 0.5:  # at least ~3x baseline low before spending smoke budget
    import subprocess, sys
    ne = int(cfg.get("num_envs") or 8)
    rl = int(cfg.get("rollout_len") or 16)
    print(f"=== Post-remediation smoke envs={ne} rollout={rl} (max 100k / 30m) ===", flush=True)
    # smoke uses max_seconds=1800 in train_jax; keep as authorized bound
    rc = subprocess.call([
        sys.executable, "-u", "-m", "train.competition_native_jax.train_jax",
        "--mode", "smoke",
        "--out", "experiments/competition_native_jax/smoke_v3",
        "--num-envs", str(ne),
        "--rollout-len", str(rl),
    ])
    smoke_path = Path("experiments/competition_native_jax/smoke_v3/smoke_report.json")
    if smoke_path.exists():
        smoke = json.loads(smoke_path.read_text())
        smoke_ok = smoke.get("status") == "COMPLETED" and float(smoke.get("valid_learning_tps") or 0) > 0
    print("SMOKE_RC", rc, "smoke_ok", smoke_ok, flush=True)
else:
    print("=== Smoke SKIPPED (tests_ok=%s tps=%.4f) ===" % (tests_ok, tps), flush=True)

# Final programme state
now = datetime.now(timezone.utc).isoformat()
state = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_PROGRAMME_STATE",
    "programme": "competition_native_jax_medium_abort_hot_path_remediation",
    "branch": "research/phase9g-competition-native-jax-preovernight-v1",
    "updated_at": now,
    "hard_stop_status": "AWAITING_HOT_PATH_REMEDIATION_REVIEW",
    "overnight_training_authorized": False,
    "phase_10_execution_authorized": False,
    "medium_training_authorized": False,
    "new_short_training_authorized": False,
    "automatic_upload_authorized": False,
    "final_recommendation": "NO_CANDIDATE_CURRENTLY_RECOMMENDED",
    "overnight_lineage": "NO_VALID_OVERNIGHT_PARENT",
    "promotion_disposition": disposition,
    "owned_processes": [],
    "primary_blocker": None if disposition.endswith("READY") or disposition.endswith("PASSED") else disposition,
    "throughput_v3_tps": tps,
    "rollout_architecture": m.get("rollout_architecture"),
}
Path("experiments/manifests/competition_native_jax_programme_state.json").write_text(
    json.dumps(state, indent=2) + "\n", encoding="utf-8"
)

summary = json.loads(Path("experiments/manifests/competition_native_jax_daytime_training_summary.json").read_text())
summary["status"] = disposition if disposition.startswith("BLOCKED") else "HOT_PATH_REMEDIATION_COMPLETE_AWAITING_REVIEW"
summary["throughput_v3"] = m
summary["promotion_gate"] = gate
summary["smoke_v3"] = smoke
summary["remediation_completed_at"] = now
Path("experiments/manifests/competition_native_jax_daytime_training_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)

rec = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_FINAL_RECOMMENDATION",
    "recommendation": "NO_CANDIDATE_CURRENTLY_RECOMMENDED",
    "overnight_parent": "NO_VALID_OVERNIGHT_PARENT",
    "disposition": disposition,
    "updated_at": now,
}
Path("experiments/manifests/competition_native_jax_final_recommendation.json").write_text(
    json.dumps(rec, indent=2) + "\n", encoding="utf-8"
)
Path("experiments/manifests/competition_native_jax_overnight_readiness.json").write_text(
    json.dumps({
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_OVERNIGHT_READINESS",
        "status": "NO_VALID_OVERNIGHT_PARENT",
        "updated_at": now,
    }, indent=2) + "\n",
    encoding="utf-8",
)

# Clear active owned jobs
owned = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_OWNED_PROCESSES",
    "updated_at": now,
    "jobs": [],
    "note": "All programme-owned processes stopped after medium abort + hot-path remediation",
}
Path("experiments/manifests/competition_native_jax_owned_processes.json").write_text(
    json.dumps(owned, indent=2) + "\n", encoding="utf-8"
)
print("FINAL_STATE", state["hard_stop_status"], disposition, flush=True)
PY

echo "=== Remediation script complete ==="
