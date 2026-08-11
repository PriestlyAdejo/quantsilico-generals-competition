#!/usr/bin/env bash
# Resume hot-path remediation after profile already completed (tests → v3 → smoke gate).
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

echo "=== Impacted pytest (resume) ==="
set +e
pytest -q tests/competition_native_jax -o cache_dir=/tmp/qs_pytest_cache 2>&1 | tee experiments/manifests/_hotpath_pytest.log
PYTEST_RC=${PIPESTATUS[0]}
set -e
echo "PYTEST_RC=${PYTEST_RC}"

echo "=== Throughput ladder V3 ==="
python -u -m train.competition_native_jax.train_jax \
  --mode throughput_v3 \
  --out experiments/competition_native_jax/throughput_v3 \
  --manifest experiments/manifests/competition_native_jax_throughput_ladder_v3.json

python - <<'PY'
import json, math, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

m = json.loads(Path("experiments/manifests/competition_native_jax_throughput_ladder_v3.json").read_text())
cfg = m.get("frozen_config") or {}
tps = float(m.get("valid_learning_tps") or 0)
eligible = bool(m.get("meaningful_short_eligible"))
pytest_log = Path("experiments/manifests/_hotpath_pytest.log").read_text(encoding="utf-8", errors="replace")
# Prefer explicit summary line
tests_ok = ("failed" not in pytest_log.lower().split("short test summary")[0] if False else True)
tests_ok = ("FAILED" not in pytest_log) and ("ERROR" not in pytest_log.split("FAILURES")[0] if "FAILURES" in pytest_log else "ERROR" not in pytest_log)
# Also require trailing passed summary without failed
import re
m_sum = re.search(r"(\d+) failed", pytest_log)
tests_ok = not (m_sum and int(m_sum.group(1)) > 0)
if re.search(r"(\d+) error", pytest_log, re.I):
    tests_ok = False

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
elif tps < gate["min_tps_for_90m_short"]:
    disposition = "BLOCKED_COMPUTE_HOST_ENV" if tps < 5.0 else "BATCHED_HOST_PATH_PROMOTION_READY"
else:
    disposition = "END_TO_END_JAX_PROMOTION_READY" if m.get("rollout_architecture") == "END_TO_END_OFFICIAL_JAX_ROLLOUT" else "HOT_PATH_REMEDIATION_PASSED"
gate["disposition"] = disposition
Path("experiments/manifests/competition_native_jax_meaningful_training_promotion_gate.json").write_text(
    json.dumps(gate, indent=2) + "\n", encoding="utf-8"
)
Path("experiments/competition_native_jax/frozen_train_config_v3.json").write_text(
    json.dumps(cfg, indent=2) + "\n", encoding="utf-8"
)
print("GATE", json.dumps(gate), flush=True)

smoke = None
smoke_ok = False
# Only smoke when tests pass AND TPS clears meaningful gate (or strong improvement warranting validation)
if tests_ok and eligible:
    ne = int(cfg.get("num_envs") or 8)
    rl = int(cfg.get("rollout_len") or 16)
    print(f"=== Post-remediation smoke envs={ne} rollout={rl} ===", flush=True)
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
        smoke_ok = smoke.get("status") == "COMPLETED"
    print("SMOKE_RC", rc, smoke_ok, flush=True)
else:
    print(f"=== Smoke SKIPPED tests_ok={tests_ok} eligible={eligible} tps={tps:.4f} ===", flush=True)

now = datetime.now(timezone.utc).isoformat()
# Update profile notes if present
prof_path = Path("experiments/manifests/competition_native_jax_hot_path_profile.json")
if prof_path.exists():
    prof = json.loads(prof_path.read_text())
    prof["notes"] = [
        "Shares extrapolated from microbenchmarks onto full_rollout_8steps wall time",
        "Env step, get_observation, legal mask, and policy forward are jax.vmap+jit batched",
        "Complete-loop valid_learning_tps measured in throughput_ladder_v3",
    ]
    # Recompute residual share
    times = prof.get("times_s") or {}
    total = float(times.get("full_rollout_8steps_s") or 1)
    accounted = (
        float(times.get("env_step_batch_20_s", 0)) / 20 * 8
        + float(times.get("obs_mask_batch_10_s", 0)) / 10 * 8
        + float(times.get("policy_forward_batch_20_s", 0)) / 20 * 8
    )
    prof["approximate_shares_pct"]["unaccounted_or_other_pct"] = max(0.0, 100.0 * (1.0 - accounted / total))
    prof_path.write_text(json.dumps(prof, indent=2) + "\n", encoding="utf-8")

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
    "primary_blocker": None if "READY" in disposition or "PASSED" in disposition else disposition,
    "throughput_v3_tps": tps,
    "rollout_architecture": m.get("rollout_architecture"),
    "smoke_v3_ok": smoke_ok,
}
Path("experiments/manifests/competition_native_jax_programme_state.json").write_text(
    json.dumps(state, indent=2) + "\n", encoding="utf-8"
)

summary_path = Path("experiments/manifests/competition_native_jax_daytime_training_summary.json")
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"schema_version": 1}
summary["status"] = "AWAITING_HOT_PATH_REMEDIATION_REVIEW"
summary["throughput_v3"] = m
summary["promotion_gate"] = gate
summary["smoke_v3"] = smoke
summary["remediation_completed_at"] = now
summary["classifications"] = [
    "SHORT_COMPLETED_RESEARCH_ONLY",
    "MEDIUM_ABORTED_BY_OPERATOR_PERFORMANCE_GATE",
    disposition,
    "NO_MEANINGFUL_TRAINED_CHECKPOINT",
]
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

Path("experiments/manifests/competition_native_jax_final_recommendation.json").write_text(
    json.dumps({
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_FINAL_RECOMMENDATION",
        "recommendation": "NO_CANDIDATE_CURRENTLY_RECOMMENDED",
        "overnight_parent": "NO_VALID_OVERNIGHT_PARENT",
        "disposition": disposition,
        "updated_at": now,
    }, indent=2) + "\n",
    encoding="utf-8",
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
Path("experiments/manifests/competition_native_jax_owned_processes.json").write_text(
    json.dumps({
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_OWNED_PROCESSES",
        "updated_at": now,
        "jobs": [],
        "note": "Cleared after hot-path remediation hard-stop",
    }, indent=2) + "\n",
    encoding="utf-8",
)
print("FINAL_STATE", state["hard_stop_status"], disposition, "tps", tps, flush=True)
PY

echo "=== Resume remediation complete ==="
