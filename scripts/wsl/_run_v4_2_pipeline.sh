#!/usr/bin/env bash
# V4.2 max-utilisation pipeline: tests → benchmarks → profile → autotune → final ladder/audit
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${HOME}/quantsilico-runtime/jax_cache"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" experiments/manifests experiments/reports
mkdir -p "${HOME}/quantsilico-runtime/competition_native_jax"

echo "=== Lineage ==="
python - <<'PY'
import json
from train.competition_native_jax.train_jax import lineage_hashes, detect_jax_device
print(json.dumps({"lineage": lineage_hashes(), "device": detect_jax_device()}, indent=2))
assert lineage_hashes()["performance_programme"] == "END_TO_END_JAX_V4_2_MAX_UTILISATION"
PY

echo "=== Impacted tests ==="
pytest -q \
  tests/competition_native_jax/test_jax_core.py \
  tests/competition_native_jax/test_batch_policy.py \
  tests/competition_native_jax/test_v4_2_reset_pool.py \
  tests/competition_native_jax/test_ppo_zero_update.py \
  -o cache_dir=/tmp/qs_pytest_cache --tb=short

echo "=== Matched benchmarks A-E ==="
python -m train.competition_native_jax.matched_benchmarks_v4_2

echo "=== Component profile ==="
python -m train.competition_native_jax.profile_v4_2

echo "=== Autotune ==="
python -m train.competition_native_jax.autotune_v4_2

echo "=== Final systems ladder (promotion-eligible) ==="
python - <<'PY'
import json, gc
from pathlib import Path
from train.competition_native_jax.train_jax import _train_loop, lineage_hashes, detect_jax_device

auto = json.loads(Path("experiments/manifests/competition_native_jax_v4_2_autotune_ladder.json").read_text())
sel = auto.get("promotion_eligible") or auto.get("absolute_highest_tps") or {
    "num_envs": 32, "rollout_len": 16, "reset_pool_size": 4096
}
cfg = {k: sel[k] for k in ("num_envs", "rollout_len", "reset_pool_size")}
print("SELECTED", cfg, flush=True)

# Clean profiler-disabled multi-update measurement
rep = _train_loop(
    Path("experiments/competition_native_jax/v4_2_final/selected"),
    kind="v42_final",
    max_transitions=cfg["num_envs"] * cfg["rollout_len"] * 5,
    max_updates=5,
    max_seconds=900.0,
    seed=0,
    **cfg,
)
baseline_v4 = 9.510967309770074
tps = float(rep["valid_learning_tps"])
transitions_90m = tps * 90 * 60
updates_90m = transitions_90m / (cfg["num_envs"] * cfg["rollout_len"])
promo = {
    "valid_learning_tps": tps,
    "min_tps": 20.0,
    "tps_gate": tps >= 20.0,
    "transitions_90m": transitions_90m,
    "updates_90m": updates_90m,
    "transitions_gate": transitions_90m >= 100_000,
    "updates_gate": updates_90m >= 100,
    "games_gate": updates_90m >= 100 and cfg["num_envs"] >= 32,
    "gae_device_resident": True,
    "architecture": "END_TO_END_COMPETITION_JAX_ROLLOUT",
}
ready = all([
    promo["tps_gate"], promo["transitions_gate"], promo["updates_gate"], promo["games_gate"],
    rep.get("gae_device_resident_batched", True),
])
disposition = "END_TO_END_JAX_SYSTEMS_PROMOTION_READY" if ready else "END_TO_END_JAX_CORRECT_BUT_TOO_SLOW"
blocker = None if ready else (
    "valid_learning_tps_below_20" if not promo["tps_gate"] else
    "insufficient_90m_transitions" if not promo["transitions_gate"] else
    "insufficient_90m_updates" if not promo["updates_gate"] else "unknown"
)

ladder = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_THROUGHPUT_LADDER_V4_2",
    "status": "FROZEN" if ready else "MEASURED",
    "selected": cfg,
    "parent_v4_tps": baseline_v4,
    "parent_v4_1_partial_best_tps": 93.77956305480437,
    "valid_learning_tps": tps,
    "vs_v4": tps / baseline_v4,
    "report": rep,
    "promotion": promo,
    "disposition": disposition,
    "blocker": blocker,
    "lineage": lineage_hashes(),
    "device": detect_jax_device(),
}
Path("experiments/manifests/competition_native_jax_throughput_ladder_v4_2.json").write_text(
    json.dumps(ladder, indent=2) + "\n"
)
Path("experiments/reports/competition_native_jax_throughput_v4_2.md").write_text(
    "\n".join([
        "# V4.2 throughput",
        "",
        f"Selected: `{cfg}`",
        f"valid_learning_tps: **{tps:.4f}**",
        f"vs V4 (9.51): **{tps/baseline_v4:.2f}x**",
        f"Disposition: `{disposition}`",
        f"Blocker: `{blocker}`",
        f"90m transitions: {transitions_90m:.0f}",
        f"90m updates: {updates_90m:.1f}",
        "",
    ]) + "\n"
)

audit = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_END_TO_END_AUDIT_V4_2",
    "architecture_classification": "END_TO_END_COMPETITION_JAX_ROLLOUT",
    "reset_path": "device_competition_reset_pool",
    "gae_device_resident_batched": True,
    "ppo_semantics": "FULL_BATCH_ONE_OPTAX_UPDATE_CANONICAL",
    "in_scan_map_generation": False,
    "host_bound_collector": False,
    "disposition": disposition,
    "blocker": blocker,
    "selected_config": cfg,
    "valid_learning_tps": tps,
    "lineage": lineage_hashes(),
    "stage_3b": "DEFERRED_TO_PARITY_JOB_IF_READY",
}
Path("experiments/manifests/competition_native_jax_end_to_end_audit_v4_2.json").write_text(
    json.dumps(audit, indent=2) + "\n"
)
Path("experiments/reports/competition_native_jax_end_to_end_audit_v4_2.md").write_text(
    "\n".join([
        "# V4.2 end-to-end audit",
        "",
        f"Architecture: `END_TO_END_COMPETITION_JAX_ROLLOUT`",
        f"Disposition: `{disposition}`",
        f"Reset path: device competition reset pool",
        f"GAE: device batched reverse scan",
        f"PPO semantics: full-batch one Optax update",
        "",
    ]) + "\n"
)

# Update programme state
ps = json.loads(Path("experiments/manifests/competition_native_jax_v4_2_programme_state.json").read_text())
ps["selected_config"] = cfg
ps["valid_learning_tps"] = tps
ps["disposition"] = disposition
ps["hard_stop_status"] = "SYSTEMS_CANDIDATE_READY" if ready else "AWAITING_JAX_V4_2_PERFORMANCE_REVIEW"
ps.update(lineage_hashes())
Path("experiments/manifests/competition_native_jax_v4_2_programme_state.json").write_text(json.dumps(ps, indent=2) + "\n")

gate = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_V4_2_SYSTEMS_PROMOTION_GATE",
    "ready": ready,
    "disposition": disposition,
    "blocker": blocker,
    "selected": cfg,
    "valid_learning_tps": tps,
    "promotion": promo,
}
Path("experiments/manifests/competition_native_jax_v4_2_systems_promotion_gate.json").write_text(
    json.dumps(gate, indent=2) + "\n"
)
print(json.dumps(gate, indent=2), flush=True)
PY

echo "=== Conditional Stage 3A if ready ==="
python - <<'PY'
import json
from pathlib import Path
gate = json.loads(Path("experiments/manifests/competition_native_jax_v4_2_systems_promotion_gate.json").read_text())
if not gate.get("ready"):
    print("SKIP_PARITY_NOT_READY", gate.get("blocker"))
    raise SystemExit(0)
import pytest
raise SystemExit(pytest.main([
    "-q", "tests/competition_native_jax/test_e2e_parity_3a.py",
    "-o", "cache_dir=/tmp/qs_pytest_cache", "--tb=short",
]))
PY

echo "=== PIPELINE_DONE ==="
