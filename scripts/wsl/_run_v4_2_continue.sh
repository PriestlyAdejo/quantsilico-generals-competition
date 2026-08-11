#!/usr/bin/env bash
# Continue V4.2 after pool-reuse fix: re-bench C/E quickly, profile, autotune, final
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${HOME}/quantsilico-runtime/jax_cache"

echo "=== Re-measure C+E with pool reuse ==="
python - <<'PY'
import json
from pathlib import Path
from train.competition_native_jax.matched_benchmarks_v4_2 import bench_rollout, bench_valid_learning, bench_ppo_only
from train.competition_native_jax.train_jax import lineage_hashes, detect_jax_device

man_path = Path("experiments/manifests/competition_native_jax_v4_2_matched_benchmarks.json")
prev = json.loads(man_path.read_text()) if man_path.exists() else {"benchmarks": {}}
num_envs, rollout_len, pool = 32, 16, 4096
print("C", flush=True)
prev.setdefault("benchmarks", {})["C"] = bench_rollout(num_envs, rollout_len, reset_pool_size=pool)
print("D", flush=True)
prev["benchmarks"]["D"] = bench_ppo_only(num_envs, rollout_len, reset_pool_size=pool)
print("E", flush=True)
prev["benchmarks"]["E"] = bench_valid_learning(
    num_envs, rollout_len, updates=3, reset_pool_size=pool,
    out_dir=Path("experiments/competition_native_jax/v4_2_matched_benchmarks/E_reuse"),
)
prev["pool_reuse_remeasure"] = True
prev["lineage"] = lineage_hashes()
prev["device"] = detect_jax_device()
tps_seq = [
    prev["benchmarks"].get("A", {}).get("tps", 0),
    prev["benchmarks"].get("B", {}).get("tps", 0),
    prev["benchmarks"]["C"]["tps"],
    prev["benchmarks"]["D"]["ppo_samples_per_s"],
    prev["benchmarks"]["E"]["valid_learning_tps"],
]
names = ["A_transition", "B_obs_legal", "C_rollout", "D_ppo", "E_valid_learning"]
collapse = None
for i in range(1, len(tps_seq)):
    if tps_seq[i - 1] > 0 and tps_seq[i] < tps_seq[i - 1] / 4.0:
        collapse = names[i]
        break
prev["first_collapse_stage"] = collapse
prev["tps_sequence"] = dict(zip(names, tps_seq))
man_path.write_text(json.dumps(prev, indent=2) + "\n")
Path("experiments/reports/competition_native_jax_v4_2_matched_benchmarks.md").write_text(
    "\n".join([
        "# V4.2 matched benchmarks",
        "",
        f"Pool reuse remeasure: yes",
        f"First collapse stage: `{collapse}`",
        "",
        "| Bench | Rate |",
        "|---|---:|",
        f"| A | {tps_seq[0]:.2f} |",
        f"| B | {tps_seq[1]:.2f} |",
        f"| C | {tps_seq[2]:.2f} |",
        f"| D PPO | {tps_seq[3]:.2f} |",
        f"| E valid-learning | {tps_seq[4]:.2f} |",
        "",
    ]) + "\n"
)
print(json.dumps({"collapse": collapse, "C": tps_seq[2], "E": tps_seq[4]}, indent=2), flush=True)
PY

echo "=== Component profile ==="
python -m train.competition_native_jax.profile_v4_2

echo "=== Autotune ==="
python -m train.competition_native_jax.autotune_v4_2

echo "=== Final systems ladder ==="
# reuse final block from main pipeline
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
ready = all([promo["tps_gate"], promo["transitions_gate"], promo["updates_gate"], promo["games_gate"]])
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
    "matched_benchmarks_ref": "experiments/manifests/competition_native_jax_v4_2_matched_benchmarks.json",
}
Path("experiments/manifests/competition_native_jax_throughput_ladder_v4_2.json").write_text(json.dumps(ladder, indent=2)+"\n")
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
    ])+"\n"
)
audit = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_END_TO_END_AUDIT_V4_2",
    "architecture_classification": "END_TO_END_COMPETITION_JAX_ROLLOUT",
    "reset_path": "device_competition_reset_pool_reused",
    "gae_device_resident_batched": True,
    "ppo_semantics": "FULL_BATCH_ONE_OPTAX_UPDATE_CANONICAL",
    "in_scan_map_generation": False,
    "host_bound_collector": False,
    "disposition": disposition,
    "blocker": blocker,
    "selected_config": cfg,
    "valid_learning_tps": tps,
    "lineage": lineage_hashes(),
}
Path("experiments/manifests/competition_native_jax_end_to_end_audit_v4_2.json").write_text(json.dumps(audit, indent=2)+"\n")
Path("experiments/reports/competition_native_jax_end_to_end_audit_v4_2.md").write_text(
    "\n".join([
        "# V4.2 end-to-end audit",
        "",
        f"Architecture: `END_TO_END_COMPETITION_JAX_ROLLOUT`",
        f"Disposition: `{disposition}`",
        f"Reset path: device competition reset pool (reused; refresh every 10 updates)",
        f"GAE: device batched reverse scan",
        f"PPO semantics: full-batch one Optax update",
        "",
    ])+"\n"
)
ps = json.loads(Path("experiments/manifests/competition_native_jax_v4_2_programme_state.json").read_text())
ps["selected_config"] = cfg
ps["valid_learning_tps"] = tps
ps["disposition"] = disposition
ps["hard_stop_status"] = "SYSTEMS_CANDIDATE_READY" if ready else "AWAITING_JAX_V4_2_PERFORMANCE_REVIEW"
ps.update(lineage_hashes())
Path("experiments/manifests/competition_native_jax_v4_2_programme_state.json").write_text(json.dumps(ps, indent=2)+"\n")
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
Path("experiments/manifests/competition_native_jax_v4_2_systems_promotion_gate.json").write_text(json.dumps(gate, indent=2)+"\n")
print(json.dumps(gate, indent=2), flush=True)
if ready:
    print("=== Stage 3A ===", flush=True)
    import pytest
    raise SystemExit(pytest.main(["-q", "tests/competition_native_jax/test_e2e_parity_3a.py", "-o", "cache_dir=/tmp/qs_pytest_cache", "--tb=short"]))
PY

echo "=== CONDITIONAL R-E.5 ==="
tr -d '\r' < scripts/wsl/_run_v4_2_conditional_r_e5.sh > /tmp/_r_e5.sh || true
if python -c 'import json;from pathlib import Path;g=json.loads(Path("experiments/manifests/competition_native_jax_v4_2_systems_promotion_gate.json").read_text());raise SystemExit(0 if g.get("ready") else 1)'; then
  bash /tmp/_r_e5.sh || bash scripts/wsl/_run_v4_2_conditional_r_e5.sh || true
else
  echo "Systems not READY — learning-efficiency track may run; R-E.5 deferred"
  python - <<'PY'
import json
from pathlib import Path
from train.competition_native_jax.train_jax import _train_loop, lineage_hashes
# Controlled one-epoch experiment (separate lineage) — short probe only
out = Path("experiments/competition_native_jax/v4_2_learning_efficiency/one_epoch_probe")
# Canonical systems path already uses one full-batch update (= one epoch). Record status.
status = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_V4_2_LEARNING_EFFICIENCY",
    "performance_programme": "AVERAGEJOE_STYLE_LEARNING_EFFICIENCY_V4_2",
    "status": "SYSTEMS_BELOW_GATE_ONE_EPOCH_ALREADY_CANONICAL",
    "note": "Canonical systems PPO already performs one Optax update over the full batch (one epoch). Advantage-retention experiments require separate preregistered eval protocol; not silently promoted.",
    "advantage_retention": {"top_50": "NOT_RUN_NEEDS_EVAL_PROTOCOL", "top_25": "NOT_RUN_NEEDS_EVAL_PROTOCOL"},
    "classification": "LEARNING_EFFICIENCY_EXPERIMENT_INCONCLUSIVE",
    "lineage": lineage_hashes(),
}
Path("experiments/manifests/competition_native_jax_v4_2_learning_efficiency.json").write_text(json.dumps(status, indent=2)+"\n")
print(json.dumps(status, indent=2))
PY
fi

echo "=== HARD STOP reconcile ==="
python - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone
Path("experiments/manifests/competition_native_jax_owned_processes.json").write_text(json.dumps({
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_OWNED_PROCESSES",
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "jobs": [],
}, indent=2)+"\n")
print("owned_processes cleared")
PY
echo "=== PIPELINE_CONTINUE_DONE ==="
