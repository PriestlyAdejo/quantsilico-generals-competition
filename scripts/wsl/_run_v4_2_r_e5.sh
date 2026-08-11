#!/usr/bin/env bash
# After systems READY: Stage 3A + R-E.5 smoke (learning-efficiency deferred)
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${HOME}/quantsilico-runtime/jax_cache"

python - <<'PY'
import json, sys
from pathlib import Path
g = json.loads(Path("experiments/manifests/competition_native_jax_v4_2_systems_promotion_gate.json").read_text())
assert g.get("ready"), g
print("GATE_OK", g["selected"], g["valid_learning_tps"])
PY

echo "=== Stage 3A ==="
pytest -q tests/competition_native_jax/test_e2e_parity_3a.py tests/competition_native_jax/test_v4_2_reset_pool.py \
  -o cache_dir=/tmp/qs_pytest_cache --tb=short

echo "=== R-E.5 smoke (100k / 30m first limit) ==="
python - <<'PY'
import json
from pathlib import Path
from train.competition_native_jax.train_jax import _train_loop, lineage_hashes
g = json.loads(Path("experiments/manifests/competition_native_jax_v4_2_systems_promotion_gate.json").read_text())
cfg = g["selected"]
rep = _train_loop(
    Path("experiments/competition_native_jax/v4_2_smoke_r_e5"),
    kind="smoke_r_e5",
    max_transitions=100_000,
    max_updates=10_000,
    max_seconds=30 * 60,
    seed=0,
    **cfg,
)
rep["lineage_at_smoke"] = lineage_hashes()
rep["parent_gate"] = g
Path("experiments/manifests/competition_native_jax_v4_2_smoke_r_e5.json").write_text(json.dumps(rep, indent=2)+"\n")
Path("experiments/reports/competition_native_jax_v4_2_smoke_r_e5.md").write_text(
    "\n".join([
        "# V4.2 R-E.5 smoke",
        "",
        f"Config: `{cfg}`",
        f"status: `{rep['status']}`",
        f"transitions: {rep['transitions']}",
        f"updates: {rep['updates']}",
        f"valid_learning_tps: {rep['valid_learning_tps']:.4f}",
        f"elapsed_s: {rep['elapsed_s']:.1f}",
        "",
    ])+"\n"
)
print(json.dumps({"status": rep["status"], "transitions": rep["transitions"], "updates": rep["updates"], "tps": rep["valid_learning_tps"]}, indent=2))
# Update programme state
ps = json.loads(Path("experiments/manifests/competition_native_jax_v4_2_programme_state.json").read_text())
ps["hard_stop_status"] = "R_E5_SMOKE_COMPLETE"
ps["smoke_r_e5"] = "experiments/manifests/competition_native_jax_v4_2_smoke_r_e5.json"
ps["final_recommendation"] = "NO_CANDIDATE_CURRENTLY_RECOMMENDED"
ps["overnight_lineage"] = "NO_VALID_OVERNIGHT_PARENT"
ps["overnight_training_authorized"] = False
Path("experiments/manifests/competition_native_jax_v4_2_programme_state.json").write_text(json.dumps(ps, indent=2)+"\n")
PY

echo "=== HARD STOP (stop before overnight; R-E.6+ needs operator) ==="
python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
Path("experiments/manifests/competition_native_jax_owned_processes.json").write_text(json.dumps({
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_OWNED_PROCESSES",
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "jobs": [],
}, indent=2)+"\n")
term = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_V4_2_TERMINAL",
    "final_status": "AWAITING_PRE_OVERNIGHT_OPERATOR_REVIEW",
    "systems_disposition": "END_TO_END_JAX_SYSTEMS_PROMOTION_READY",
    "learning_efficiency": "DEFERRED_SYSTEMS_QUALIFIED",
    "overnight_training_authorized": False,
    "automatic_upload_authorized": False,
    "phase_10_execution_authorized": False,
    "final_recommendation": "NO_CANDIDATE_CURRENTLY_RECOMMENDED",
    "overnight_lineage": "NO_VALID_OVERNIGHT_PARENT",
    "next_human_action": "Review R-E.5 smoke; authorise R-E.6 short / R-E.7 medium / R-F if smoke healthy; do not upload or overnight yet.",
}
Path("experiments/manifests/competition_native_jax_v4_2_terminal.json").write_text(json.dumps(term, indent=2)+"\n")
Path("experiments/reports/competition_native_jax_v4_2_terminal.md").write_text(
    "\n".join([
        "# V4.2 terminal",
        "",
        "Systems: `END_TO_END_JAX_SYSTEMS_PROMOTION_READY`",
        "R-E.5 smoke completed under promotion-eligible config.",
        "Learning-efficiency experiments deferred (systems already qualified).",
        "Upload / overnight / Phase 10: false",
        "",
        "Next human action: review smoke artefacts; authorise R-E.6+ if desired.",
        "",
    ])+"\n"
)
print(json.dumps(term, indent=2))
PY
echo "=== R_E5_DONE ==="
