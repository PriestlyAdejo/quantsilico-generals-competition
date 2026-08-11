#!/usr/bin/env bash
# After 3A: Stage 4 scan smoke + v4_dev throughput profile
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${HOME}/quantsilico-runtime/jax_cache"

echo "=== Stage 4 collect_selfplay_batch smoke ==="
python - <<'PY'
import json, time
from pathlib import Path
import jax
from generals_bot.competition_native_jax.transformer_jax import init_params
from train.competition_native_jax.rollout_selfplay_jax import collect_selfplay_batch, ROLLOUT_ARCHITECTURE
from train.competition_native_jax.train_jax import env_implementation_hash

key = jax.random.PRNGKey(0)
params = init_params(key)
t0 = time.perf_counter()
batch = collect_selfplay_batch(params, num_envs=4, rollout_len=8, seed=0)
jax.block_until_ready(batch["rewards"])
compile_s = time.perf_counter() - t0
t1 = time.perf_counter()
batch = collect_selfplay_batch(params, num_envs=4, rollout_len=8, seed=1)
jax.block_until_ready(batch["rewards"])
run_s = time.perf_counter() - t1
tps = (4 * 8) / max(run_s, 1e-9)
print(json.dumps({"compile_s": compile_s, "run_s": run_s, "tps": tps, "arch": ROLLOUT_ARCHITECTURE, "env_hash": env_implementation_hash()}, indent=2), flush=True)
assert batch["rewards"].shape == (8, 4)
assert batch["rollout_architecture"] == "END_TO_END_COMPETITION_JAX_ROLLOUT"
PY

echo "=== v4_dev preliminary throughput (not final ladder) ==="
python - <<'PY'
import json, math, time
from pathlib import Path
from train.competition_native_jax.train_jax import _train_loop, _vram_used_mib, _rss_bytes, detect_jax_device, env_implementation_hash

baseline = [0.1513, 0.185]
candidates = [
    {"num_envs": 8, "rollout_len": 16},
    {"num_envs": 16, "rollout_len": 16},
    {"num_envs": 32, "rollout_len": 8},
]
rows = []
best = None
for c in candidates:
    try:
        rep = _train_loop(
            Path(f"experiments/competition_native_jax/throughput_v4_dev/probe_e{c['num_envs']}_r{c['rollout_len']}"),
            kind="probe_v4_dev",
            max_transitions=c["num_envs"] * c["rollout_len"] * 2,
            max_updates=2,
            max_seconds=600.0,
            num_envs=c["num_envs"],
            rollout_len=c["rollout_len"],
            seed=0,
        )
        tps = float(rep["valid_learning_tps"])
        row = {**c, "valid_learning_tps": tps, "peak_vram_mib": rep.get("peak_vram_mib"),
               "elapsed_s": rep["elapsed_s"], "status": "OK",
               "improvement_vs_baseline_low": tps / baseline[0],
               "rollout_architecture": rep.get("rollout_architecture"),
               "env_implementation_hash": rep.get("env_implementation_hash")}
        rows.append(row)
        if best is None or tps > best["valid_learning_tps"]:
            best = row
    except Exception as e:
        rows.append({**c, "status": "ERROR", "error": str(e)})

report = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_THROUGHPUT_V4_DEV",
    "note": "Preliminary development profile ONLY — not the final promotion ladder",
    "baseline_valid_tps": baseline,
    "candidates": rows,
    "best": best,
    "device": detect_jax_device(),
    "env_implementation_hash": env_implementation_hash(),
}
Path("experiments/manifests/competition_native_jax_throughput_v4_dev.json").write_text(json.dumps(report, indent=2)+"\n")
md = ["# Throughput V4 development profile", "", f"Best TPS: {best}", "", "Not final ladder_v4.", ""]
for r in rows:
    md.append(f"- {r}")
Path("experiments/reports/competition_native_jax_throughput_v4_dev.md").write_text("\n".join(md)+"\n")
print(json.dumps(report, indent=2), flush=True)
PY
echo "STAGE4_V4DEV_DONE"
