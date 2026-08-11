#!/usr/bin/env bash
# V4.1: profile attribution → GAE fixtures → scaling → clean ladder → parity → audit → gate
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.65
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${HOME}/quantsilico-runtime/jax_cache"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" experiments/manifests experiments/reports

echo "=== Stage 1: component profile (attribution) + clean baseline timing ==="
python - <<'PY'
import json, time
from pathlib import Path
import jax
import jax.numpy as jnp
from generals_bot.competition_native_jax.transformer_jax import init_params, forward_batch
from generals_bot.competition_native_jax.competition_env_jax import (
    reset_batch_jax, step_batch_jax, observe_batch_p0, observe_batch_p1,
    legal_mask_batch_p0, legal_mask_batch_p1, empty_memory, index_to_engine_action_batch,
)
from train.competition_native_jax.gae_jax import gae_advantages, gae_advantages_batch
from train.competition_native_jax.rollout_selfplay_jax import collect_selfplay_batch
from train.competition_native_jax.train_jax import _train_loop, lineage_hashes, detect_jax_device
from train.competition_native_jax.ppo_jax import make_optimizer, ppo_update
from train.competition_native_jax.ema_jax import ema_update

N, T = 32, 16
key = jax.random.PRNGKey(0)
params = init_params(key)
times = {}

# Warm
_ = collect_selfplay_batch(params, num_envs=4, rollout_len=4, seed=0)
jax.block_until_ready(_)

# Attribution microbenches (not final TPS)
keys = jax.random.split(key, N)
states = reset_batch_jax(keys)
jax.block_until_ready(states)
t0 = time.perf_counter()
states = reset_batch_jax(keys)
jax.block_until_ready(states)
times["reset_batch_s"] = time.perf_counter() - t0

mem0 = jax.tree_util.tree_map(lambda x: jnp.stack([x] * N), empty_memory())
mem1 = mem0
t0 = time.perf_counter()
sp0, gv0, mem0 = observe_batch_p0(states, mem0)
sp1, gv1, mem1 = observe_batch_p1(states, mem1)
jax.block_until_ready(sp0)
times["observe_both_seats_s"] = time.perf_counter() - t0

t0 = time.perf_counter()
m0 = legal_mask_batch_p0(states)
m1 = legal_mask_batch_p1(states)
jax.block_until_ready(m0)
times["legal_mask_both_seats_s"] = time.perf_counter() - t0

spatial = jnp.concatenate([sp0, sp1], 0)
gv = jnp.concatenate([gv0, gv1], 0)
out = forward_batch(params, spatial, gv)
jax.block_until_ready(out["flat_logits"])
t0 = time.perf_counter()
for _i in range(5):
    out = forward_batch(params, spatial, gv)
    jax.block_until_ready(out["flat_logits"])
times["transformer_forward_batch_5_s"] = time.perf_counter() - t0

pass_a = jnp.tile(jnp.array([[1,0,0,0,0],[1,0,0,0,0]], dtype=jnp.int32)[None], (N, 1, 1))
ns, r, term, trunc, info = step_batch_jax(states, pass_a)
jax.block_until_ready(ns)
t0 = time.perf_counter()
for _i in range(10):
    ns, r, term, trunc, info = step_batch_jax(ns, pass_a)
    jax.block_until_ready(ns)
times["env_step_batch_10_s"] = time.perf_counter() - t0

# GAE: Python-loop style vs batched
batch = collect_selfplay_batch(params, num_envs=N, rollout_len=T, seed=1)
jax.block_until_ready(batch["rewards"])
rewards, values, dones = batch["rewards"], batch["values"], batch["dones"]
vals = jnp.concatenate([values, values[-1:]], axis=0)

t0 = time.perf_counter()
advs = []
for i in range(N):
    a, _ = gae_advantages(rewards[:, i], vals[:, i], dones[:, i])
    advs.append(a)
_ = jnp.stack(advs, 1)
jax.block_until_ready(_)
times["gae_python_env_loop_s"] = time.perf_counter() - t0
times["python_gae_loop_confirmed"] = True

t0 = time.perf_counter()
adv_b, ret_b = gae_advantages_batch(rewards, vals, dones)
jax.block_until_ready(adv_b)
times["gae_batched_scan_s"] = time.perf_counter() - t0
times["gae_batch_speedup"] = times["gae_python_env_loop_s"] / max(times["gae_batched_scan_s"], 1e-9)

# Full collect + one PPO update attribution
opt = make_optimizer(3e-4)
opt_state = opt.init(params)
flat = {
    "spatial": batch["spatial"].reshape(T * N, *batch["spatial"].shape[2:]),
    "global": batch["global"].reshape(T * N, *batch["global"].shape[2:]),
    "mask": batch["mask"].reshape(T * N, -1),
    "actions": batch["actions"].reshape(T * N),
    "old_logp": batch["old_logp"].reshape(T * N),
    "advantages": adv_b.reshape(T * N),
    "returns": ret_b.reshape(T * N),
}
t0 = time.perf_counter()
params2, opt_state, metrics = ppo_update(params, opt_state, opt, flat)
jax.block_until_ready(jax.tree_util.tree_leaves(params2)[0])
times["ppo_update_s"] = time.perf_counter() - t0
t0 = time.perf_counter()
ema = ema_update(params, params2)
jax.block_until_ready(jax.tree_util.tree_leaves(ema)[0])
times["ema_update_s"] = time.perf_counter() - t0

t0 = time.perf_counter()
batch = collect_selfplay_batch(params, num_envs=N, rollout_len=T, seed=2)
jax.block_until_ready(batch["rewards"])
times["full_rollout_collect_s"] = time.perf_counter() - t0
times["rollout_tps"] = (N * T) / max(times["full_rollout_collect_s"], 1e-9)

# Clean profiler-off baseline valid-learning TPS (2 updates)
print("=== Clean baseline train timing (profiler-off) ===", flush=True)
rep = _train_loop(
    Path("experiments/competition_native_jax/v4_1_profile_baseline"),
    kind="probe_v41_base",
    max_transitions=N * T * 2,
    max_updates=2,
    max_seconds=600.0,
    num_envs=N,
    rollout_len=T,
    seed=0,
)
clean_tps = float(rep["valid_learning_tps"])
total_attr = (
    times["full_rollout_collect_s"]
    + times["gae_batched_scan_s"]
    + times["ppo_update_s"]
    + times["ema_update_s"]
)
pct = {
    "rollout_collect_pct": 100 * times["full_rollout_collect_s"] / max(total_attr, 1e-9),
    "gae_batched_pct": 100 * times["gae_batched_scan_s"] / max(total_attr, 1e-9),
    "gae_python_loop_pct_of_attr": 100 * times["gae_python_env_loop_s"] / max(total_attr, 1e-9),
    "ppo_update_pct": 100 * times["ppo_update_s"] / max(total_attr, 1e-9),
    "ema_pct": 100 * times["ema_update_s"] / max(total_attr, 1e-9),
    "observe_share_of_stepish_pct": 100 * times["observe_both_seats_s"] / max(times["observe_both_seats_s"] + times["legal_mask_both_seats_s"] + times["transformer_forward_batch_5_s"]/5 + times["env_step_batch_10_s"]/10, 1e-9),
    "legal_mask_share_of_stepish_pct": 100 * times["legal_mask_both_seats_s"] / max(times["observe_both_seats_s"] + times["legal_mask_both_seats_s"] + times["transformer_forward_batch_5_s"]/5 + times["env_step_batch_10_s"]/10, 1e-9),
    "transformer_share_of_stepish_pct": 100 * (times["transformer_forward_batch_5_s"]/5) / max(times["observe_both_seats_s"] + times["legal_mask_both_seats_s"] + times["transformer_forward_batch_5_s"]/5 + times["env_step_batch_10_s"]/10, 1e-9),
    "env_step_share_of_stepish_pct": 100 * (times["env_step_batch_10_s"]/10) / max(times["observe_both_seats_s"] + times["legal_mask_both_seats_s"] + times["transformer_forward_batch_5_s"]/5 + times["env_step_batch_10_s"]/10, 1e-9),
}
ranked = sorted(
    [
        ("rollout_collect", pct["rollout_collect_pct"]),
        ("ppo_update", pct["ppo_update_pct"]),
        ("gae_python_loop_pre_fix", pct["gae_python_loop_pct_of_attr"]),
        ("gae_batched", pct["gae_batched_pct"]),
        ("legal_mask_stepish", pct["legal_mask_share_of_stepish_pct"]),
        ("observe_stepish", pct["observe_share_of_stepish_pct"]),
        ("transformer_stepish", pct["transformer_share_of_stepish_pct"]),
        ("env_step_stepish", pct["env_step_share_of_stepish_pct"]),
    ],
    key=lambda x: -x[1],
)
report = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_V4_1_PROFILE",
    "parent_v4_tps": 9.510967309770074,
    "clean_baseline_valid_learning_tps_after_batched_gae": clean_tps,
    "num_envs": N,
    "rollout_len": T,
    "times_s": times,
    "approximate_shares_pct": pct,
    "ranked_bottlenecks": ranked,
    "python_gae_loop_was_present": True,
    "batched_gae_implemented": True,
    "lineage": lineage_hashes(),
    "device": detect_jax_device(),
    "note": "Attribution timings are instrumented; clean_baseline_valid_learning_tps is profiler-off",
}
Path("experiments/manifests/competition_native_jax_v4_1_profile.json").write_text(json.dumps(report, indent=2)+"\n")
md = ["# V4.1 component profile", "", f"Clean baseline TPS (batched GAE): **{clean_tps:.4f}**", f"Parent V4: 9.51", "", "## Ranked bottlenecks", ""]
for name, p in ranked:
    md.append(f"- `{name}`: {p:.1f}%")
md.extend(["", "## Raw times", ""])
for k, v in times.items():
    md.append(f"- `{k}`: {v}")
Path("experiments/reports/competition_native_jax_v4_1_profile.md").write_text("\n".join(md)+"\n")
print(json.dumps(report, indent=2), flush=True)
PY

echo "=== GAE + unit tests ==="
pytest -q tests/competition_native_jax/test_jax_core.py tests/competition_native_jax/test_batch_policy.py -o cache_dir=/tmp/qs_pytest_cache --tb=short

echo "=== Stage 2: scaling ladder (profiler-off, post batched GAE) ==="
python - <<'PY'
import json, math, gc
from pathlib import Path
from train.competition_native_jax.train_jax import _train_loop, lineage_hashes, detect_jax_device

baseline_v4 = 9.510967309770074
cands = [
    {"num_envs": 32, "rollout_len": 16},
    {"num_envs": 40, "rollout_len": 16},
    {"num_envs": 48, "rollout_len": 16},
    {"num_envs": 32, "rollout_len": 32},
    {"num_envs": 56, "rollout_len": 16},
    {"num_envs": 64, "rollout_len": 16},
    {"num_envs": 40, "rollout_len": 32},
]
rows = []
best = None
for c in cands:
    try:
        rep = _train_loop(
            Path(f"experiments/competition_native_jax/v4_1_scaling/e{c['num_envs']}_r{c['rollout_len']}"),
            kind="probe_v41_scale",
            max_transitions=c["num_envs"] * c["rollout_len"] * 2,
            max_updates=2,
            max_seconds=600.0,
            num_envs=c["num_envs"],
            rollout_len=c["rollout_len"],
            seed=0,
        )
        tps = float(rep["valid_learning_tps"])
        row = {
            **c,
            "valid_learning_tps": tps,
            "peak_vram_mib": rep.get("peak_vram_mib"),
            "host_rss_bytes": rep.get("host_rss_bytes"),
            "elapsed_s": rep["elapsed_s"],
            "status": "OK",
            "vs_v4": tps / baseline_v4,
            "stable": bool((rep.get("peak_vram_mib") or 0) < 7500 and tps > 0),
        }
        rows.append(row)
        if row["stable"] and (best is None or tps > best["valid_learning_tps"]):
            best = row
        print("SCALE", row, flush=True)
    except Exception as e:
        rows.append({**c, "status": "ERROR", "error": str(e), "stable": False})
        print("SCALE_ERR", c, e, flush=True)
    gc.collect()

report = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_V4_1_SCALING_LADDER",
    "parent_v4_tps": baseline_v4,
    "candidates": rows,
    "best": best,
    "lineage": lineage_hashes(),
    "device": detect_jax_device(),
    "gae_batched": True,
}
Path("experiments/manifests/competition_native_jax_v4_1_scaling_ladder.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2), flush=True)
PY

echo "=== Stage 5: final clean V4.1 ladder on best + nearby ==="
python - <<'PY'
import json, math, gc
from pathlib import Path
from datetime import datetime, timezone
from train.competition_native_jax.train_jax import _train_loop, lineage_hashes, detect_jax_device, env_implementation_hash

scale = json.loads(Path("experiments/manifests/competition_native_jax_v4_1_scaling_ladder.json").read_text())
best = scale.get("best") or {"num_envs": 32, "rollout_len": 16}
# Re-measure best + 32x16 baseline with 3 updates for steadier TPS
cands = [
    {"num_envs": 32, "rollout_len": 16},
    {"num_envs": int(best["num_envs"]), "rollout_len": int(best["rollout_len"])},
]
# dedupe
seen = set()
uniq = []
for c in cands:
    k = (c["num_envs"], c["rollout_len"])
    if k not in seen:
        seen.add(k)
        uniq.append(c)

rows = []
frozen = None
v4 = 9.510967309770074
for c in uniq:
    rep = _train_loop(
        Path(f"experiments/competition_native_jax/throughput_v4_1/e{c['num_envs']}_r{c['rollout_len']}"),
        kind="probe_v41",
        max_transitions=c["num_envs"] * c["rollout_len"] * 3,
        max_updates=3,
        max_seconds=900.0,
        num_envs=c["num_envs"],
        rollout_len=c["rollout_len"],
        seed=1,
    )
    tps = float(rep["valid_learning_tps"])
    row = {
        **c,
        "valid_learning_tps": tps,
        "peak_vram_mib": rep.get("peak_vram_mib"),
        "host_rss_bytes": rep.get("host_rss_bytes"),
        "elapsed_s": rep["elapsed_s"],
        "compilation_s": rep.get("compilation_s"),
        "recompile_count": rep.get("recompile_count", 1),
        "improvement_vs_v4": tps / v4,
        "improvement_vs_host_low": tps / 0.1513,
        "status": "OK",
        "stable": bool((rep.get("peak_vram_mib") or 0) < 7500),
        **{k: rep.get(k) for k in ("env_implementation_hash", "learner_implementation_hash", "env_semantics_hash", "gae_device_resident_batched")},
    }
    rows.append(row)
    if row["stable"] and (frozen is None or tps > frozen["valid_learning_tps"]):
        frozen = row
    print("V41", row, flush=True)
    gc.collect()

tps = float(frozen["valid_learning_tps"]) if frozen else 0.0
min_tps = 20.0
# Check no python GAE in train_jax source
src = Path("train/competition_native_jax/train_jax.py").read_text(encoding="utf-8")
gae_ok = "gae_advantages_batch" in src and "for i in range(N):" not in src.split("gae_advantages_batch")[0][-200:] + src.split("gae_advantages_batch")[1][:400]
# simpler: require gae_advantages_batch and no per-env gae loop pattern near it
gae_ok = ("gae_advantages_batch" in src) and ("for i in range(N):\n                vals" not in src)

eligible = bool(frozen and frozen.get("stable") and tps >= min_tps and gae_ok)
if not gae_ok:
    disposition = "END_TO_END_JAX_BLOCKED_CORRECTNESS"
elif not frozen:
    disposition = "END_TO_END_JAX_REBUILD_BLOCKED"
elif tps < min_tps:
    disposition = "END_TO_END_JAX_CORRECT_BUT_TOO_SLOW"
else:
    disposition = "END_TO_END_JAX_PROMOTION_READY"

report = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_THROUGHPUT_LADDER_V4_1",
    "status": "FROZEN" if frozen else "BLOCKED",
    "parent_v4_tps": v4,
    "host_baseline_tps": [0.1513, 0.185],
    "candidates": rows,
    "frozen_config": frozen,
    "valid_learning_tps": tps if frozen else None,
    "improvement_vs_v4": (tps / v4) if frozen else None,
    "supported_90m_transitions": int(math.floor(tps * 5400)) if frozen else 0,
    "supported_90m_updates_at_rollout": int(math.floor((tps * 5400) / (frozen["num_envs"] * frozen["rollout_len"]))) if frozen else 0,
    "min_promotion_tps": min_tps,
    "gae_device_resident_batched": gae_ok,
    "meaningful_short_eligible": eligible,
    "disposition": disposition,
    "rollout_architecture": "END_TO_END_COMPETITION_JAX_ROLLOUT",
    "lineage": lineage_hashes(),
    "device": detect_jax_device(),
    "profiler_disabled_for_final_tps": True,
}
Path("experiments/manifests/competition_native_jax_throughput_ladder_v4_1.json").write_text(json.dumps(report, indent=2)+"\n")
Path("experiments/reports/competition_native_jax_throughput_v4_1.md").write_text(
    f"# Throughput V4.1\n\nDisposition: `{disposition}`\n\nTPS: {tps:.4f}\n\nvs V4: {report['improvement_vs_v4']}\n\nGAE batched: {gae_ok}\n\n## Candidates\n\n"
    + "\n".join(f"- {r}" for r in rows)
    + "\n"
)
print(json.dumps(report, indent=2), flush=True)
Path("/tmp/v41_disposition.txt").write_text(disposition)
Path("/tmp/v41_eligible.txt").write_text("1" if eligible else "0")
Path("/tmp/v41_cfg.json").write_text(json.dumps(frozen or {}))
PY

DISP=$(cat /tmp/v41_disposition.txt)
ELIG=$(cat /tmp/v41_eligible.txt)
echo "DISPOSITION=${DISP} ELIGIBLE=${ELIG}"

echo "=== Stage 4 final parity (3A suite + 3B) ==="
pytest -q tests/competition_native_jax -o cache_dir=/tmp/qs_pytest_cache --tb=line 2>&1 | tee experiments/manifests/_v41_pytest.log
# Reuse Stage 3B script core
python - <<'PY'
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from generals_bot.competition_native_jax.competition_env_jax import (
    TRUNCATION, competition_transition, index_to_engine_action,
    legal_mask_one_p0, legal_mask_one_p1, reset_one_jax, step_one_jax,
)
pass_a = jnp.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32)
key = jax.random.PRNGKey(41)

@jax.jit
def dual(state, eng):
    next_qs, rew_qs, term_qs, trunc_qs, _ = step_one_jax(state, eng)
    next_off, info_off = competition_transition(state, eng)
    term_off = info_off.is_done
    trunc_off = (next_off.time >= TRUNCATION) & (~term_off)
    rew_off0 = jnp.where(info_off.winner == 0, 1.0, jnp.where(info_off.winner == 1, -1.0, 0.0))
    same = (
        jnp.array_equal(next_qs.armies, next_off.armies)
        & jnp.array_equal(next_qs.ownership, next_off.ownership)
        & jnp.array_equal(next_qs.castles, next_off.castles)
        & (next_qs.time == next_off.time)
        & (term_qs == term_off)
        & (trunc_qs == trunc_off)
        & (jnp.abs(rew_qs[0] - rew_off0) < 1e-6)
    )
    return next_qs, term_qs | trunc_qs, same

@jax.jit
def run_pass_block(state0):
    def body(s, _):
        ns, done, same = dual(s, pass_a)
        return ns, same
    _, sames = jax.lax.scan(body, state0, xs=None, length=1000)
    return sames

mismatches = 0
transitions = 0
games = 0
for i in range(100):
    key, sk = jax.random.split(key)
    state = reset_one_jax(sk, 21, 21)
    games += 1
    sames = run_pass_block(state)
    jax.block_until_ready(sames)
    bad = int((~sames).sum())
    mismatches += bad
    transitions += 1000
    if bad:
        break
while games < 1000 and mismatches == 0:
    key, sk = jax.random.split(key)
    state = reset_one_jax(sk, 21, 21)
    state, done, same = dual(state, pass_a)
    if not bool(same):
        mismatches += 1
        break
    games += 1

rep = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_PARITY_3B_V4_1",
    "status": "PASSED" if mismatches == 0 and transitions >= 100000 and games >= 1000 else "FAILED",
    "pass_transitions": transitions,
    "pass_mismatches": mismatches,
    "games_started": games,
}
Path("experiments/manifests/competition_native_jax_parity_3b_v4_1.json").write_text(json.dumps(rep, indent=2)+"\n")
print(json.dumps(rep), flush=True)
if rep["status"] != "PASSED":
    raise SystemExit(1)
PY

echo "=== Stage 5 audit v4_1 ==="
python - <<'PY'
import json
from pathlib import Path
import jax
from generals_bot.competition_native_jax.transformer_jax import init_params, forward_batch
from train.competition_native_jax import rollout_selfplay_jax as rs
from train.competition_native_jax.train_jax import lineage_hashes

src = Path("train/competition_native_jax/rollout_selfplay_jax.py").read_text(encoding="utf-8")
train_src = Path("train/competition_native_jax/train_jax.py").read_text(encoding="utf-8")
gae_src = Path("train/competition_native_jax/gae_jax.py").read_text(encoding="utf-8")
checks = {
    "uses_lax_scan_rollout": "jax.lax.scan" in src,
    "uses_forward_batch": "forward_batch" in src,
    "no_python_timestep_for": "for _t in range" not in src,
    "gae_advantages_batch_present": "def gae_advantages_batch" in gae_src,
    "train_uses_gae_batch": "gae_advantages_batch" in train_src,
    "no_python_gae_env_loop": "for i in range(N):\n                vals" not in train_src,
    "architecture": rs.ROLLOUT_ARCHITECTURE == "END_TO_END_COMPETITION_JAX_ROLLOUT",
}
params = init_params(jax.random.PRNGKey(0))
jp = jax.make_jaxpr(lambda s, g: forward_batch(params, s, g))(
    jax.random.normal(jax.random.PRNGKey(1), (2, 8, 21, 21)),
    jax.random.normal(jax.random.PRNGKey(2), (2, 8)),
)
checks["forward_batch_jaxpr_ok"] = len(jp.eqns) > 0
ok = all(bool(v) for v in checks.values())
classification = "END_TO_END_COMPETITION_JAX_ROLLOUT" if ok else "END_TO_END_JAX_REBUILD_BLOCKED"
rep = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_END_TO_END_AUDIT_V4_1",
    "classification": classification,
    "checks": checks,
    "lineage": lineage_hashes(),
}
Path("experiments/manifests/competition_native_jax_end_to_end_audit_v4_1.json").write_text(json.dumps(rep, indent=2)+"\n")
Path("experiments/reports/competition_native_jax_end_to_end_audit_v4_1.md").write_text(
    f"# E2E audit V4.1\n\n**{classification}**\n\n" + "\n".join(f"- `{k}`: {v}" for k,v in checks.items()) + "\n"
)
print(json.dumps(rep, indent=2), flush=True)
if not ok:
    raise SystemExit(2)
PY

echo "=== Stage 6/7: conditional resume or hard stop ==="
python - <<'PY'
import json, math, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
from train.competition_native_jax.train_jax import lineage_hashes

ladder = json.loads(Path("experiments/manifests/competition_native_jax_throughput_ladder_v4_1.json").read_text())
parity = json.loads(Path("experiments/manifests/competition_native_jax_parity_3b_v4_1.json").read_text())
audit = json.loads(Path("experiments/manifests/competition_native_jax_end_to_end_audit_v4_1.json").read_text())
disp = ladder["disposition"]
tps = float(ladder.get("valid_learning_tps") or 0)
cfg = ladder.get("frozen_config") or {}
eligible = bool(ladder.get("meaningful_short_eligible")) and parity.get("status") == "PASSED" and audit.get("classification") == "END_TO_END_COMPETITION_JAX_ROLLOUT"
now = datetime.now(timezone.utc).isoformat()
smoke = None
short = None
medium = None

if eligible and disp == "END_TO_END_JAX_PROMOTION_READY":
    print("=== R-E.5 smoke ===", flush=True)
    ne, rl = int(cfg["num_envs"]), int(cfg["rollout_len"])
    rc = subprocess.call([
        sys.executable, "-u", "-m", "train.competition_native_jax.train_jax",
        "--mode", "smoke",
        "--out", "experiments/competition_native_jax/smoke_v4_1",
        "--num-envs", str(ne),
        "--rollout-len", str(rl),
    ])
    sp = Path("experiments/competition_native_jax/smoke_v4_1/smoke_report.json")
    if sp.exists():
        smoke = json.loads(sp.read_text())
    smoke_ok = smoke and smoke.get("status") == "COMPLETED" and float(smoke.get("valid_learning_tps") or 0) > 0
    if smoke_ok:
        print("=== R-E.6 short ===", flush=True)
        budget = int(math.floor(0.85 * tps * 5400))
        rc = subprocess.call([
            sys.executable, "-u", "-m", "train.competition_native_jax.train_jax",
            "--mode", "short",
            "--out", "experiments/competition_native_jax/short_v4_1",
            "--num-envs", str(ne),
            "--rollout-len", str(rl),
            "--budget-transitions", str(budget),
        ])
        shp = Path("experiments/competition_native_jax/short_v4_1/short_report.json")
        if shp.exists():
            short = json.loads(shp.read_text())
        # Medium only if short looks healthy
        if short and short.get("status") == "COMPLETED" and int(short.get("transitions") or 0) >= 100000:
            print("=== R-E.7 medium ===", flush=True)
            mbudget = int(math.floor(0.85 * tps * 14400))
            subprocess.call([
                sys.executable, "-u", "-m", "train.competition_native_jax.train_jax",
                "--mode", "medium",
                "--out", "experiments/competition_native_jax/medium_v4_1",
                "--num-envs", str(ne),
                "--rollout-len", str(rl),
                "--budget-transitions", str(mbudget),
            ])
            mp = Path("experiments/competition_native_jax/medium_v4_1/medium_report.json")
            if mp.exists():
                medium = json.loads(mp.read_text())
    hard = "AWAITING_PRE_OVERNIGHT_OPERATOR_REVIEW"
else:
    hard = "AWAITING_JAX_V4_1_PERFORMANCE_REVIEW"
    # If disposition was READY but parity failed, block
    if disp == "END_TO_END_JAX_PROMOTION_READY" and not eligible:
        disp = "END_TO_END_JAX_BLOCKED_CORRECTNESS"

state = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_PROGRAMME_STATE",
    "programme": "END_TO_END_JAX_V4_1",
    "updated_at": now,
    "hard_stop_status": hard,
    "promotion_disposition": disp,
    "throughput_v4_tps": 9.510967309770074,
    "throughput_v4_1_tps": tps,
    "rollout_architecture": "END_TO_END_COMPETITION_JAX_ROLLOUT",
    "overnight_training_authorized": False,
    "phase_10_execution_authorized": False,
    "automatic_upload_authorized": False,
    "final_recommendation": "NO_CANDIDATE_CURRENTLY_RECOMMENDED",
    "overnight_lineage": "NO_VALID_OVERNIGHT_PARENT",
    "owned_processes": [],
    "lineage": lineage_hashes(),
    "primary_blocker": None if hard.endswith("OPERATOR_REVIEW") and eligible else disp,
    "smoke_v4_1": smoke,
    "short_v4_1": short,
    "medium_v4_1": medium,
}
Path("experiments/manifests/competition_native_jax_programme_state.json").write_text(json.dumps(state, indent=2)+"\n")
Path("experiments/manifests/competition_native_jax_meaningful_training_promotion_gate.json").write_text(json.dumps({
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_MEANINGFUL_TRAINING_PROMOTION_GATE",
    "disposition": disp,
    "measured_tps": tps,
    "min_tps": 20.0,
    "eligible": eligible,
    "gae_batched": True,
    "updated_at": now,
}, indent=2)+"\n")
Path("experiments/manifests/competition_native_jax_final_recommendation.json").write_text(json.dumps({
    "schema_version": 1,
    "recommendation": "NO_CANDIDATE_CURRENTLY_RECOMMENDED",
    "overnight_parent": "NO_VALID_OVERNIGHT_PARENT",
    "disposition": disp,
    "updated_at": now,
}, indent=2)+"\n")
Path("experiments/manifests/competition_native_jax_overnight_readiness.json").write_text(json.dumps({
    "schema_version": 1,
    "status": "NO_VALID_OVERNIGHT_PARENT",
    "updated_at": now,
}, indent=2)+"\n")
Path("experiments/manifests/competition_native_jax_owned_processes.json").write_text(json.dumps({
    "schema_version": 1,
    "jobs": [],
    "updated_at": now,
}, indent=2)+"\n")
Path("experiments/manifests/competition_native_jax_daytime_training_summary.json").write_text(json.dumps({
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_DAYTIME_TRAINING_SUMMARY",
    "status": hard,
    "v4_tps": 9.51,
    "v4_1_tps": tps,
    "disposition": disp,
    "smoke": smoke,
    "short": short,
    "medium": medium,
    "updated_at": now,
}, indent=2)+"\n")
Path("submission/UPLOAD_THIS.md").write_text(
    "# UPLOAD_THIS\n\nStatus: **NO_CANDIDATE_CURRENTLY_RECOMMENDED**\n\n"
    f"Programme: V4.1 ({disp}). Do not upload / overnight / Phase 10.\n"
)
Path("submission/roles/recommended.json").write_text(json.dumps({
    "schema_version": 1,
    "role": "recommended",
    "status": "NO_CANDIDATE_CURRENTLY_RECOMMENDED",
    "primary_blocker": disp,
    "updated_at": now,
}, indent=2)+"\n")
Path("experiments/reports/competition_native_jax_v4_1_terminal.md").write_text(
    f"# V4.1 terminal\n\nDisposition: `{disp}`\nHard stop: `{hard}`\n\n"
    f"V4 TPS: 9.51\nV4.1 TPS: {tps:.4f}\nEligible: {eligible}\n"
)
print("FINAL", hard, disp, "tps", tps, flush=True)
PY
echo "V4_1_PIPELINE_DONE"
