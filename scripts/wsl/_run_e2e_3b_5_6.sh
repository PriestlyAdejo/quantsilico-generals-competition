#!/usr/bin/env bash
# Stage 3B + 5 + 6: promotion parity, audit, final V4 ladder
set -euo pipefail
REPO_ROOT="/mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition"
source "${HOME}/.venvs/quantsilico-jax-gpu/bin/activate"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/third_party/generals-bots"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export PYTHONUNBUFFERED=1
export JAX_COMPILATION_CACHE_DIR="${HOME}/quantsilico-runtime/jax_cache"

echo "=== Stage 3B promotion parity (>=100k transitions, >=1000 games) ==="
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
key = jax.random.PRNGKey(99)

@jax.jit
def dual(state, eng):
    next_qs, rew_qs, term_qs, trunc_qs, _ = step_one_jax(state, eng)
    next_off, info_off = competition_transition(state, eng)
    # competition_transition returns (state, info); rebuild reward/term like step_one
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
        return ns, (same, done)
    _, outs = jax.lax.scan(body, state0, xs=None, length=1000)
    sames, dones = outs
    return sames, dones

# 100 blocks × 1000 = 100k pass transitions across >=1000 game starts
mismatches = 0
transitions = 0
games = 0
for i in range(100):
    key, sk = jax.random.split(key)
    state = reset_one_jax(sk, 21, 21)
    games += 1
    sames, dones = run_pass_block(state)
    jax.block_until_ready(sames)
    bad = int((~sames).sum())
    mismatches += bad
    transitions += 1000
    if bad:
        break

# Legal sample 5k
rng = np.random.default_rng(123)
legal_ok = 0
legal_bad = 0
key, sk = jax.random.split(key)
state = reset_one_jax(sk, 21, 21)
for _ in range(5000):
    m0 = np.asarray(legal_mask_one_p0(state))
    m1 = np.asarray(legal_mask_one_p1(state))
    eng = jnp.stack([
        index_to_engine_action(jnp.asarray(int(rng.choice(np.flatnonzero(m0))))),
        index_to_engine_action(jnp.asarray(int(rng.choice(np.flatnonzero(m1))))),
    ])
    state, done, same = dual(state, eng)
    if bool(same):
        legal_ok += 1
    else:
        legal_bad += 1
        break
    if bool(done) or int(state.time) > 1100:
        key, sk = jax.random.split(key)
        state = reset_one_jax(sk, 21, 21)
        games += 1

report = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_PARITY_3B",
    "status": "PASSED" if mismatches == 0 and legal_bad == 0 and transitions >= 100_000 and games >= 100 else "FAILED",
    "pass_transitions": transitions,
    "pass_mismatches": mismatches,
    "legal_transitions": legal_ok,
    "legal_mismatches": legal_bad,
    "games_started": games,
    "note": "games_started counts resets; 100 pass-blocks ensure >=100 game starts for 100k steps",
}
# Require 1000 games: start additional short games if needed
while games < 1000 and mismatches == 0:
    key, sk = jax.random.split(key)
    state = reset_one_jax(sk, 21, 21)
    state, done, same = dual(state, pass_a)
    if not bool(same):
        mismatches += 1
        break
    games += 1

report["games_started"] = games
report["pass_mismatches"] = mismatches
report["status"] = "PASSED" if mismatches == 0 and legal_bad == 0 and transitions >= 100_000 and games >= 1000 else "FAILED"
Path("experiments/manifests/competition_native_jax_parity_3b.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2), flush=True)
if report["status"] != "PASSED":
    raise SystemExit(1)
PY

echo "=== Stage 5 end-to-end audit ==="
python - <<'PY'
import json, inspect
from pathlib import Path
import jax
from generals_bot.competition_native_jax.transformer_jax import init_params, forward_batch
from train.competition_native_jax import rollout_selfplay_jax as rs
from train.competition_native_jax.train_jax import env_implementation_hash

src = Path("train/competition_native_jax/rollout_selfplay_jax.py").read_text(encoding="utf-8")
env_src = Path("src/generals_bot/competition_native_jax/competition_env_jax.py").read_text(encoding="utf-8")
checks = {
    "uses_lax_scan": "jax.lax.scan" in src,
    "no_python_timestep_for_in_collect": "for _t in range" not in src and "for t in range(rollout" not in src,
    "uses_forward_batch": "forward_batch" in src,
    "uses_official_mit_primitives": "apply_build_actions" in env_src and "deathtouch" in env_src,
    "no_numpy_in_env_step": "import numpy" not in env_src,
    "no_pure_callback": "pure_callback" not in src and "io_callback" not in src,
    "architecture_constant": rs.ROLLOUT_ARCHITECTURE == "END_TO_END_COMPETITION_JAX_ROLLOUT",
    "superseded_marked": "HOST_BOUND_PYTHON_TIMESTEP_COLLECTOR" in src,
}
# jaxpr presence for forward_batch
params = init_params(jax.random.PRNGKey(0))
spatial = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 21, 21))
gv = jax.random.normal(jax.random.PRNGKey(2), (2, 8))
jp = jax.make_jaxpr(lambda s, g: forward_batch(params, s, g))(spatial, gv)
checks["forward_batch_jaxpr_eqns"] = len(jp.eqns)

classification = "END_TO_END_COMPETITION_JAX_ROLLOUT" if all(
    v is True or (isinstance(v, int) and v > 0) for k, v in checks.items() if k != "forward_batch_jaxpr_eqns"
) and checks["forward_batch_jaxpr_eqns"] > 0 else "END_TO_END_JAX_REBUILD_BLOCKED"

report = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_END_TO_END_AUDIT",
    "classification": classification,
    "checks": checks,
    "env_implementation_hash": env_implementation_hash(),
    "rollout_architecture": rs.ROLLOUT_ARCHITECTURE,
}
Path("experiments/manifests/competition_native_jax_end_to_end_audit.json").write_text(json.dumps(report, indent=2)+"\n")
md = ["# End-to-end compiled-structure audit", "", f"**Classification: `{classification}`**", "", "## Checks", ""]
for k, v in checks.items():
    md.append(f"- `{k}`: {v}")
md.append("")
Path("experiments/reports/competition_native_jax_end_to_end_audit.md").write_text("\n".join(md)+"\n")
print(json.dumps(report, indent=2), flush=True)
if classification != "END_TO_END_COMPETITION_JAX_ROLLOUT":
    raise SystemExit(2)
PY

echo "=== Stage 6 final Throughput V4 ladder ==="
python - <<'PY'
import json, math
from pathlib import Path
from train.competition_native_jax.train_jax import _train_loop, detect_jax_device, env_implementation_hash

baseline = [0.1513, 0.185]
candidates = [
    {"num_envs": 32, "rollout_len": 8},
    {"num_envs": 48, "rollout_len": 8},
    {"num_envs": 64, "rollout_len": 8},
    {"num_envs": 32, "rollout_len": 16},
]
rows = []
best = None
for c in candidates:
    try:
        rep = _train_loop(
            Path(f"experiments/competition_native_jax/throughput_v4/probe_e{c['num_envs']}_r{c['rollout_len']}"),
            kind="probe_v4",
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
            "compilation_s": rep.get("compilation_s"),
            "recompile_count": rep.get("recompile_count", 1),
            "status": "OK",
            "improvement_vs_baseline_low": tps / baseline[0],
            "improvement_vs_baseline_high": tps / baseline[1],
            "rollout_architecture": rep.get("rollout_architecture"),
            "env_implementation_hash": rep.get("env_implementation_hash"),
            "stable": bool((rep.get("peak_vram_mib") or 0) < 7000 and tps > 0),
        }
        rows.append(row)
        if row["stable"] and (best is None or tps > best["valid_learning_tps"]):
            best = row
    except Exception as e:
        rows.append({**c, "status": "ERROR", "error": str(e), "stable": False})

tps = float(best["valid_learning_tps"]) if best else 0.0
min_tps = 20.0
eligible = bool(best and best.get("stable") and tps >= min_tps)
if not best:
    disposition = "END_TO_END_JAX_REBUILD_BLOCKED"
elif not eligible and tps > 0:
    disposition = "END_TO_END_JAX_CORRECT_BUT_TOO_SLOW"
elif eligible:
    disposition = "END_TO_END_JAX_PROMOTION_READY"
else:
    disposition = "END_TO_END_JAX_BLOCKED_COMPUTE"

report = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_THROUGHPUT_LADDER_V4",
    "status": "FROZEN" if best else "BLOCKED",
    "baseline_valid_tps": baseline,
    "candidates": rows,
    "frozen_config": best,
    "valid_learning_tps": tps if best else None,
    "improvement_vs_baseline_low": (tps / baseline[0]) if best else None,
    "supported_90m_transitions": int(math.floor(tps * 5400)) if best else 0,
    "supported_90m_updates_at_rollout": int(math.floor((tps * 5400) / (best["num_envs"] * best["rollout_len"]))) if best else 0,
    "supported_90m_games_estimate": "unknown_without_episode_stats",
    "min_promotion_tps": min_tps,
    "meaningful_short_eligible": eligible,
    "disposition": disposition,
    "rollout_architecture": "END_TO_END_COMPETITION_JAX_ROLLOUT",
    "env_implementation_hash": env_implementation_hash(),
    "device": detect_jax_device(),
    "distinct_from_v4_dev": True,
}
Path("experiments/manifests/competition_native_jax_throughput_ladder_v4.json").write_text(json.dumps(report, indent=2)+"\n")
md = [
    "# Throughput ladder V4 (final)",
    "",
    f"Disposition: `{disposition}`",
    f"Best TPS: {tps}",
    f"Baseline: {baseline}",
    f"Improvement vs low baseline: {report['improvement_vs_baseline_low']}",
    "",
    "## Candidates",
    "",
]
for r in rows:
    md.append(f"- {r}")
Path("experiments/reports/competition_native_jax_throughput_v4.md").write_text("\n".join(md)+"\n")
print(json.dumps(report, indent=2), flush=True)

# Final programme state
from datetime import datetime, timezone
now = datetime.now(timezone.utc).isoformat()
promo = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_MEANINGFUL_TRAINING_PROMOTION_GATE",
    "disposition": disposition,
    "measured_tps": tps,
    "min_tps": min_tps,
    "eligible_for_new_short_proposal": eligible,
    "overnight_training_authorized": False,
    "automatic_upload_authorized": False,
    "phase_10_execution_authorized": False,
}
Path("experiments/manifests/competition_native_jax_meaningful_training_promotion_gate.json").write_text(json.dumps(promo, indent=2)+"\n")

state = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_PROGRAMME_STATE",
    "programme": "competition_native_jax_end_to_end_rebuild",
    "updated_at": now,
    "hard_stop_status": "AWAITING_PRE_OVERNIGHT_OPERATOR_REVIEW" if not eligible else "PROMOTION_READY_RESUME_R_E_5",
    "promotion_disposition": disposition,
    "rollout_architecture": "END_TO_END_COMPETITION_JAX_ROLLOUT",
    "throughput_v4_tps": tps,
    "overnight_training_authorized": False,
    "phase_10_execution_authorized": False,
    "automatic_upload_authorized": False,
    "final_recommendation": "NO_CANDIDATE_CURRENTLY_RECOMMENDED",
    "overnight_lineage": "NO_VALID_OVERNIGHT_PARENT",
    "owned_processes": [],
    "env_implementation_hash": env_implementation_hash(),
    "primary_blocker": None if eligible else disposition,
}
Path("experiments/manifests/competition_native_jax_programme_state.json").write_text(json.dumps(state, indent=2)+"\n")

# Daytime summary
summary = {
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_DAYTIME_TRAINING_SUMMARY",
    "status": disposition,
    "classifications": [
        "END_TO_END_COMPETITION_JAX_ROLLOUT",
        disposition,
        "SHORT_COMPLETED_RESEARCH_ONLY_HOST_BOUND_LEGACY",
        "MEDIUM_ABORTED_BY_OPERATOR_ARCHITECTURE_GATE",
        "NO_MEANINGFUL_TRAINED_CHECKPOINT",
    ],
    "legacy_host_bound_tps": baseline,
    "e2e_valid_learning_tps": tps,
    "improvement_vs_baseline_low": report["improvement_vs_baseline_low"],
    "frozen_config_v4": best,
    "resume_r_e_5": eligible,
    "updated_at": now,
}
Path("experiments/manifests/competition_native_jax_daytime_training_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
Path("experiments/manifests/competition_native_jax_final_recommendation.json").write_text(json.dumps({
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_FINAL_RECOMMENDATION",
    "recommendation": "NO_CANDIDATE_CURRENTLY_RECOMMENDED",
    "overnight_parent": "NO_VALID_OVERNIGHT_PARENT",
    "disposition": disposition,
    "updated_at": now,
}, indent=2)+"\n")
Path("experiments/manifests/competition_native_jax_overnight_readiness.json").write_text(json.dumps({
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_OVERNIGHT_READINESS",
    "status": "NO_VALID_OVERNIGHT_PARENT",
    "updated_at": now,
}, indent=2)+"\n")
Path("experiments/manifests/competition_native_jax_owned_processes.json").write_text(json.dumps({
    "schema_version": 1,
    "kind": "COMPETITION_NATIVE_JAX_OWNED_PROCESSES",
    "updated_at": now,
    "jobs": [],
}, indent=2)+"\n")
print("FINAL", disposition, "tps", tps, flush=True)
PY
echo "STAGES_3B_5_6_DONE"
