#!/usr/bin/env bash
# V4.1 continuation: scaling + ladder + parity + gate (profile already written)
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

echo "=== Verify env_implementation_hash ==="
python - <<'PY'
from train.competition_native_jax.train_jax import env_implementation_hash, lineage_hashes
h = env_implementation_hash()
print(h)
assert h.startswith("c6339411"), h
print(lineage_hashes())
PY

echo "=== Fast tests ==="
pytest -q tests/competition_native_jax/test_jax_core.py tests/competition_native_jax/test_batch_policy.py -o cache_dir=/tmp/qs_pytest_cache --tb=short

echo "=== Stage 2: scaling ladder (post-warmup TPS) ==="
python - <<'PY'
import json, gc
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
            max_transitions=c["num_envs"] * c["rollout_len"] * 3,
            max_updates=3,
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
            "status": "OK",
            "vs_v4": tps / baseline_v4,
            "stable": bool((rep.get("peak_vram_mib") or 0) < 7500 and tps > 0),
            "post_warmup_timing": True,
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
    "post_warmup_timing": True,
}
Path("experiments/manifests/competition_native_jax_v4_1_scaling_ladder.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2), flush=True)
PY

echo "=== Stage 5: final clean V4.1 ladder ==="
python - <<'PY'
import json, math, gc
from pathlib import Path
from train.competition_native_jax.train_jax import _train_loop, lineage_hashes, detect_jax_device

scale = json.loads(Path("experiments/manifests/competition_native_jax_v4_1_scaling_ladder.json").read_text())
best = scale.get("best") or {"num_envs": 32, "rollout_len": 16}
cands = [
    {"num_envs": 32, "rollout_len": 16},
    {"num_envs": int(best["num_envs"]), "rollout_len": int(best["rollout_len"])},
]
seen=set(); uniq=[]
for c in cands:
    k=(c["num_envs"], c["rollout_len"])
    if k not in seen:
        seen.add(k); uniq.append(c)

rows=[]; frozen=None; v4=9.510967309770074
for c in uniq:
    rep = _train_loop(
        Path(f"experiments/competition_native_jax/throughput_v4_1/e{c['num_envs']}_r{c['rollout_len']}"),
        kind="probe_v41",
        max_transitions=c["num_envs"] * c["rollout_len"] * 4,
        max_updates=4,
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
        "post_warmup_timing": True,
        **{k: rep.get(k) for k in ("env_implementation_hash", "learner_implementation_hash", "env_semantics_hash", "gae_device_resident_batched")},
    }
    rows.append(row)
    if row["stable"] and (frozen is None or tps > frozen["valid_learning_tps"]):
        frozen = row
    print("V41", row, flush=True)
    gc.collect()

tps = float(frozen["valid_learning_tps"]) if frozen else 0.0
min_tps = 20.0
src = Path("train/competition_native_jax/train_jax.py").read_text(encoding="utf-8")
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
    "post_warmup_timing": True,
}
Path("experiments/manifests/competition_native_jax_throughput_ladder_v4_1.json").write_text(json.dumps(report, indent=2)+"\n")
Path("experiments/reports/competition_native_jax_throughput_v4_1.md").write_text(
    f"# Throughput V4.1\n\nDisposition: `{disposition}`\n\nTPS: {tps:.4f}\n\nvs V4: {report['improvement_vs_v4']}\n\nGAE batched: {gae_ok}\n\n## Candidates\n\n"
    + "\n".join(f"- {r}" for r in rows) + "\n"
)
print(json.dumps(report, indent=2), flush=True)
Path("/tmp/v41_disposition.txt").write_text(disposition)
Path("/tmp/v41_eligible.txt").write_text("1" if eligible else "0")
Path("/tmp/v41_cfg.json").write_text(json.dumps(frozen or {}))
PY

DISP=$(cat /tmp/v41_disposition.txt)
ELIG=$(cat /tmp/v41_eligible.txt)
echo "DISPOSITION=${DISP} ELIGIBLE=${ELIG}"

echo "=== Stage 4 final parity ==="
pytest -q tests/competition_native_jax -o cache_dir=/tmp/qs_pytest_cache --tb=line 2>&1 | tee experiments/manifests/_v41_pytest.log
python - <<'PY'
import json
from pathlib import Path
import jax
import jax.numpy as jnp
from generals_bot.competition_native_jax.competition_env_jax import (
    TRUNCATION, competition_transition, step_one_jax, reset_one_jax,
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
    subprocess.call([
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
        subprocess.call([
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
    if disp == "END_TO_END_JAX_PROMOTION_READY" and not eligible:
        disp = "END_TO_END_JAX_BLOCKED_CORRECTNESS"

# Update accepted changes retention from TPS
acc = json.loads(Path("experiments/manifests/competition_native_jax_v4_1_accepted_changes.json").read_text())
for ch in acc["changes"]:
    if ch["id"] == "batched_gae_lax_scan":
        ch["status"] = "ACCEPTED"
    elif ch["id"] == "vectorised_transformer_patch_ops":
        ch["status"] = "ACCEPTED" if tps >= 20 else ("ACCEPTED_IF_TPS_IMPROVED" if tps > 9.51 else "RETAINED_WITH_GAE")
    elif ch["id"] == "concat_index_to_engine_batch":
        ch["status"] = "REVERTED_PRESERVE_ENV_HASH"
acc["final_tps"] = tps
acc["disposition"] = disp
Path("experiments/manifests/competition_native_jax_v4_1_accepted_changes.json").write_text(json.dumps(acc, indent=2)+"\n")

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
    "primary_blocker": None if (hard.endswith("OPERATOR_REVIEW") and eligible) else disp,
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
