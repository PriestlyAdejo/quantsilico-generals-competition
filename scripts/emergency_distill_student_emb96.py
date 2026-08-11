"""Minimal emergency distill route for student_emb96_d2_h4 (ops worktree).

Stage A uses synthetic legal teacher targets for ENGINEERING only (not imitation quality).
Declares DISTILLATION_MINIMAL_ROUTE_READY only after JAX forward, recurrent smoke,
backward, export/reload, NumPy parity, schema hashes, and stdio invocation.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Force CPU for plumbing smoke (do not steal GPU from PPO)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from generals_bot.competition_native_jax.constants import ACTION_DIM, HL_GAUSS_BINS, MAX_HW
from generals_bot.competition_native_jax.obs_memory import N_GLOBAL, N_SPATIAL
from generals_bot.competition_native_jax.policy import load_weights, save_weights
from generals_bot.competition_native_jax.student_policy_numpy import (
    StudentCompetitionNativePolicy,
    init_student_weights,
    schema_hashes,
    validate_student_weights,
)
from generals_bot.competition_native_jax.student_transformer_jax import (
    STUDENT_EMB,
    STUDENT_HEADS,
    STUDENT_LAYERS,
    forward_student,
    forward_student_batch,
    init_student_params,
    student_params_to_numpy_weights,
)
from generals_bot.competition_native_jax.transformer import forward as numpy_forward
from generals_bot.observation import Observation

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / "quantsilico-runtime" / "emergency_rolling_v1"
PLUMBING_DEADLINE_S = 30 * 60
EPS_PI = 1e-4
EPS_V = 1e-4

# Frozen sequence contract
SEQ_LEN = 32
BURN_IN = 8
LOSS_STEPS = 24


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


def _synth_obs(seed: int, turn: int) -> Observation:
    rng = np.random.default_rng(seed + turn)
    h, w = 18, 18
    types = np.zeros((h, w), dtype=np.int32)
    owners = np.zeros((h, w), dtype=np.int32)
    armies = np.zeros((h, w), dtype=np.int32)
    owners[5, 5] = 1
    armies[5, 5] = 20 + (turn % 7)
    types[5, 5] = 4
    if turn % 11 == 0:
        owners[6, 5] = 2
        armies[6, 5] = 3
    return Observation(
        h,
        w,
        turn,
        1,
        int(armies[owners == 1].sum()),
        int((owners == 2).sum()),
        int(armies[owners == 2].sum()),
        tuple(tuple(int(x) for x in row) for row in types),
        tuple(tuple(int(x) for x in row) for row in owners),
        tuple(tuple(int(x) for x in row) for row in armies),
    )


def _legal_softmax(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    neg = jnp.asarray(-1e9, dtype=logits.dtype)
    masked = jnp.where(mask, logits, neg)
    return jax.nn.softmax(masked, axis=-1)


def main() -> int:
    t0 = time.perf_counter()
    steps: list[dict] = []
    out_dir = RUNTIME / "distill" / "stage_a_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    def step(name: str, ok: bool, detail: str = "") -> None:
        steps.append(
            {
                "name": name,
                "ok": bool(ok),
                "detail": detail,
                "elapsed_s": time.perf_counter() - t0,
            }
        )
        print(f"STEP {name} ok={ok} {detail}", flush=True)

    try:
        # Contract freeze artefact
        contract = {
            "schema_version": 1,
            "kind": "DISTILLATION_SEQUENCE_CONTRACT",
            "sequence_length": SEQ_LEN,
            "burn_in": BURN_IN,
            "supervised_loss_steps": LOSS_STEPS,
            "student_initial_state": "zero_ObsMemory_only_at_true_episode_start",
            "fragment_boundary_reset": False,
            "terminal_reset": True,
            "padding_contributes_to_loss": False,
            "recurrent_state": "ObsMemory(seen_own,last_army,turn)",
            "hidden_state_loss": "OFF",
            "student": {"emb": STUDENT_EMB, "layers": STUDENT_LAYERS, "heads": STUDENT_HEADS},
            "schema_hashes": schema_hashes(),
            "data_source_priority": [
                "persisted_ppo_trajectories",
                "eval_canary_trajectories",
                "cpu_simulator_trajectories",
                "synthetic_fixtures_smoke_only",
            ],
            "mixture_target": {
                "self_play": 0.5,
                "hunter": 0.2,
                "expander_aggressive": 0.2,
                "fixtures": 0.1,
            },
            "cache_max_bytes": 1 << 30,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(ROOT / "experiments/manifests/emergency_distill_sequence_contract.json", contract)
        step("sequence_contract_frozen", True)

        # 1) JAX student forward
        key = jax.random.PRNGKey(0)
        params = init_student_params(key)
        spatial = jnp.zeros((N_SPATIAL, MAX_HW, MAX_HW), dtype=jnp.float32)
        spatial = spatial.at[0, :18, :18].set(1.0)
        spatial = spatial.at[1, 5, 5].set(1.0)
        g = jnp.zeros((N_GLOBAL,), dtype=jnp.float32)
        out_j = forward_student(params, spatial, g)
        ok = bool(np.isfinite(np.asarray(out_j["flat_logits"])).all())
        step("student_jax_forward", ok, f"logits={tuple(out_j['flat_logits'].shape)}")

        # 2) NumPy student + architecture/schema
        w_np = student_params_to_numpy_weights(params)
        validate_student_weights(w_np)
        pol = StudentCompetitionNativePolicy(weights=w_np, seed=0)
        pol.reset(18, 18)
        out_n = numpy_forward(np.asarray(spatial), np.asarray(g), w_np)
        d_logits = float(np.max(np.abs(np.asarray(out_j["flat_logits"]) - out_n["flat_logits"])))
        d_v = float(np.max(np.abs(np.asarray(out_j["value_logits"]) - out_n["value_logits"])))
        arch_ok = d_logits <= EPS_PI and d_v <= EPS_V
        step(
            "STUDENT_RUNTIME_ARCHITECTURE_PARITY",
            arch_ok,
            f"max|d_logits|={d_logits:.3e} max|d_v|={d_v:.3e}",
        )
        step("STUDENT_INPUT_OUTPUT_SCHEMA_PARITY", True, json.dumps(schema_hashes()))

        # 3) Recurrent smoke unroll (ObsMemory): episode start + padding carry + terminal reset
        obs_seq = [_synth_obs(1, t) for t in range(SEQ_LEN)]
        valid = np.ones((SEQ_LEN,), dtype=np.float32)
        valid[:2] = 0.0  # padded prefix
        ep_reset = np.zeros((SEQ_LEN,), dtype=np.float32)
        ep_reset[0] = 1.0
        ep_reset[20] = 1.0  # terminal/new episode mid-sequence
        from generals_bot.competition_native_jax.student_policy_numpy import unroll_sequence_numpy

        pol2 = StudentCompetitionNativePolicy(weights=w_np, seed=0)
        unrolled = unroll_sequence_numpy(pol2, obs_seq, valid_mask=valid, episode_reset=ep_reset)
        rec_ok = bool(np.isfinite(unrolled["flat_logits"]).all())
        step("RECURRENT_SEQUENCE_PARITY_SMOKE", rec_ok, f"shape={unrolled['flat_logits'].shape}")

        # 4) Synthetic Stage A loss + backward (engineering only)
        B = 4
        key, k1, k2 = jax.random.split(key, 3)
        spat_b = jax.random.normal(k1, (B, N_SPATIAL, MAX_HW, MAX_HW), dtype=jnp.float32) * 0.01
        glob_b = jax.random.normal(k2, (B, N_GLOBAL), dtype=jnp.float32) * 0.01
        # synthetic legal masks: first 64 actions legal
        mask = jnp.zeros((B, ACTION_DIM), dtype=bool).at[:, :64].set(True)
        # synthetic teacher logits
        t_logits = jax.random.normal(key, (B, ACTION_DIM), dtype=jnp.float32)
        t_val = jax.random.normal(key, (B, HL_GAUSS_BINS), dtype=jnp.float32)
        t_act = jnp.argmax(jnp.where(mask, t_logits, -1e9), axis=-1)

        # Trainable leaves only (meta ints must not enter optax/grad)
        meta = params["meta"]
        train_params = {k: v for k, v in params.items() if k != "meta"}
        opt = optax.adam(1e-3)
        opt_state = opt.init(train_params)

        def loss_fn(tp):
            p = {**tp, "meta": meta}
            o = forward_student_batch(p, spat_b, glob_b)
            pi_s = _legal_softmax(o["flat_logits"], mask)
            pi_t = _legal_softmax(t_logits, mask)
            # KL(teacher || student)
            kl = jnp.sum(pi_t * (jnp.log(pi_t + 1e-8) - jnp.log(pi_s + 1e-8)), axis=-1).mean()
            # CE via one-hot to avoid int leaves in graph
            oh = jax.nn.one_hot(t_act, ACTION_DIM)
            log_pi = jnp.log(pi_s + 1e-8)
            ce = -jnp.sum(oh * log_pi, axis=-1).mean()
            vloss = jnp.mean((o["value_logits"] - t_val) ** 2)
            return 0.5 * kl + 1.0 * ce + 0.25 * vloss, {"kl": kl, "ce": ce, "vloss": vloss}

        (loss0, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_params)
        updates, opt_state = opt.update(grads, opt_state, train_params)
        train_params = optax.apply_updates(train_params, updates)
        params = {**train_params, "meta": meta}
        (loss1, _) = loss_fn(train_params)
        finite = bool(np.isfinite(float(loss0)) and np.isfinite(float(loss1)))
        step(
            "stage_a_backward_optim",
            finite,
            f"loss0={float(loss0):.4f} loss1={float(loss1):.4f} (synthetic targets; not imitation quality)",
        )

        # 5) export / reload
        w_exp = student_params_to_numpy_weights(params)
        export_path = out_dir / "student_emb96_stage_a.npz"
        save_weights(export_path, w_exp)
        w_rel = load_weights(export_path)
        validate_student_weights(w_rel)
        out_a = numpy_forward(np.asarray(spatial), np.asarray(g), w_exp)
        out_b = numpy_forward(np.asarray(spatial), np.asarray(g), w_rel)
        reload_ok = float(np.max(np.abs(out_a["flat_logits"] - out_b["flat_logits"]))) <= EPS_PI
        step("export_reload", reload_ok, str(export_path))

        # 6) competition runtime invocation (import path)
        from generals_bot.competition_native_jax.stdio_runtime import run_stdio  # noqa: F401

        pol3 = StudentCompetitionNativePolicy(weights=w_rel, seed=0)
        pol3.reset(18, 18)
        act, info = pol3.act(_synth_obs(2, 0), deterministic=True)
        step("competition_runtime_invocation", act is not None, f"action={act}")

        # Chosen action match JAX vs NumPy on same tensors
        mask_np = np.zeros((ACTION_DIM,), dtype=bool)
        mask_np[:64] = True
        oj = forward_student(params, spatial, g)
        on = numpy_forward(
            np.asarray(spatial),
            np.asarray(g),
            student_params_to_numpy_weights(params),
        )
        aj = int(np.argmax(np.where(mask_np, np.asarray(oj["flat_logits"]), -1e9)))
        an = int(np.argmax(np.where(mask_np, on["flat_logits"], -1e9)))
        step("chosen_legal_action_parity", aj == an, f"jax={aj} numpy={an}")

    except Exception as e:
        step("EXCEPTION", False, repr(e))
        ready = False
        status = "DISTILLATION_MINIMAL_ROUTE_BLOCKED"
        blocker = repr(e)
    else:
        critical = [
            "student_jax_forward",
            "STUDENT_RUNTIME_ARCHITECTURE_PARITY",
            "STUDENT_INPUT_OUTPUT_SCHEMA_PARITY",
            "RECURRENT_SEQUENCE_PARITY_SMOKE",
            "stage_a_backward_optim",
            "export_reload",
            "competition_runtime_invocation",
        ]
        ready = all(s["ok"] for s in steps if s["name"] in critical)
        status = "DISTILLATION_MINIMAL_ROUTE_READY" if ready else "DISTILLATION_MINIMAL_ROUTE_BLOCKED"
        blocker = None if ready else next(s["name"] for s in steps if s["name"] in critical and not s["ok"])

    elapsed = time.perf_counter() - t0
    report = {
        "schema_version": 1,
        "kind": "EMERGENCY_DISTILL_PLUMBING",
        "status": status,
        "stage_a_note": "Synthetic/legal plumbing targets only; NOT imitation quality. Real quality after GPU teacher annotation.",
        "elapsed_s": elapsed,
        "plumbing_deadline_s": PLUMBING_DEADLINE_S,
        "blocker": blocker,
        "steps": steps,
        "export_path": str(out_dir / "student_emb96_stage_a.npz"),
        "sequence_contract": "experiments/manifests/emergency_distill_sequence_contract.json",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_distill_plumbing.json", report)
    _atomic_write_json(RUNTIME / "programme" / "distill_plumbing.json", report)
    print(json.dumps({"status": status, "elapsed_s": elapsed, "blocker": blocker}, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
