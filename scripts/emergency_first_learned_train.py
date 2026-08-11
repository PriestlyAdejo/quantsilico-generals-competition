"""Train student_emb96 from annotated sequence dataset (Stage B / first learned)."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from generals_bot.competition_native_jax.constants import ACTION_DIM
from generals_bot.competition_native_jax.policy import save_weights
from generals_bot.competition_native_jax.student_transformer_jax import (
    forward_student_batch,
    init_student_params,
    student_params_to_numpy_weights,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / "quantsilico-runtime" / "emergency_rolling_v1"
DEFAULT_DATA = RUNTIME / "distill" / "dataset_stage_b" / "sequences.npz"
OUT = RUNTIME / "distill" / "student_v1"


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


def _legal_softmax(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    neg = jnp.asarray(-1e9, dtype=logits.dtype)
    masked = jnp.where(mask, logits, neg)
    return jax.nn.softmax(masked, axis=-1)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-minutes", type=float, default=75.0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    args = ap.parse_args()

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif args.device == "gpu":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("JAX_PLATFORMS", None)

    data_path = Path(args.data)
    if not data_path.is_file():
        print("MISSING_DATASET", data_path)
        return 2
    raw = np.load(data_path)
    spatial = jnp.asarray(raw["spatial"].astype(np.float32))
    global_vec = jnp.asarray(raw["global_vec"].astype(np.float32))
    t_logits = jnp.asarray(raw["teacher_logits"].astype(np.float32))
    t_value = jnp.asarray(raw["teacher_value"].astype(np.float32))
    t_act = jnp.asarray(raw["teacher_action"].astype(np.int32))
    masks = jnp.asarray(raw["legal_mask"].astype(bool))
    valid = jnp.asarray(raw["valid"].astype(np.float32))
    # supervised steps: skip burn-in first 8 when valid
    BURN = 8
    T = spatial.shape[1]
    step_w = jnp.ones((T,), dtype=jnp.float32)
    step_w = step_w.at[:BURN].set(0.0)

    N = int(spatial.shape[0])
    key = jax.random.PRNGKey(0)
    params = init_student_params(key)
    meta = params["meta"]
    train_params = {k: v for k, v in params.items() if k != "meta"}
    opt = optax.adam(args.lr)
    opt_state = opt.init(train_params)

    def loss_fn(tp, idx):
        p = {**tp, "meta": meta}
        # idx: [batch]
        spat = spatial[idx]  # B,T,C,H,W
        glob = global_vec[idx]
        # flatten time into batch for feedforward student (ObsMemory already baked into spatial channels)
        Bsz, Tt = spat.shape[0], spat.shape[1]
        spat_f = spat.reshape(Bsz * Tt, *spat.shape[2:])
        glob_f = glob.reshape(Bsz * Tt, -1)
        o = forward_student_batch(p, spat_f, glob_f)
        logits = o["flat_logits"].reshape(Bsz, Tt, ACTION_DIM)
        vlogits = o["value_logits"].reshape(Bsz, Tt, -1)
        m = masks[idx]
        v = valid[idx] * step_w[None, :]
        pi_s = _legal_softmax(logits, m)
        pi_t = _legal_softmax(t_logits[idx], m)
        kl = jnp.sum(pi_t * (jnp.log(pi_t + 1e-8) - jnp.log(pi_s + 1e-8)), axis=-1)
        oh = jax.nn.one_hot(t_act[idx], ACTION_DIM)
        ce = -jnp.sum(oh * jnp.log(pi_s + 1e-8), axis=-1)
        vloss = jnp.mean((vlogits - t_value[idx]) ** 2, axis=-1)
        wsum = jnp.maximum(jnp.sum(v), 1.0)
        loss = (jnp.sum(v * (0.5 * kl + 1.0 * ce + 0.25 * vloss))) / wsum
        return loss, {"kl": jnp.sum(v * kl) / wsum, "ce": jnp.sum(v * ce) / wsum}

    @jax.jit
    def train_step(tp, opt_state, idx):
        (loss, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(tp, idx)
        updates, opt_state = opt.update(grads, opt_state, tp)
        tp = optax.apply_updates(tp, updates)
        return tp, opt_state, loss, parts

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    history = []
    for step in range(args.steps):
        if (time.perf_counter() - t0) > args.max_minutes * 60:
            break
        key, k1 = jax.random.split(key)
        idx = jax.random.randint(k1, (args.batch,), 0, N)
        train_params, opt_state, loss, parts = train_step(train_params, opt_state, idx)
        if step % 10 == 0 or step == args.steps - 1:
            row = {
                "step": step,
                "loss": float(loss),
                "kl": float(parts["kl"]),
                "ce": float(parts["ce"]),
                "elapsed_s": time.perf_counter() - t0,
            }
            history.append(row)
            print(row, flush=True)

    params = {**train_params, "meta": meta}
    w = student_params_to_numpy_weights(params)
    export = OUT / "student_emb96_v1.npz"
    save_weights(export, w)
    report = {
        "schema_version": 1,
        "kind": "EMERGENCY_STUDENT_TRAIN_V1",
        "export": str(export),
        "steps_ran": len(history),
        "history_tail": history[-5:],
        "data": str(data_path),
        "n_sequences": N,
        "device": str(jax.devices()[0]),
        "elapsed_s": time.perf_counter() - t0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Trained on real teacher-annotated sequences (not Stage A synthetic).",
    }
    _atomic_write_json(OUT / "train_report.json", report)
    _atomic_write_json(ROOT / "experiments/manifests/emergency_student_train_v1.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
