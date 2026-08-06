# Competition-native JAX architecture audit

**Classification: `JAX_CORE_IMPLEMENTED_GPU_UNVERIFIED`**
(prior at resume start: `NUMPY_PROTOTYPE_ONLY`)

## Present after R-C

| Area | Status | Path |
|------|--------|------|
| A. Model JAX PyTree | PRESENT | `src/generals_bot/competition_native_jax/transformer_jax.py` |
| B. Inference masked log-softmax | PRESENT | `inference_jax.py` (+ NumPy reference) |
| C. Self-play rollout | PRESENT (JAX policy; env step still Python) | `train/competition_native_jax/rollout_selfplay_jax.py` |
| D. JAX GAE (`lax.scan`) | PRESENT | `gae_jax.py` |
| E. JAX PPO + Optax | PRESENT | `ppo_jax.py` |
| F. EMA | PRESENT | `ema_jax.py` |
| G. Train entry | PRESENT | `train_jax.py` |

## GPU

`GPU_JAX_VERIFIED=false`. Windows JAX is CPU-only. Ubuntu WSL2 is registered but first-user OOBE blocks non-interactive CUDA JAX bootstrap (`AWAITING_OPERATOR_ACTION`, not a CUDA failure).

## Allowed role of NumPy code

CPU deployment / parity / student-shape feasibility reference only. Not the canonical learner.
