# Competition-native JAX architecture audit

**Classification: `NUMPY_PROTOTYPE_ONLY`**

Audited commit: `ba0e1b6` / resume start `3b776d9`

## Grep evidence

Under `src/generals_bot/competition_native_jax` and `train/competition_native_jax`:
zero matches for `jax.jit`, `jax.vmap`, `lax.scan`, `value_and_grad`, `jax.random`,
`jax.device_put`, or `optax`.

## Checklist

| Area | Required | Status |
|------|----------|--------|
| A. Model JAX PyTree / JIT forward | Yes | MISSING (NumPy transformer only) |
| B. Self-play batched JAX rollout | Yes | MISSING (Python/NumPy paired_eval loop) |
| C. JAX GAE | Yes | MISSING (NumPy GAE helper only) |
| D. JAX PPO + Optax | Yes | MISSING (NumPy ratio helper only) |
| E. Device-resident rollout storage | Yes | MISSING |

## Allowed role of existing NumPy code

CPU deployment / parity reference only. Not the canonical learner.

## Remediation

Implement pure-JAX + Optax core (Resume R-C) before any GPU training claims.
