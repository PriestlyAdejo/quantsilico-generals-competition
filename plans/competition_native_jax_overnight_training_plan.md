# Competition-native JAX overnight training plan (write-only)

**Status:** overnight training **NOT authorised**  
**Parent classification:** `NO_VALID_OVERNIGHT_PARENT`  
**OVERNIGHT_READY:** false

## Why no parent

| Lineage | Result |
|---------|--------|
| Smoke NumPy prototype raw/EMA | Research-only; not competitively trained; p99 ≫ 150 ms |
| Short daytime | SKIPPED_WITH_REASON (no JAX GPU / BLOCKED_COMPUTE) |
| Medium daytime | SKIPPED_WITH_REASON |
| Controlled / student | Not produced |

## Minimum remediation before another overnight proposal

1. Install/verify CUDA-enabled JAX in the training environment (`jax_gpu=true` with parameters/rollouts/update on device).
2. Re-run GPU execution gate + throughput ladder; freeze transition budgets via `⌊0.85×TPS×seconds⌋`.
3. Pass early deployment with a deployable student or distilled runtime under 150 ms p99.
4. Complete short daytime (and medium if justified) under frozen evaluation protocol.
5. Produce `BASE_DAYTIME_UPLOAD_READY` or an upload-qualified controlled/student package.

## Proposed next human actions

1. Review daytime artefacts under `experiments/manifests/competition_native_jax_*.json`.
2. Decide whether to authorise GPU JAX environment repair + another daytime run.
3. Do **not** start overnight until a valid parent is selected under a future authorisation.
