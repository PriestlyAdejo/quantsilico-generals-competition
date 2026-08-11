# V4.2 Max-Utilisation — operator terminal

## Verdict

- V4.1 superseded safely (`RUNNING_SCALING` orphan stopped via STOP_REQUEST→SIGINT).
- Systems disposition: `END_TO_END_JAX_SYSTEMS_PROMOTION_READY`
- Selected config: `num_envs=32`, `rollout_len=32`, `reset_pool_size=4096`
- Final clean valid-learning TPS: **189.71** (V4 was 9.51 ≈ **20×**)
- R-E.5 smoke: **COMPLETED** — 100352 transitions, 98 updates, ~136 TPS wall
- Learning-efficiency track: **deferred** (systems already qualified)
- Upload / overnight / Phase 10: **false**
- Final status: `AWAITING_PRE_OVERNIGHT_OPERATOR_REVIEW`
- Recommendation: `NO_CANDIDATE_CURRENTLY_RECOMMENDED`
- Overnight parent: `NO_VALID_OVERNIGHT_PARENT`

## V4.1 supersession

- Classification: orphaned WSL continue still in `RUNNING_SCALING` despite dead Windows supervisor
- Stop: STOP_REQUEST (ineffective during compile) → SIGINT → dead; GPU freed
- Preserved: V4 frozen ladder/audit; V4.1 profile; partial scaling probes (best partial 93.8 TPS @ 40×16, not promoted)

## Matched benchmarks (pool reused)

| Bench | Rate |
|---|---:|
| A env transition | 67682 TPS |
| B env+obs+legal | 65334 TPS |
| C complete rollout | 173 TPS |
| D PPO samples/s | 348 |
| E valid-learning | 108 TPS |

First collapse: **C_rollout** (transformer/legal path), not the environment.

## Profile (attribution)

- PPO full-batch update ~81%
- Rollout collect ~18%
- Device GAE ~1%
- GAE: device reverse `lax.scan`; bootstrap from post-scan value
- Reset: device competition pool; **no in-scan map gen**; reused (refresh every 10 updates)

## Systems changes accepted

- Device competition reset pool (canonical `reset_one_jax` semantics)
- Pool reuse outside timed loop (critical TPS fix)
- Static geometry module (lookups/tests; hot-path action codec remains inline to avoid tracer leaks)
- GAE bootstrap fix
- Optional PPO gradient accumulation (one Optax step; semantics preserved)
- BF16 / bitpack masks: deferred pending further measurement

## Autotune

- Peak and promotion-eligible selected: **32×32×4096**
- Promotion budget OK: ≥100k transitions, ≥100 updates, games gate in 90 minutes

## Parity

- Stage 3A: **passed** (14 tests) after tracer-leak fix
- Full Stage 3B (≥1000 games / ≥100k differential transitions): **not yet run** — required before overnight/package confidence

## R-E.5

- Config from promotion-eligible systems candidate
- Completed at transition budget (100352 / 30m wall not binding)
- Artefacts: `experiments/manifests/competition_native_jax_v4_2_smoke_r_e5.json`

## Safety

- owned_processes: empty
- overnight / upload / Phase 10 remain false

## Exact next human action

1. Review R-E.5 smoke metrics/checkpoints under `experiments/competition_native_jax/v4_2_smoke_r_e5/`.
2. Authorise full Stage 3B parity on the V4.2 env implementation hash before R-F packaging.
3. If smoke looks healthy, authorise R-E.6 short (≤90m) then conditional R-E.7 medium.
4. Do **not** upload or start overnight yet.
