# Competition-native JAX — end-to-end rebuild terminal

## END-TO-END COMPETITION JAX ROLLOUT ACHIEVED

Architecture classification: `END_TO_END_COMPETITION_JAX_ROLLOUT`

Promotion disposition: `END_TO_END_JAX_CORRECT_BUT_TOO_SLOW`

### Headline metrics

| Item | Value |
|------|-------|
| Old host-bound valid TPS | 0.151–0.185 |
| New V4 valid TPS (frozen) | **9.51** |
| Improvement vs baseline low | **~63×** |
| Frozen config | num_envs=32, rollout_len=16 |
| Peak VRAM | 5435 MiB |
| Host RSS | ~4.7 GB |
| 90m supported transitions | ~51k (need ≥100k for meaningful short) |
| Min promotion TPS | 20 |
| Eligible for R-E.5 | **false** |

### Evidence

- Stage 3A: 24 pytest passed; 10k pass + 2k legal differential transitions, 0 mismatches
- Stage 3B: 100k pass + 5k legal; 1000 games; 0 mismatches
- Stage 4: fused `vmap`+`lax.scan` collect; v4_dev best ~3.6 TPS (dev only)
- Stage 5: end-to-end audit → `END_TO_END_COMPETITION_JAX_ROLLOUT`
- Stage 6: final ladder_v4 distinct from v4_dev

### What was built

- Official MIT JAX primitives (`game.step`, build_castles, deathtouch) in compiled hot path
- QuantSilico wrappers: obs/memory, exact 3970 legal mask, fused scan collect, native `forward_batch`
- Host Python timestep collector demoted
- Env implementation hash on checkpoints: `c63394113f2abd67d206de8df6dd88e3fa699a35088603704146d9253c02b4dd`
- Host-bound short/medium optimiser lineage **not** resumed

### Hard stop

`AWAITING_PRE_OVERNIGHT_OPERATOR_REVIEW`

- overnight / upload / Phase 10: **false**
- `NO_CANDIDATE_CURRENTLY_RECOMMENDED`
- `NO_VALID_OVERNIGHT_PARENT`
- Did **not** start R-E.5 smoke / short / medium (promotion TPS gate failed)
- Did **not** fall back to host-bound path

### Exact next human action

Profile remaining JAX bottlenecks (obs/mask cost, scan compile, batch size) to reach ≥20 valid-learning TPS, then re-run Stage 6 and only then authorise R-E.5.
