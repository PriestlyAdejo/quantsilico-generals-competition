# Hot-path profile

Architecture: `END_TO_END_OFFICIAL_JAX_ROLLOUT`

## Measured times

- `env_step_batch_compile_s`: 14.645397
- `env_step_batch_20_s`: 0.100245
- `env_transition_per_env_ms`: 0.626529
- `obs_mask_batch_10_s`: 2.885524
- `obs_mask_per_env_ms`: 36.069050
- `policy_forward_single_compile_s`: 50.468532
- `policy_forward_single_20_s`: 35.589123
- `policy_forward_batch_compile_s`: 50.601605
- `policy_forward_batch_20_s`: 0.171380
- `policy_batch_speedup_vs_naive`: 1661.291597
- `full_rollout_2steps_s`: 264.553866
- `full_rollout_8steps_s`: 40.963972
- `full_rollout_tps_estimate`: 1.562349

## Approximate shares of 8-step rollout

- `env_step_share_pct`: 0.1%
- `obs_mask_share_pct`: 5.6%
- `policy_batch_share_pct`: 0.2%

Full-rollout TPS estimate (8 steps × 8 envs): **1.562**

Baseline complete-loop TPS: 0.151–0.185

