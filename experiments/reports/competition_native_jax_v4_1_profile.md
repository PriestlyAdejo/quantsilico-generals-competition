# V4.1 component profile

Clean baseline TPS (batched GAE): **55.0119**
Parent V4: 9.51

## Ranked bottlenecks

- `ppo_update`: 92.5%
- `legal_mask_stepish`: 68.2%
- `observe_stepish`: 31.6%
- `gae_python_loop_pre_fix`: 13.6%
- `rollout_collect`: 6.3%
- `gae_batched`: 1.1%
- `transformer_stepish`: 0.2%
- `env_step_stepish`: 0.0%

## Raw times

- `reset_batch_s`: 0.3196323250000006
- `observe_both_seats_s`: 6.304112630999995
- `legal_mask_both_seats_s`: 13.589053596000014
- `transformer_forward_batch_5_s`: 0.16408508999998617
- `env_step_batch_10_s`: 0.046148396999996066
- `gae_python_env_loop_s`: 13.981917661000011
- `python_gae_loop_confirmed`: True
- `gae_batched_scan_s`: 1.1511465320000127
- `gae_batch_speedup`: 12.146079818967701
- `ppo_update_s`: 95.163414577
- `ema_update_s`: 0.07555859999996528
- `full_rollout_collect_s`: 6.494393967999997
- `rollout_tps`: 78.83722523191409
