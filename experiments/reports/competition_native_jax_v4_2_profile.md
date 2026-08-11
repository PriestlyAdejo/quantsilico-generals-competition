# V4.2 component profile

Clean one-update valid-learning TPS (profiler-disabled): **97.02**

GAE: device-resident batched reverse `lax.scan` (no Python env loop).
Reset: device competition reset pool reused across collects.

| Component | seconds | % |
|---|---:|---:|
| ppo_full_batch_update | 52.0618 | 81.2 |
| complete_rollout_collect | 11.2330 | 17.5 |
| gae_batch_scan | 0.7836 | 1.2 |
