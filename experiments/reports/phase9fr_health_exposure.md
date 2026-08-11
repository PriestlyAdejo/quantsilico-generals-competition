# Phase 9F-R optimisation health & exposure (from overnight PPO manifests)

created_at_utc: 2026-08-05T10:20:00Z

## RL_CONTROL arm

- Checkpoint: `experiments/phase9f_overnight_ppo/rl_control/final.json`
- Updates: 144
- Synchronous PPO: true
- Final history (last update observed):
  - `entropy`: 0.0 (update 143)
  - `policy_loss`: ~1.75e-07
  - `value_loss`: ~0.00125
  - `loss`: ~0.000627
  - `grad_norm`: ~0.020
- Exposure / completion:
  - `actor_meta.completed_games`: 1
  - `actor_meta.turn`: 34219
  - `actor_meta.map_seed`: 43
  - Opponent during RL_CONTROL rollout: `pass`

## RL_CURRICULUM arm

- Checkpoint: `experiments/phase9f_overnight_ppo/rl_curriculum/final.json`
- Updates: 127
- Synchronous PPO: true
- Final history (last update observed):
  - `entropy`: 0.0 (update 126)
  - `policy_loss`: ~7.45e-08
  - `value_loss`: ~0.00158
  - `loss`: ~0.000790
  - `grad_norm`: ~0.010
- Exposure / completion:
  - `actor_meta.completed_games`: 0
  - `actor_meta.turn`: 32512
  - `actor_meta.map_seed`: 42
  - Opponent during RL_CURRICULUM rollout: `official_expander`

## Health interpretation (diagnostic-only)

Both arms “stabilise” late into near-zero policy loss / value loss with `entropy=0.0`. Combined with the step-zero audits (where the behaviour policy becomes deterministic on `PASS`), this indicates the optimiser has converged to a degenerate low-entropy policy, not a stable high-competence stochastic policy.

