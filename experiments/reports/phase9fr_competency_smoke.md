# Phase 9F-R competency smoke (diagnostic-only)

created_at_utc: 2026-08-05T10:22:00Z

This smoke is intentionally *not* a tournament/protocol claim. It only checks whether the learnt PPO policies preserve non-degenerate stochasticity relative to the BC init, using the measured policy entropy channel from the step-zero audit.

## Behaviour stochasticity comparison (FULL legal mask entropy)

- BC init (`experiments/phase9f_cnn_ranker_v1/checkpoints/bc/model.json`):
  - `H_behaviour_full_legal_mean`: 0.679772
  - `P(pass)` under FULL legal mask: 0.185885 (non-trivial PASS probability)

- PPO CONTROL final (`experiments/phase9f_overnight_ppo/rl_control/final.json`):
  - `H_behaviour_full_legal_mean`: 2.1414e-05
  - `P(pass)` under FULL legal mask: ~1.0

- PPO CURRICULUM final (`experiments/phase9f_overnight_ppo/rl_curriculum/final.json`):
  - `H_behaviour_full_legal_mean`: ~1.005e-09
  - `P(pass)` under FULL legal mask: 1.0

## Smoke verdict

On the measured stochasticity channel, both PPO arms collapse to a near-deterministic `PASS`-only behaviour. Therefore this run is **DIAGNOSTICALLY_BELOW_BC** (competence not preserved), consistent with the entropy collapse telemetry.

