# Phase 9F-R PPO ratio semantics audit (step-zero)

Created: 2026-08-05T10:05:37.954281+00:00

Checkpoint: `experiments\phase9f_overnight_ppo\rl_control\update_10.json`

## Step-zero gate

- Pass transitions: 2/32 (6.250%)
- Numerical tolerance: CUDA_float32 (abs(logp_delta) <= 0.0001; abs(ratio-1) <= 0.0001)

## Entropy diagnosis (means)

- Behaviour entropy (FULL legal mask): 0.973901
- Update-mask entropy on same logits: 0.156798
- Update-impl entropy (ppo_update_from_fragment semantics): 0.132447

## Support mismatch evidence

- Mean legal support size: 8.12 (min=1, max=13)
- Update support sizes unique (chosen vs PASS only): [1, 2]
- PASS action fraction: 31.25%

## Interpretation constraints

Old_logp uses persistent sequential hidden/cell_mem evolution. New_logp_impl uses update semantics which reset hidden/cell_mem per-transition batch element. Therefore ratio failure can be caused by (1) support mismatch, (2) hidden/state reconstruction mismatch, or (3) stochastic strategic-mixture sampling differences.

## Samples (first few transitions)

See JSON for detailed per-step numbers.
