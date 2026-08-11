# Phase 9F-R entropy diagnosis (recorded vs update-time)

Created: 2026-08-05T10:18:56.244356+00:00

## Summary

Entropy collapses along with the behaviour policy becoming effectively deterministic on `PASS`. After collapse, the PPO ratio gate becomes trivially satisfied because the selected action equals PASS, making update-time support restriction irrelevant for log-prob.

## Evidence table (means)

- BC_init: step-zero-pass 3/32 (rate 9.38%); H_behaviour=0.679772; H_update_mask_same=0.0332807; H_update_impl=0.103221; P(pass)=0.185885; update_support_unique=[1, 2]
- CONTROL_update10: step-zero-pass 2/32 (rate 6.25%); H_behaviour=0.973901; H_update_mask_same=0.156798; H_update_impl=0.132447; P(pass)=0.324853; update_support_unique=[1, 2]
- CONTROL_update50: step-zero-pass 32/32 (rate 100.00%); H_behaviour=0.000148537; H_update_mask_same=0; H_update_impl=0; P(pass)=0.999988; update_support_unique=[1]
- CONTROL_final: step-zero-pass 32/32 (rate 100.00%); H_behaviour=2.14142e-05; H_update_mask_same=0; H_update_impl=0; P(pass)=NA; update_support_unique=[1]
- CURRICULUM_update10: step-zero-pass 2/32 (rate 6.25%); H_behaviour=1.00224; H_update_mask_same=0.190099; H_update_impl=0.158787; P(pass)=0.31119; update_support_unique=[1, 2]
- CURRICULUM_final: step-zero-pass 32/32 (rate 100.00%); H_behaviour=1.00522e-09; H_update_mask_same=0; H_update_impl=0; P(pass)=1; update_support_unique=[1]

## Where the 0.0 entropy comes from

In `src/generals_bot/training/rollout.py::ppo_update_from_fragment`, update-time entropy is computed from a Categorical distribution whose logits have been masked down to the support `{acts_t, PASS}` only. This will yield entropy 0.0 whenever `acts_t == PASS` (support size 1), which is exactly what happens in the late PPO checkpoints.

