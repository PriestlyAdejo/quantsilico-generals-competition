# Phase 9F-R selected primary repair (Outcome B)

selected_primary_outcome: `B — likelihood-ratio support mismatch`

## Repair objective

Make PPO’s `old_log_prob` and `new_log_prob` be computed over the *same action support* (same legal mask / same ordered candidate set, including forced-inserted actions and any deterministic PASS conventions), before any optimizer step.

This specifically addresses the defect in:

- `src/generals_bot/training/actors.py::PersistentActor.collect_fragment` (collection-time support uses `legal_mask_observation(obs)`)
- `src/generals_bot/training/rollout.py::ppo_update_from_fragment` (update-time support currently restricted to `{acts_t, PASS}`)

## Primary repair steps (do these first)

1. **Persist the exact collection-time action support per transition.**  
   In `PersistentActor.collect_fragment`, extend `FragmentTransition` and/or the rollout buffer to store:
   - `support_kind` for each transition (at minimum: `FULL_ACTION_SPACE_LEGAL_MASK` for the action-mask scheme currently used),
   - the boolean `legal_mask` (or a reconstructable canonical representation) used to compute `old_logp`,
   - (optional but recommended) `support_hash` to enforce replay equality.

2. **Use that persisted support when recomputing `new_logp` during PPO update.**  
   In `ppo_update_from_fragment`, replace the current restricted mask:
   - current (defect): mask only includes `{acts_t, PASS}`
   - repair: rebuild `masked = apply_action_mask(logits, legal_mask_from_transition)` and compute:
     - `dist = Categorical(logits=masked)`
     - `new_logp = dist.log_prob(acts_t)`
     - `entropy` on the same distribution as `new_logp`

3. **Gate training updates when support reconstruction differs.**  
   If the persisted `support_hash` (or `legal_mask` equality) fails to match, quarantine the sequence from the PPO policy-ratio loss and record the mismatch rate.

## Secondary repair steps (expected follow-up; not the primary outcome)

1. **Fix recurrent logit reconstruction per timestep.**  
   The learner currently recomputes logits with `hidden = model.initial_hidden(b)` and `cell_mem = model.initial_cell_memory(b)`, but collection-time logits were computed with sequentially evolved `actor.hidden`/`actor.cell_mem`.
   - Store pre-action recurrent state per transition (or store sequence windows and burn-in so update-time evaluation reconstructs correct hidden).

2. **Ensure mixture gating / stochastic components are controlled for step-zero audits.**  
   `StrategicMixtureGate` uses `torch.multinomial` when `deterministic=False`; step-zero ratio tests must either:
   - run with deterministic mixture gating, or
   - snapshot/reset RNG state so that `old_logp` and `new_logp` correspond to identical mixture option choices.

## Success criterion (must be checked with the step-zero audit)

After implementing the primary repair:

1. On `BC init` and `RL_CONTROL update_10` checkpoints, the step-zero ratio gate must pass for non-trivial action selections (i.e., when `acts_t != PASS` occurs).
2. The gate should satisfy the plan tolerances:
   - CUDA float32: `abs(new_logp - old_logp) <= 1e-4` and `abs(ratio - 1.0) <= 1e-4`
3. Update-time entropy computed during `ppo_update_from_fragment` should track behaviour entropy much more closely and should not collapse to exactly `0.0` unless the *behaviour* distribution itself is near-deterministic.

## Falsification test

If, after the support-reconstruction fix, the step-zero ratio gate still fails on early checkpoints (before PASS-only collapse) with substantial non-PASS action frequency, then the primary outcome classification must be revised (and recurrent-state reconstruction should be promoted to primary for the next repair iteration).

