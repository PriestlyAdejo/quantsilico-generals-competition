# Phase 9F-R root-cause analysis (Entropy collapse)

created_at_utc: 2026-08-05T10:08:00Z
git_commit: `ab7c9c0f541c47300406b8c1ae5932ba698b3973`

## Stage F diagnostic competence tier (diagnostic-only)

Using the same step-zero entropy+ratio audit across checkpoints:

- BC checkpoint (`experiments/phase9f_cnn_ranker_v1/checkpoints/bc/model.json`):
  - `H_behaviour_full_legal_mean`: 0.679772
  - `step_zero_ratio_gate`: 3/32 transitions within tolerance
- PPO CONTROL final (`experiments/phase9f_overnight_ppo/rl_control/final.json`, evidence: `experiments/manifests/phase9fr_ppo_ratio_semantics.json`):
  - `H_behaviour_full_legal_mean`: 2.1414e-05
  - `pass_probability_full_legal_mean`: ~1.0
  - `update_impl_entropy_mean`: 0.0

Both PPO arms therefore become effectively deterministic on `PASS` in the behaviour policy, so this run is **DIAGNOSTICALLY_BELOW_BC** (competence not preserved on the measured policy entropy channel).

## Stage G root-cause matrix

### primary_outcome (must pick exactly one)

`B — likelihood-ratio support mismatch`

### secondary_findings (multiple allowed)

1. **Learner recomputes `new_logp` over a *restricted* action support `{acts_t, PASS}` rather than the FULL legal action support used for `old_logp`.**  
   - Repository provenance (verified):
     - Code: `src/generals_bot/training/rollout.py::ppo_update_from_fragment`
       - Mask construction: `mask.scatter_(...)` plus `mask[:, 0] = True`
       - Entropy computed from `Categorical(logits=masked)` over that restricted mask  
       - Relevant code region: `src/generals_bot/training/rollout.py` lines ~70-87 (mask + `new_logp` + entropy)
     - Code: `src/generals_bot/training/actors.py::PersistentActor.collect_fragment`
       - `mask = legal_mask_observation(obs)` and `apply_action_mask(logits, mask)` before sampling `action` and computing `logp`
       - Relevant code region: `src/generals_bot/training/actors.py` lines ~191-214 (legal mask + `dist.log_prob(action)`)
     - Commit: `ab7c9c0f541c47300406b8c1ae5932ba698b3973`
   - Evidence (verified with exact numerics from step-zero audit):
     - BC init checkpoint (ratio fails when non-PASS actions are sampled):
       - Evidence JSON: `experiments/manifests/phase9fr_ppo_ratio_semantics_model.json`
       - Evidence SHA-256: `074c261e35d2384baea459be016c5792ebcf1288954d6850237c819058b55505`
       - Results: `step_zero_ratio_gate = 3/32` transitions within tolerance
       - Update-time support sizes unique: `[1, 2]` (so selected actions include non-PASS cases)
     - CONTROL update_10 (ratio fails; behaviour entropy still high):
       - Evidence JSON: `experiments/manifests/phase9fr_ppo_ratio_semantics_rl_control_update_10.json`
       - Evidence SHA-256: `22478dd29e252dae5ed3645f952e4642fee30628a079820460913ddb0781792d`
       - Results: `step_zero_ratio_gate = 2/32`
       - `H_behaviour_full_legal_mean = 0.973901`
     - CONTROL update_50 and later (collapse to PASS makes restricted support irrelevant):
       - Evidence JSON: `experiments/manifests/phase9fr_ppo_ratio_semantics_rl_control_update_50.json`
       - Evidence SHA-256: `ec6e91423de13f18fbf1bb97112ef8061411b223983ff65ca71e92432e7eb7e6`
       - Results: `step_zero_ratio_gate = 32/32`
       - `pass_probability_full_legal_mean ~ 0.99999`
       - `update_support_sizes_unique = [1]` and `update_impl_entropy_mean = 0.0`
   - Reproduction command(s):
     - BC:
       - `python scripts/phase9fr_step_zero_ratio_entropy_audit.py --checkpoint-json experiments/phase9f_cnn_ranker_v1/checkpoints/bc/model.json --seed 123 --rollout-steps 32 --device cuda`
     - CONTROL update_10:
       - `python scripts/phase9fr_step_zero_ratio_entropy_audit.py --checkpoint-json experiments/phase9f_overnight_ppo/rl_control/update_10.json --seed 123 --rollout-steps 32 --device cuda`
     - CONTROL update_50:
       - `python scripts/phase9fr_step_zero_ratio_entropy_audit.py --checkpoint-json experiments/phase9f_overnight_ppo/rl_control/update_50.json --seed 123 --rollout-steps 32 --device cuda`
   - verified_vs_inferred: **verified** (code mismatch + numerics)

2. **Recurrent-state reconstruction mismatch likely further worsens ratio consistency (but is not needed to establish the *primary* support-mismatch defect).**  
   - Repository provenance (verified):
     - Learner update sets `hidden = model.initial_hidden(b, ...)` and `cell_mem = model.initial_cell_memory(b, ...)` for the per-transition batch, with no per-timestep stored hidden state.
   - Evidence (verified + partially explanatory):
     - The step-zero audit explicitly notes this mismatch between old_logp (persistent sequential hidden/cell_mem) and new_logp_impl (reset hidden/cell_mem per-transition).
     - Commit: `ab7c9c0f541c47300406b8c1ae5932ba698b3973`
   - verified_vs_inferred: **verified that the reset occurs**, inferred contribution to ratio failures beyond the support-mismatch.

3. **The entropy collapse is not purely telemetry: the behaviour policy entropy is near-zero at the end, indicating a genuine distribution collapse onto `PASS`.**  
   - Evidence (verified):
     - RL_CONTROL final audit:
       - `experiments/manifests/phase9fr_ppo_ratio_semantics.json`
       - SHA-256: `66f6f019443d69ae971bbbb4b61c51a5cdfe499b988a1c3c8d481a30c48d6941`
       - `H_behaviour_full_legal_mean = 2.1414e-05`, `pass_probability_full_legal_mean ~ 1.0`
     - RL_CURRICULUM final audit:
       - `experiments/manifests/phase9fr_ppo_ratio_semantics_rl_curriculum.json`
       - SHA-256: `f5f26cc767ae90983940ffd0a0ff8eb8c43afc046436a27be3970c3dfbdbdded`
       - `H_behaviour_full_legal_mean = 1.0052e-09`, `pass_probability_full_legal_mean = 1.0`
   - verified_vs_inferred: **verified**

## Why outcomes A/C/D/E are disfavoured

- **A Telemetry-only** is disfavoured because behaviour entropy collapses (not just update-time entropy).
- **C Candidate-order mismatch** is disfavoured because the failing mechanism matches an explicit *mask support* mismatch in `ppo_update_from_fragment`, and the flat action index space does not depend on candidate ordering.
- **D Genuine collapse with valid ratios** is disfavoured as the primary outcome because the step-zero ratio gate fails on BC and early updates (update_10), i.e. likelihood-ratio semantics were not valid at the time collapse began.
- **E Historical reconstruction unavailable** may contribute (recurrent hidden reset without per-timestep storage), but the support-mismatch defect is directly observable and sufficient to explain the ratio failures and their disappearance after `PASS`-only collapse.

