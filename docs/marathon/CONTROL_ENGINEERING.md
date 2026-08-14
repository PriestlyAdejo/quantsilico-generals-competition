# Marathon Control Engineering Authority

**Plan ID:** `MARATHON_REDESIGN_LOCKED_V1`

## Scope

This authority governs anything that changes action selection, training difficulty, policy constraints, intervention, counterfactual data, or tactical risk. No controller is canonical merely because it is architecturally attractive.

## Required PPO semantics

Every action-touching experiment declares exactly one:

- `PPO_SEMANTICS=UNCHANGED`: sampled behaviour and PPO recomputation are the same unmodified policy distribution.
- `PPO_SEMANTICS=PRE_SAMPLING_MASK`: a deterministic legal/safety mask is applied before sampling and identically during PPO recomputation.
- `PPO_SEMANTICS=OFF_POLICY_AUXILIARY`: intervention/counterfactual data is excluded from on-policy ratio loss and used only by a declared auxiliary/off-policy objective.
- `PPO_SEMANTICS=EVAL_ONLY`: the controller is excluded from training and evaluated as a deployment/expert overlay.

Missing or ambiguous semantics invalidates an experiment before training.

## Non-negotiable invariants

1. Post-policy interventions do not enter PPO ratio loss unless the complete behaviour distribution is formally represented and tested.
2. Counterfactual alternatives are not relabelled as sampled on-policy actions.
3. Pre-sampling masks are deterministic from recorded legal inputs and identical in rollout/recomputation.
4. Belief observers use only information legally available to the deployed agent.
5. A controller cannot hide zero reward, missing terminals, non-finite gradients, invalid transitions, or resume drift.
6. Controller state/configuration is hashed, checkpointed, restored, and included in semantic-state verification.
7. Every controller has explicit bounds, units, update cadence, rate limits, fail-safe value, telemetry, and disable/rollback path.

## Early training-control lane

`KL_CONTROLLER`, `CURRICULUM_CONTROLLER`, and `ANCHOR_CONTROLLER` are early Stage 4A ablations. Each is compared against a fixed schedule, isolated where practical, and evaluated through valid learning plus external paired strength. They do not modify tactical actions unless separately declared under the PPO semantics enum.

## Strategic/control lane

Castle telemetry and schema support arrive early, but castle losses/interventions remain disabled until promoted by controlled evidence. Castle counterfactual, successor-value, preference, and intervention-cost effects are separate experiments.

Residual heuristics, safety governors, belief-sensitive control, constrained/PID-Lagrangian risk, MPC, Expert Iteration, DAgger, and PSRO/PFSP enter only from a strong baseline. MPC begins as `EVAL_ONLY` or expert/data generation unless another semantics class is formally justified.

## Controller record

Every controller registry record contains:

```text
CONTROLLER_ID
PPO_SEMANTICS
INPUTS_AND_INFORMATION_BOUNDARY
ACTION_DISTRIBUTION_EFFECT
STATE_AND_CHECKPOINT_SCHEMA
BOUNDS_AND_RATE_LIMITS
FAIL_SAFE
TELEMETRY
HYPOTHESIS
BASELINE
TESTS
EVIDENCE
PROMOTION_STATUS
ROLLBACK
```

## Current status

No Marathon control experiment is active. This document authorizes schemas and tests only; it does not promote a controller or enable castle intervention losses.
