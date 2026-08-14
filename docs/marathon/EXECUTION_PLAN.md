# Marathon Redesign Execution Plan

**Plan ID:** `MARATHON_REDESIGN_LOCKED_V1`

**Status:** Architecturally locked

**Scope:** Marathon research, evaluation, platform, deployment, and release programme

## 1. Mission and governing principles

Build the strongest qualified, CPU-deployable Generals.io agent through a reproducible evidence loop:

```text
repository truth -> hypothesis -> bounded implementation -> validation
-> paired evaluation -> statistical evidence -> promotion or rejection
```

The programme begins by preserving and reproducing the trusted sprint learner. A large repository migration must not precede `MARATHON_BASELINE_V0`. Research and platform work separate after the baseline, evaluator, and minimum registry exist. Platform polish must not block strength research unless a shared integrity dependency is missing.

The official engine submodule is authoritative for rules, protocol, runners, and sandbox dependencies. It may not be silently patched. Results, metrics, and provenance may never be invented.

This plan is context-free and is the canonical implementation specification. Material changes require an explicit `PLAN_CONFLICT` and architect decision under `AGENTIC_EXECUTION_PROTOCOL.md`; an implementer must not silently redesign it.

## 2. Human-controlled boundaries

The repository is intentionally public during early development so research tools can inspect it. Active tactics must remain protected in public artefacts. Before a serious competitive upload or when tactics become competitively or commercially sensitive, raise `REPOSITORY_PRIVACY_CUTOVER`. Never change visibility automatically.

The following always require explicit human authorization:

- creation or expansion of paid compute, storage, endpoints, or other paid resources;
- repository visibility changes;
- destructive removal of potentially unique checkpoints, datasets, manifests, replays, provenance, or evidence;
- pushes to a remote unless the bounded task explicitly permits them;
- competition submission uploads.

The dashboard has no authority for arbitrary shell commands, browser-supplied filesystem paths, Git mutation, paid resources, visibility changes, pushes, or uploads.

## 3. Persistent authority and execution state

Stage 0 creates and maintains:

- `docs/marathon/DESIGN_AUTHORITY.md`: canonical architecture, invariants, decisions, and supersessions;
- `docs/marathon/EVIDENCE_LEDGER.md`: source classification, hypotheses, experiments, results, and promotion decisions;
- `docs/marathon/CONTROL_ENGINEERING.md`: action-control and PPO-semantics invariants;
- `configs/marathon/programme.yaml`: typed programme, evaluator, promotion, and controller policy;
- `experiments/marathon/ACTIVE_STATE.json`: current handoff and operational state;
- the root `AGENTS.md` and `docs/marathon/AGENTIC_EXECUTION_PROTOCOL.md`.

Repository state is shared memory. Chat history is never required. Every working session updates `ACTIVE_STATE.json` before stopping. Disagreements among code, configuration, evidence, and authorities are surfaced and reconciled; they are not silently resolved.

## 4. Independent research authority

The architect/research agent must inspect primary sources and public implementations relevant to the active decision, including the current competition rules, AverageJoe code and paper, zero-v3 / Artificial Generals Intelligence, relevant PPO/JAX implementations, constrained RL, shielding, Expert Iteration, residual RL, MPC/RL hybrids, PBT/PFSP, locally supplied Discord evidence, and newer relevant work it discovers.

The agent may propose ideas not listed here. It may not silently make them canonical. Every new idea follows:

```text
SOURCE -> EVIDENCE_LEDGER -> mechanism -> interaction and risk
-> bounded experiment -> observed result -> PROMOTE / REJECT / DEFER
```

Source classifications must distinguish official rules, repository fact, local evidence, primary literature, public implementation, community report, and speculation.

## 5. Stage 0 — authority and forensic freeze

### 5.1 Objectives

Freeze repository truth and unique evidence before structural changes. Create the missing authority files and the canonical configuration. Inventory training paths, entrypoints, checkpoints, datasets, manifests, generated artefacts, dashboard routes, packaging paths, build outputs, worktrees, external runtime directories, and active processes.

### 5.2 Classification

Every material artefact receives one of:

- `KEEP`
- `MIGRATE`
- `ARCHIVE`
- `REGENERABLE`
- `DELETE_CANDIDATE`
- `UNKNOWN`

Only proven caches, temporary scratch, and build staging may be deleted automatically. A delete candidate requires evidence that it is duplicated or regenerable and contains no unique provenance. Unknown artefacts are preserved.

### 5.3 Freeze record

Record at minimum:

- current branch, commit, submodule commit, worktrees, and clean/dirty state;
- every trainer and operational entrypoint and its observed consumers;
- checkpoint, dataset, engine, config, and source provenance where recoverable;
- output directories, archive paths, and stale/overlapping `dist` or submission paths;
- active local and remote processes, with `UNKNOWN` used when an audit is unavailable;
- repository visibility fact and the privacy-cutover gate;
- contradictions and unresolved repository truth.

### 5.4 Acceptance

Stage 0 passes when authorities and configuration parse, the inventory is evidence-backed, unique research evidence is frozen, no unsupported deletion occurred, and `ACTIVE_STATE.json` identifies the next bounded Stage 1 action.

## 6. Stage 1 — reproduce `MARATHON_BASELINE_V0` before migration

No large JAX source relocation, legacy-entrypoint removal, trainer rewrite, or performance-affecting repository migration may occur until the trusted learner is reproduced and fingerprinted.

### 6.1 Baseline source

The verified checkpoint at:

`C:\Users\pries\quantsilico-runtime\cloud_assisted_deadline_salvage_v1_final\ckpt_final_u482_t7593984`

is the historical Stage 1 source and must be registered as raw and EMA variants of `SPRINT_VALID_PPO_7M59`. It represents update 482 and approximately 7.59 million valid transitions. Repository forensics must verify, not assume, every associated claim.

### 6.2 Baseline capsule

Produce a content-addressed capsule containing:

- source commit, official-engine commit/hash, configuration hash, runtime lock, dataset hashes, and checkpoint hashes;
- exact commands, seed set, environment variables, and runtime metadata;
- first N observations, actions, legal masks, rewards, dones, and relevant discrete states;
- rollout checksum and first PPO update metrics/checksums with per-field tolerances;
- raw model, EMA model, optimizer, rollout carry, RNG, curriculum, and controller state before and after checkpoint/resume;
- hot-path TPS, end-to-end TPS, and valid-learning TPS, reported separately;
- terminal outcomes, reward/valid-transition counts, KL, clip fraction, entropy, policy/value loss, gradient norms, and finite-value checks;
- external paired W/D/L or score evidence and replay identities.

### 6.3 File and semantic hashing

For checkpoint, model, optimizer, carry, RNG, and controller state, record both:

- `FILE_SHA256`: hash of the serialized file bytes;
- `SEMANTIC_STATE_SHA256`: hash of canonical logical state.

Semantic hashing orders array keys deterministically and includes each key, dtype, shape, and canonical contiguous value bytes. Container metadata or serialization differences alone must not imply learner-state drift.

### 6.4 Determinism contract

Record:

- `JAX_VERSION`
- `XLA_VERSION`
- `CUDA_VERSION`
- `GPU_MODEL`
- `XLA_FLAGS`
- `DTYPE_POLICY`
- `DETERMINISM_MODE`

Report these distinct levels:

- `BITWISE_DETERMINISM`
- `STATE_SEMANTIC_DETERMINISM`
- `BEHAVIOURAL_DETERMINISM`

Exact discrete outputs and state transitions must match where the implementation promises determinism. Floating fields use explicitly recorded, tight, per-field absolute/relative tolerances. A floating-bit difference on GPU is not automatically a failure when semantic state and behaviour remain equivalent within the declared contract.

### 6.5 Gate

`MARATHON_BASELINE_V0` passes only when valid learning and checkpoint/resume are demonstrated, fingerprints are stored, external gameplay is measured, and the capsule can be independently rechecked. Any later migration compares against this reference. An unexplained behavioural drift or material performance regression blocks migration promotion.

## 7. Stage 2 — serious paired evaluator

### 7.1 Pair definition

```text
PAIR = the same canonical generated map and seed, with the candidate playing
       seat A once and seat B once across two games.

PAIR_SCORE = mean candidate score across those two games.
```

Always report `PAIR_COUNT` and `ACTUAL_GAME_COUNT`. Thus 64 pairs are 128 games, 512 pairs are 1,024 games, and 2,048 pairs are 4,096 games. Sequential inference operates on paired candidate-minus-incumbent observations and must not pretend correlated seat/map games are independent. The canonical per-game score is win `1`, draw `0.5`, loss `0` unless programme configuration explicitly supersedes it.

### 7.2 Evaluation design

- Cheap screening starts at 64 fresh pairs per configured opponent.
- Promotion evaluation starts at 512 fresh pairs and may continue to 2,048 pairs.
- Use a statistically valid sequential procedure: an anytime-valid confidence sequence, SPRT, alpha-spending design, or another documented valid method.
- Do not repeatedly peek at an ordinary fixed-sample 95% confidence interval.
- Preserve common seeds, pairing, seat swaps, opponent identity, engine identity, and replay provenance.
- Promotion seeds are disjoint from training, tuning, and cheap-screen seeds.

The default method is a 95% anytime-valid empirical-Bernstein confidence sequence over bounded paired differences, with its assumptions and implementation tested.

### 7.3 Promotion policy

Canonical defaults live in `configs/marathon/programme.yaml`, not source constants:

- `promotion.practical_margin = 0.01`
- `promotion.robustness_noninferiority_margin = -0.005`
- `promotion.worst_matchup_improvement = 0.05`

Normal promotion requires the lower confidence-sequence bound on paired advantage to exceed the practical margin and all integrity/latency/fault gates to pass. Statistical evidence above zero alone is insufficient.

A documented robustness promotion may be used when aggregate performance is no worse than the configured noninferiority margin and a known worst matchup improves by at least the configured threshold, with no unacceptable new exploitability, latency, or fault regression.

### 7.4 Metrics

Record average score/strength, uncertainty, W/D/L, Elo only when its model is declared, latency, faults, replays, and:

- `WORST_MATCHUP_SCORE`
- `BOTTOM_QUARTILE_MATCHUP_SCORE`
- `MATCHUP_SCORE_STD`

Report simulator FPS, hot-path TPS, end-to-end TPS, and valid-learning TPS separately. Aggregate strength must not hide a catastrophic concentrated matchup.

### 7.5 Acceptance

Evaluator tests must cover seat symmetry, deterministic seed/map construction, pair accounting, confidence-sequence coverage/sanity, optional-stopping behaviour, promotion margins, robustness exceptions, fault attribution, replay identity, and resumable/atomic results.

## 8. Stage 3 — minimum canonical registry

Implement only the registry needed to run research safely:

- experiment;
- run;
- checkpoint;
- candidate;
- evaluation;
- opponent/reference.

Records use stable readable IDs plus hashes, atomic writes, explicit schema versions, source/engine/config identity, lineage, seeds, commands, budgets, stop reasons, artefact locations, and evidence links. Candidate and checkpoint discovery is registry-driven; filenames are presentation, not truth.

Anything that touches action selection must declare exactly one required enum:

- `PPO_SEMANTICS=UNCHANGED`
- `PPO_SEMANTICS=PRE_SAMPLING_MASK`
- `PPO_SEMANTICS=OFF_POLICY_AUXILIARY`
- `PPO_SEMANTICS=EVAL_ONLY`

Missing or ambiguous semantics invalidates the experiment before training. The registry must reject incompatible lineage, unresolved hashes, missing evaluator identity, and silent overwrite.

## 9. Stage 4A — high-confidence research track

Stage 4A starts immediately after Stages 1–3 pass and does not wait for full dashboard, repository, or packaging work.

### 9.1 Successive halving

Use bounded successive halving instead of fully training every arm:

1. all candidates, small budget, one seed;
2. eliminate dominated or integrity-failing candidates;
3. survivors, larger budget, two or three seeds;
4. finalists, fresh promotion-scale paired evaluation.

Predeclare screening metrics and stop rules. Cheap results cannot directly promote a champion.

### 9.2 Rollout geometry without confounding

Separate two experiment families.

`HORIZON_ABLATION` approximately controls transitions per PPO update while varying temporal horizon, when feasible. A conceptual geometry is:

```text
512 envs x 32 steps
256 envs x 64 steps
128 envs x 128 steps
64 envs x 256 steps
32 envs x 512 steps
```

Record unavoidable differences in minibatches, optimizer exposure, GAE context, temporal correlation, memory, and throughput.

`THROUGHPUT_GEOMETRY_ABLATION` then tests promising horizons at operationally large environment counts, such as `512 x 128`, `512 x 256`, and `512 x 512` when memory permits. It optimizes external strength per GPU-hour. Do not attribute a confounded result solely to horizon.

### 9.3 Initial ablations

Prioritize bounded tests of rollout horizon/geometry, spawn/opponent curriculum, top-advantage sampling (including representative 100/50/25 variants), anchor type and decay, sparse reward integrity, and fixed schedule versus KL feedback control.

Maintain raw and EMA state in every compatible training run. Evaluate raw and EMA from the same transitions at negligible additional training cost; do not create duplicate training arms initially. Ablate EMA decay only if evidence warrants it.

### 9.4 Early training controllers

Test `KL_CONTROLLER`, `CURRICULUM_CONTROLLER`, and `ANCHOR_CONTROLLER` early because they directly govern training stability and difficulty. Controllers must be bounded, finite-safe, rate-limited where appropriate, checkpointed, resume-equivalent, observable, and unable to hide missing rewards or invalid gradients. Fixed-schedule baselines remain available.

Implement castle telemetry and schemas early. Keep castle intervention losses disabled until controlled evidence promotes them.

## 10. Stage 4B — platform track in parallel

Stage 4B may proceed independently after shared prerequisites pass. It may not block Stage 4A except where experiment identity, baseline integrity, or evaluator correctness requires it. Independent work must avoid conflicting edits and preserve equivalence gates.

### 10.1 Repository migration and cleanup

- Migrate training paths only after the baseline capsule exists.
- Preserve compatibility shims until command, checkpoint/resume, behaviour, and performance equivalence pass.
- A hot-path or end-to-end TPS regression greater than 5% blocks migration unless a documented valid-learning benefit justifies it.
- Remove obsolete entrypoints only after consumer searches and replacement tests.
- Reconcile overlapping `dist`, generated output, submission, and archive directories through the forensic classifications.
- Cleanup tooling defaults to dry-run, displays exact resolved targets, and requires evidence for destructive classes.
- Preserve unique manifests, working scripts, dataset hashes, checkpoint provenance, replays, and executable/LF attributes.

### 10.2 Dashboard

The dashboard consumes typed registry and job schemas; it is not the source of experiment truth. Jobs are durable, allowlisted, config-ID-driven, restart-safe, observable, and auditable. Browser APIs accept neither arbitrary commands nor arbitrary filesystem paths and possess none of the human-controlled authorities in Section 2. Generated frontend types or equivalently checked contracts prevent backend/frontend drift.

Dashboard acceptance includes schema validation, job persistence/recovery, allowlist rejection, progress/error reporting, route/component integration, no arbitrary shell path, and no Git/paid/visibility/upload mutation.

### 10.3 Packaging

Keep the existing qualified fallback path until the replacement passes equivalence. Generate readable, deterministic, atomic, non-overwriting outbox packages such as:

`submission/outbox/qs-<readable-candidate>-vNNN-YYYY-MM-DD.zip`

Also store an immutable content-addressed archive and registry record. Package discovery is registry-driven. Validate protocol files, dependencies, size, deterministic content, source/candidate/checkpoint identity, hashes, smoke execution, and official sandbox compatibility. Upload remains manual.

## 11. Stage 5 — isolated architecture, value, and capacity research

Isolate major changes where practical:

- `T0`: existing architecture, existing observation, scalar critic;
- `T1`: patch transformer only, same observation and scalar critic;
- `T2`: best T1 plus legal temporal/global history features;
- `T3`: best T2 plus HL-Gauss or other promoted critic change;
- `T4`: capacity scaling around the winner.

Keep optimizer, schedule, training budget, evaluator, and other factors fixed unless the experiment explicitly declares the difference. Do not bundle architecture, observation, critic, optimizer, and scale changes and then attribute the result to one component. Use successive halving and multi-seed confirmation.

## 12. Stage 6 — strategic and control research

Begin only from a strong, valid baseline. Candidate lanes include castle counterfactual control/successor value/preference learning, DAgger, residual heuristic policies, legal belief observers, risk governors, MPC experts, Expert Iteration, constrained/PID-Lagrangian control, and later population/PSRO mechanisms.

Invariants:

- controls and counterfactuals respect the required `PPO_SEMANTICS` enum;
- a deterministic pre-sampling mask is identical during rollout and PPO recomputation;
- post-policy intervention data does not enter on-policy PPO ratio loss unless the composite behaviour distribution is formally represented;
- counterfactual alternatives are off-policy auxiliary evidence, not sampled on-policy actions;
- belief/state observers use only legal information available to the deployed policy;
- MPC begins as an evaluation/expert/data-generation lane before any evidence-backed policy integration;
- castle build/control, successor value, preference, and intervention-cost effects are separable experiments;
- forcing `BUILD` or enabling shaping is not canonical without promotion evidence.

## 13. Stage 7 — population, adaptive scale, distillation, and release

### 13.1 Opponent population

Maintain diagnostic opponents including Pass, Random, Expander, Hunter, the strongest qualified heuristic, historical references, prior champions, and current champion. Introduce PFSP/PBT/PSRO only after the basic mixture and evaluator are reliable.

### 13.2 Adaptive budgets

The transition budgets `5M`, `25M`, `50M`, `100M`, `250M`, and `500M` are reference checkpoints, not mandatory stops. A controller may stop early, skip levels, or extend to `1B+` according to external-strength evidence and marginal value.

Record with uncertainty where possible:

- `DELTA_STRENGTH_PER_TRANSITION`
- `DELTA_STRENGTH_PER_GPU_HOUR`
- `DELTA_STRENGTH_PER_GPU_COST`
- `MARGINAL_STRENGTH_PER_GPU_HOUR`

Compute escalation requires valid learning, improving external strength, acceptable faults/robustness, and competitive marginal value. Improvement without useful learning velocity is insufficient justification for unlimited scale.

### 13.3 Deployable-student objective

The objective is the strongest qualified deployable student, not the highest-Elo teacher. Periodically shadow-distill promising teachers and automatically start a tracked distillation job when a teacher becomes champion. Record:

- teacher strength;
- student strength;
- distillation gap;
- student CPU p99 latency;
- memory and package size;
- faults and sandbox qualification.

A slightly weaker teacher may be preferred if its student retains materially more strength and qualifies reliably.

## 14. Permanent historical reference set

Register these diagnostic references permanently:

- `HEURISTIC_V2_FALLBACK`
- `SPRINT_BC_STEP450`
- `SPRINT_VALID_PPO_7M59` raw
- `SPRINT_VALID_PPO_7M59` EMA

Register `FORENSIC_ZERO_REWARD_35M` only in the evidence ledger as `INVALID_LEARNING_INTEGRITY`. It must never be used as a strength opponent, learning reference, or teacher.

## 15. Evidence, testing, acceptance, and promotion

Every bounded implementation records hypothesis, plan ID, base/end commit, source/engine/config/checkpoint hashes, seeds, exact commands, environment, budget, stop criteria, tests, results, limitations, and decision. Failed and negative experiments remain discoverable.

Behavioural changes require tests. Structural migrations require old/new equivalence and consumer searches. Training changes require reward/terminal/gradient/finite-value health plus checkpoint/resume. Evaluator changes require statistical tests. Packaging requires deterministic official-runtime smoke tests. Dashboard operations require authority and persistence tests.

No candidate promotes from training telemetry alone. Promotion requires:

1. learning-integrity and provenance gates;
2. reproducible checkpoint/resume;
3. fresh paired external evaluation with sequentially valid statistics;
4. practical or documented robustness improvement;
5. latency, memory, fault, matchup, and sandbox qualification appropriate to its role;
6. evidence-ledger and registry updates;
7. a named rollback candidate.

`COMPLETE` is invalid for implementation work with an unexplained empty diff or without tests/evidence. A verification-only task may legitimately have no diff only when its scope and evidence explicitly say so.

## 16. Canonical execution order

```text
Stage 0:  authority + forensic freeze
Stage 1:  MARATHON_BASELINE_V0 reproduction
Stage 2:  paired evaluator with sequentially valid statistics
Stage 3:  minimum experiment/run/checkpoint/candidate/evaluation registry
Stage 4A: high-confidence research begins immediately
Stage 4B: repository/dashboard/package platform work in parallel
Stage 5:  isolated transformer/value/teacher-capacity experiments
Stage 6:  strategic/control/castle/belief/MPC research
Stage 7:  population + adaptive scaling + continuous distillation + release
```

Do not begin a wholesale redesign to restate this plan. Execute the next bounded safe action from `ACTIVE_STATE.json`, surface conflicts, and promote only with evidence.
