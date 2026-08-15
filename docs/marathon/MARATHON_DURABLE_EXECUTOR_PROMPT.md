# Marathon permanent durable executor prompt (commissioned 2026-08-15)

Commissioned via `/loop 30m --durable`. This is the versioned instruction the
recurring durable loop executes on every wake. Sentinel:
`AGENT_LOOP_TICK_MAREXEC` (30 min). Supersedes the earlier
`AGENT_LOOP_TICK_MARATHON` sentinel loop (deprecated, proven insufficient to
drive agent turns in EV-0036 — retained only until its terminal is reaped).

Proven boundaries: local computation cannot execute while Windows is
suspended; remote self-stop (`scripts/dev/remote_orchestrator_with_stop.sh`)
protects paid compute independently of this loop.

---

QUANTSILICO MARATHON PERMANENT EXECUTOR — Resume and advance the existing QuantSilico Generals Marathon from the ACTUAL current repository state. This recurring task exists specifically so the operator NEVER needs to type "continue" after an experiment, research family, stage, Goal run, Quest run, remote GPU job, or ordinary agent session ends.

FIRST acquire/reconcile the Marathon execution lease and query RunPod BEFORE doing anything else. Then read experiments/marathon/ACTIVE_STATE.json, the canonical Marathon programme and amendments, Stage-3 registry, evidence ledger, RunPod resource ledger, current Git state, current processes, experiment artefacts and telemetry. These are the source of truth. Historical planning snapshots must never override newer empirical state.

DO NOT assume this prompt describes the current stage. Discover it from the repository every iteration.

## CORE TERMINATION RULE

THE MARATHON IS NOT COMPLETE merely because: an experiment completed; an ablation family completed; Stage 4A completed; Stage 4B completed; Stage 5 completed; Stage 6 completed; a remote GPU run was launched; a remote GPU run completed; an evaluation completed; a candidate was rejected; a candidate was promoted; a Git commit was created; an interactive Agent/Quest run would normally stop.

The overall recurring task remains useful until canonical completion through Stage 7, final qualification, adversarial review, repairs, integration, PR/checks, merge, synced main and post-merge evidence proof.

At the beginning of EVERY iteration:

1. Acquire the existing Marathon execution lease.
2. If another genuinely live Marathon executor owns the lease, do not duplicate its work. Reconcile only resource safety if appropriate, then finish this loop iteration cheaply.
3. Query all QuantSilico RunPod resources.
4. Correlate every paid resource with experiments/marathon/runpod_resources.json, ACTIVE_STATE and actual workload telemetry.
5. Preserve healthy owned workloads.
6. Detect completed, failed, idle, duplicate or orphaned resources.
7. Preserve required outputs before stopping resources.
8. Enforce RUNPOD_ZERO_IDLE_BURN.
9. Then read ACTIVE_STATE.NEXT_SAFE_ACTION and the canonical dependency graph.
10. Execute real work.

## RUNPOD INVARIANT

A RUNNING paid Pod must correspond to useful active compute.

For completed remote experiments: checkpoint/flush → durable completion marker → preserve/fetch artefacts as required → verify integrity → STOP paid GPU → billing evidence → adjudication.

Use the existing remote self-stop wrapper for future long-running experiments wherever valid so GPU shutdown does NOT depend on this laptop, Qoder session or loop being awake.

Never reproduce the previous A100 + fallback A40 duplicate-idle incident. Before fallback provisioning, cancel obsolete preferred-resource retry loops. Do not stop genuinely active training. Do not create duplicate Pods for an experiment that already has active owned capacity.

## BOUNDED REPAIR AUTHORITY

The operator authorises autonomous bounded engineering repair throughout the Marathon. When something fails, classify the failure.

If it is an ENGINEERING/INFRASTRUCTURE DEFECT (path/cwd error; serving adapter problem; observation mismatch; action mapping bug; checkpoint loading failure; process orchestration error; RunPod lifecycle bug; broken telemetry; transfer failure; throughput implementation defect; packaging problem; evaluator implementation bug; stale/malformed programme state): reproduce → diagnose → smallest justified repair → test/A-B validate → record evidence → resume. Do NOT ask the operator for routine repair permission. If the repair alters scientific identity, preserve the original evidence and PREDECLARE a successor experiment before running it.

If the failure is an EXPERIMENT-DESIGN DEFECT: preserve the original result; do not modify its rules post-hoc; record why it could not answer the question; predeclare/register a better successor; execute it.

If implementation and experiment are valid and the candidate genuinely loses: REJECT IT. Do not keep changing the rules until it wins. Extract the scientific conclusion and move to the next canonical hypothesis.

The serving failure that invalidated EV-0033 is the precedent: suspicious result → audit → concrete protocol defect → repair serving only → invalidate contaminated evidence → rerun controlled evaluation → preserve corrected scientific result.

## SCIENTIFIC PROMOTION AUTHORITY

Training telemetry is a screening signal, NOT final strength evidence. External Stage-2 gameplay evaluation remains promotion authority.

Never promote because: vloss looks good; entropy looks healthy; PPO ratio is near one; TPS is high; a candidate survived training; a candidate beats one named opponent; a leaderboard rank briefly improved.

Use predeclared rules, multi-seed evidence where required, fault-free serving and controlled paired gameplay.

Current known research lesson: healthy PPO telemetry can coexist with draw-at-truncation gameplay. Win conversion/generalised gameplay is therefore a critical external signal.

## STAGE 4A

Continue ALL remaining canonical Stage-4A families rather than treating one family as the whole stage. The existing programme includes, where not already resolved: spawn-distance / opponent-difficulty curriculum; throughput/training geometry; raw vs EMA; top-advantage fractions; anchor/reference-policy variants and decay; sparse/terminal reward integrity; fixed schedule vs KL feedback; KL controller; curriculum controller; anchor controller; other declared Stage-4A ablations.

Do not automatically repeat four SH rounds for every hypothesis. Use the smallest rigorous evidence funnel appropriate to each research question.

Generic funnel: PREDECLARE → REGISTER → CORRECTNESS SMOKE → LEARNER HEALTH → SUCCESSIVE HALVING → MULTI-SEED WHERE WARRANTED → STAGE-2 GAMEPLAY → PROMOTE / REJECT / UNRESOLVED.

When a family answers its question, move on. Negative results are valid results. Do not restart horizon research without a specific predeclared interaction hypothesis.

## ELITE_REPLAY_AUGMENTATION (canonical Stage-4A family)

Preserve and advance the existing elite replay programme and DATASET-ELITE snapshots. The research question is not "imitate ResBot". It is: does elite replay augmentation improve GENERALIZED external gameplay strength, strategy coverage and/or sample efficiency over matched self-play controls without privileged-information leakage or opponent-specific overfitting?

Continue the replay data plane in dependency-safe parallel capacity. Use public replay data for legitimate research paths: behavioural-cloning warm start; decaying replay auxiliary loss; elite opponent imitation clones; replay-derived scenario curriculum; strategy/style analysis; elite disagreement-state mining; later DAgger/MPC/counterfactual research. Never pretend a fixed historical replay is an interactive opponent.

### FOG-OF-WAR HARD GATE

Raw historical replay state may include information unavailable to the competition policy. Before replay samples reach the policy: full replay → exact player-perspective reconstruction → canonical legal competition observation → action sample. Prove parity against the existing legal observation path. No hidden enemy information may enter deployed policy inputs. Privileged historical state may only be used in explicitly declared training-only/off-policy analysis mechanisms that do not contaminate policy observation semantics.

### ANTI-OVERFITTING

Replay research must use: player-disjoint holdouts; time-disjoint holdouts; seed/game-disjoint splits; style coverage where practical; per-player contribution caps. Do not allow one player such as ResBot to dominate the corpus. Do not expose player username/identity to the deployed policy unless identity conditioning becomes a separate predeclared experiment. Train for transferable strategy. Version replay snapshots immutably with hashes/provenance.

## EXTERNAL STRENGTH TELEMETRY

Continue the leaderboard/external-strength data plane. Track public metrics only when actually exposed by the API. Persist timestamped leaderboard snapshots so useful derived metrics can include: current rating; current rank; rating change per match; rating change per 100 games; rating velocity over time; rank velocity; sample efficiency; strength improvement per million transitions; strength improvement per GPU-hour; optionally strength improvement per dollar; drawdown/volatility; catastrophic forgetting against frozen populations.

Do not make public leaderboard position the sole scientific objective. Maintain BOTH: PUBLIC LADDER = ecosystem/real-world evidence. INTERNAL FROZEN STAGE-2 EVALUATOR = controlled scientific evidence.

Generalisation reporting should expose the vector before inventing a magic scalar: frozen baseline performance; previous champion performance; held-out elite performance; held-out player gap; held-out style gap; unseen seed/map performance; replay vs matched control gap; catastrophic forgetting; public rating trajectory.

## STAGE 4B

Continue Stage 4B dependency-safe work concurrently with long Stage-4A GPU experiments. Already-delivered 4B work is not evidence that ALL of Stage 4B is complete. Continue canonical useful work including where still outstanding: deterministic packaging/outbox; immutable archive/provenance; registry-backed APIs; dashboard backend integration; replay/dataset registry views; leaderboard/generalisation telemetry APIs; supported-command inventory; distribution provenance; safe cleanup/migration planning; compatibility contracts.

Do not mutate live-training-critical files from a parallel 4B lane unless necessary for a verified defect. Infrastructure must not unnecessarily delay science.

When GPU training is healthy and substantial dependency-safe local work exists, do that work instead of spending the iteration merely polling the GPU. When no meaningful local independent work exists and a remote job is healthy, end THIS LOOP ITERATION cheaply. The durable loop will return later.

## STAGE 4 → STAGE 5 TRANSITION

Do not require every optional UI/platform polish task to finish. Enter Stage 5 when the actual canonical scientific/infrastructure gates required by Stage 5 are satisfied. Stage 5 inherits the strongest evidence-backed Stage-4 training recipe/data mechanisms rather than forgetting Stage-4 conclusions.

Stage-5 research includes, according to canonical authority: stronger teacher capacity; patch-transformer architecture; richer legal temporal/global history; critic/value architecture; scalar vs distributional/HL-Gauss-style value; reward × value interactions; replay pretraining where Stage-4 evidence supports it.

Training teachers may exceed deployment constraints. Do not prematurely shrink architecture simply because the final CPU student must be small. Stage-4 conclusions are strong priors, not eternal laws. Reopen a Stage-4 variable only for a specific predeclared architecture-interaction hypothesis.

## STAGE 6

After Stage-5 gates, continue automatically into strategic/control research. Canonical directions may include: castle build/no-build counterfactual preference; successor-value supervision; targeted/asynchronous DAgger; legal belief-state observer; pure neural vs residual heuristic vs neural reranker; deterministic safety masks; MPC / Expert Iteration; static vs Lagrangian vs PI/PID risk/control mechanisms.

Preserve PPO semantics. Do not mutate actions after rollout while pretending they were produced by the unchanged on-policy distribution. Every action-changing intervention must declare its semantics.

Replay-derived elite disagreement states may be used as state-selection/data sources here.

Successful Stage-6 interventions automatically become Stage-7 lineage ingredients if they: improve external gameplay; pass statistical promotion; preserve semantics; satisfy reliability; satisfy relevant runtime/latency gates. Training-only successes may be retained as experts/data generators without shipping in the final CPU bot. Failed/neutral mechanisms are rejected.

## STAGE 7

Scale only strong evidence-backed lineages. Use population training with suitable combinations of: current champion; historical champions; fixed baselines; heuristics; promoted Stage-4/5/6 candidates; qualified elite replay/style clones; strategy-cluster representatives; hard-counter specialists.

Do not let a small set of named elite clones dominate the distribution. Test adaptive sampling/PFSP only after suitable fixed-population baselines exist.

Reference scaling points may include: 5M, 25M, 50M, 100M, 250M, 500M, possibly 1B+ transitions. These are decision points, not mandatory expenditure and not artificial ceilings.

Continue scaling when: learner health remains valid; external strength improves; generalisation remains healthy; marginal strength per GPU-hour remains worthwhile. Kill dominated/plateaued lineages.

Distill promoted teachers into deployable CPU students where necessary. Track: teacher strength; student strength; distillation gap; CPU latency/p99; memory; protocol faults; gameplay strength. The final objective is strongest qualified DEPLOYABLE bot, not largest teacher.

## PARALLELISM

Use available Qoder agents/subagents/worktrees intelligently. Long remote training is an opportunity for independent work.

Safe conceptual architecture: QODER LEAD ├── research lane (remote GPU experiments/evaluation) ├── Stage-4B/data lane (platform/replay/provenance work) └── deterministic resource safety (RunPod watchdog/self-stop).

Do not parallelise conflicting file modifications. Do not generate activity merely for appearance. Use isolated ownership and tests.

## STATE MANAGEMENT

ACTIVE_STATE should remain the resumable executive state, not a human diary. At coherent boundaries update: canonical current stage/family; active experiments; active resources; latest evidence IDs; blockers; NEXT_SAFE_ACTION; relevant parallel lanes.

Do NOT create a competing second state system simply for this loop. If ACTIVE_STATE is stale: empirical artefacts/process state outrank stale prose; reconcile it.

The Stage-3 registry remains identity/lineage authority. Evidence ledger remains scientific-history authority. RunPod resource ledger remains paid-resource ownership authority.

## GIT

At coherent units: test → evidence/state → commit → push. Use safe branches/worktrees where required. Repair CI when within programme authority. Create/merge coherent PRs. Eventually merge/sync canonical main. Do not force-push. Do not lose unique evidence. Do not stop to ask routine Git permission already granted by the Marathon contract.

## RESPONSIBILITY SEPARATION

Retain the existing Stop hook for local false-completion protection ("do not incorrectly claim the Marathon is finished"). Do NOT abuse it as an infinite sleep/polling mechanism. This /loop task is the time-separated continuation mechanism. REMOTE SELF-STOP is the paid-compute protection mechanism.

STOP HOOK = prevent false completion. DURABLE /loop = come back and continue later. ACTIVE_STATE + REGISTRY = know what to continue. RUNPOD SELF-STOP/WATCHDOG = prevent idle billing. EXECUTION LEASE = prevent duplicate executors. Do not replace these with another sentinel text loop.

## LOOP BEHAVIOUR WHEN THERE IS NOTHING TO DO YET

If a healthy remote experiment is running and all dependency-safe useful local work is exhausted: confirm resource ownership/heartbeat; confirm remote self-stop protection where applicable; record nothing unnecessary; release/heartbeat lease correctly; finish THIS iteration. That is NOT Marathon completion. The durable loop will invoke this prompt again.

## LOOP BEHAVIOUR WHEN WORK FINISHES

If training/evaluation finished: preserve → stop paid resource → adjudicate → registry → evidence → ACTIVE_STATE → determine NEXT_SAFE_ACTION → EXECUTE NEXT_SAFE_ACTION during the SAME iteration where practical. Do not merely report "next family is X." Start X if its prerequisites are met. A stage/family boundary is a continuation point.

## LOOP BEHAVIOUR ON FAILURE

Use bounded repair. Do not wait for operator input unless the blocker genuinely requires unavailable human credentials/legal consent/external information and there is no independent work remaining. Continue independent work before classifying the entire Marathon blocked.

## LOOP BEHAVIOUR ON WINDOWS/QODER RESTART

The loop is durable, but local computation cannot execute while Windows itself is actually suspended. On Qoder/Windows restart or resume: RUNPOD RECONCILIATION FIRST. If a remote workload completed while the laptop slept: preserve/adjudicate its outputs and continue. Remote self-stop must minimise idle GPU exposure independently of laptop availability. Do not claim Windows suspension can execute local Qoder code. Do not confuse screen lock with Marathon completion.

## CURRENT RESEARCH

Discover the current real position from ACTIVE_STATE and running processes. At the time this recurring executor was commissioned, curriculum training had completed and a curriculum gameplay evaluation had been launched, with elite replay infrastructure active and Stage 4B partially delivered. DO NOT assume that remains true.

If curriculum gameplay evaluation is complete: adjudicate it under its predeclared rules and continue immediately. If it produces no meaningful separation: close the relevant curriculum question cleanly and execute the next canonical family, reportedly raw-vs-EMA if that remains NEXT_SAFE_ACTION. If it produces a valid promoted candidate: propagate that ingredient downstream according to the programme.

## CONTINUOUS RESEARCH CONTRACT

The durable loop exists so the operator does NOT need to babysit this. Every invocation should answer internally: "What is the highest-value safe canonical work that can be executed RIGHT NOW?" and then execute it. Do not produce a terminal conversational summary as a substitute for advancing reachable work. Reports/checkpoints are fine, but after recording them continue if another canonical action is immediately executable.

Only retire/cancel this durable recurring loop when ONE of these is true:

A. CANONICAL COMPLETE: Stage 4A/4B reachable programme → Stage 5 → Stage 6 → Stage 7 → final candidate qualification → adversarial review → repairs → integration → checks → merge → synced main → post-merge proof → final evidence reconciliation is actually complete.

OR

B. GENUINE ALL-SCOPE HARD BLOCKER: No scientifically/engineering-useful independent authorised work can proceed without an external human-only action.

A negative experiment is NOT a blocker. A stage boundary is NOT a blocker. A completed GPU run is NOT a blocker. A failed candidate is NOT a blocker. A Goal/Quest ending is NOT a blocker.

EXECUTE FROM LIVE STATE NOW.
