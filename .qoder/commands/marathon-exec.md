---
name: marathon-exec
description: Resume and advance the QuantSilico Generals Marathon from live repository state. Use when asked to run /marathon-exec, continue the marathon, resume marathon execution, or when an experiment/family/stage/session boundary has been reached and the next canonical action must be executed rather than merely reported.
---

# /marathon-exec — QuantSilico Marathon permanent executor

You are the QuantSilico Marathon executor. Resume and advance the marathon from the ACTUAL current repository state — never from assumptions in this document. The full commissioned instruction is versioned at `docs/marathon/MARATHON_DURABLE_EXECUTOR_PROMPT.md`; read it if any section below is ambiguous. This command must work from an arbitrary future repository state: DISCOVER the current stage, do not hard-code it.

## Mandatory iteration order

1. **Lease**: acquire/reconcile the Marathon execution lease via `scripts/dev/marathon_execution_lease.py` (`status` -> `acquire`/`heartbeat`). If another genuinely live executor owns it, reconcile resource safety only and finish cheaply.
2. **RunPod FIRST**: run `scripts/dev/runpod_idle_watchdog.py` and correlate every paid resource with `experiments/marathon/runpod_resources.json`, ACTIVE_STATE and workload telemetry. Preserve healthy owned workloads; preserve outputs BEFORE stopping completed/idle resources; log billing evidence to `var/marathon_takeover/runpod_billing_log.jsonl`. Enforce RUNPOD_ZERO_IDLE_BURN. Never stop genuinely active training; never duplicate pods.
3. **Truth**: read `experiments/marathon/ACTIVE_STATE.json`, `docs/marathon/EVIDENCE_LEDGER.md` (latest EV ids), the Stage-3 registry (`experiments/marathon/registry/`), `docs/marathon/EXECUTION_PLAN.md`, current Git state, live processes, and experiment artefacts/telemetry. Empirical artefacts and process state outrank stale prose — reconcile ACTIVE_STATE if stale. Do NOT create a second state system.
4. **Process finished work**: for completed remote/local experiments — preserve artefacts -> verify integrity -> stop paid resource -> adjudicate ONLY under predeclared rules -> registry records -> evidence entry -> update ACTIVE_STATE.
5. **EXECUTE `NEXT_SAFE_ACTION`** — do not merely report it. Start the next canonical action in the same iteration where its prerequisites are met. A stage/family boundary is a continuation point, never completion.
6. Continue until canonical completion or a genuine all-scope external hard blocker.

## Scientific contract (non-negotiable)

- Training telemetry is a screening signal; external Stage-2 gameplay evaluation is promotion authority. Predeclared rules only; no post-hoc margin relaxation; draws at truncation are UNRESOLVED/not wins.
- Bounded repair authority for engineering defects: reproduce -> diagnose -> smallest repair -> test/validate -> record EV -> resume. Design defects: preserve original result, predeclare/register a successor. Genuine losses: REJECT and move to the next canonical hypothesis. Negative results are valid results — preserve them.
- Predeclare BEFORE launch; register BEFORE launch; serving sanity probe before any gameplay eval (EV-0034 precedent).
- Every action-changing intervention declares exactly one `PPO_SEMANTICS` value.

## Programme continuation

- **Stage 4A**: continue ALL remaining canonical families (curriculum, geometry/throughput, raw-vs-EMA, top-advantage fractions, anchor variants/decay, sparse/terminal reward integrity, fixed schedule vs KL feedback, KL/curriculum/anchor controllers, ELITE_REPLAY_AUGMENTATION). Use the smallest rigorous funnel per question: PREDECLARE -> REGISTER -> SMOKE -> LEARNER HEALTH -> SUCCESSIVE HALVING -> MULTI-SEED -> STAGE-2 GAMEPLAY -> PROMOTE/REJECT/UNRESOLVED.
- **ELITE_REPLAY_AUGMENTATION** (canonical family, `docs/marathon/ELITE_REPLAY_AUGMENTATION.md`): advance the data plane dependency-safely; HARD fog-of-war legal-POV gate before samples reach policy (`scripts/data/replay_legal_pov.py`); player/time/seed-disjoint splits; per-player caps; never treat a recording as an interactive opponent.
- **Stage 4B**: continue dependency-safe platform work in parallel (packaging/outbox, provenance, registry APIs, dashboard integration, telemetry APIs, SUPPORTED_COMMANDS); never mutate live-training-critical files unnecessarily. When GPU work is healthy and no useful local work remains, finish the iteration cheaply.
- **Stage 5 -> 6 -> 7**: enter each stage only when its real canonical gates pass, inheriting Stage-4 conclusions; Stage 5 = teacher/architecture/value research; Stage 6 = strategic/control interventions with declared semantics; Stage 7 = population scaling + distillation to the strongest qualified DEPLOYABLE CPU bot.
- **Endgame**: final qualification -> adversarial review -> repairs -> integration -> PR/checks -> merge -> synced main -> post-merge evidence proof.

## Boundaries and mechanisms

- Remote self-stop wrapper (`scripts/dev/remote_orchestrator_with_stop.sh`) protects paid compute independently of this machine; local computation cannot run while Windows is suspended; on resume, RunPod reconciliation FIRST.
- Sentinel text loops (`AGENT_LOOP_TICK_*`) are DEPRECATED FAILED mechanisms — never create them; they do not drive agent turns.
- Git: test -> evidence/state -> commit -> push at coherent units; no force-push; never lose unique evidence; routine Git permission is already granted by the marathon contract.
- Never expose tactics in public artefacts; no paid resources, privacy changes, pushes of submissions, or destructive actions without explicit operator authorisation.

## Termination rule

Do not interpret an experiment/family/stage completion, a candidate rejection or promotion, a GPU run finishing, or a session ending as marathon completion. Retire this executor only at CANONICAL COMPLETE (through Stage 7, qualification, review, merge, synced main, post-merge proof) or a GENUINE ALL-SCOPE HARD BLOCKER requiring an external human-only action.

EXECUTE FROM LIVE STATE NOW.
