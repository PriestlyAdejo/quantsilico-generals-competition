# Marathon durable continuation loop — canonical resume prompt

Amendment `ELITE-REPLAY-INTELLIGENCE-DURABLE-CONTINUATION-2026-08-15` §33.
This is the versioned instruction a recurring Qoder loop/scheduled task must
execute on every wake. Target cadence: ~30 minutes (semantic continuation;
the deterministic RunPod watchdog handles frequent resource safety). Do NOT
wake an expensive model every minute; align with job ETAs where supported.

The recurring task terminates ONLY at canonical Marathon completion
(Stage 7 + final integration + post-merge proof) or a genuine ALL-scope hard
blocker. `TurnLimited` is NOT `MARATHON_COMPLETE`.

---

## QUANTSILICO MARATHON RESUME ITERATION

1. Acquire the Marathon execution lease:
   `.venv/Scripts/python.exe scripts/dev/marathon_execution_lease.py acquire --owner <iteration-id>`
   (exit 3 = another live owner → step 6; exit 4 = stale takeover, proceed and note it).
2. FIRST query RunPod and reconcile every paid resource against
   `experiments/marathon/runpod_resources.json`
   (`.venv/Scripts/python.exe scripts/dev/runpod_idle_watchdog.py`; stop-idle only after preservation).
3. Preserve healthy owned workloads (verified trainer PID + GPU util, not pod status).
4. Stop completed/orphaned/idle resources AFTER outputs are safely fetched; billing-log every stop.
5. Read the live truth: `experiments/marathon/ACTIVE_STATE.json`, canonical programme
   (`docs/marathon/EXECUTION_PLAN.md` + family charters), Stage-3 registry
   (`experiments/marathon/registry`), `docs/marathon/EVIDENCE_LEDGER.md`, current Git state.
6. If another valid Marathon executor owns the lease: exit cleanly (heartbeat nothing).
7. If a remote workload is still healthy: do NOT restart it; execute dependency-safe
   local/Stage-4B work if worthwhile; otherwise finish this iteration cheaply.
   Heartbeat the lease during long work.
8. If a workload completed: fetch/preserve results → STOP compute → adjudicate against
   the PREDECLARED rules → registry/evidence/ACTIVE_STATE → NEXT_SAFE_ACTION.
9. If a workload failed: classify (engineering defect / design defect / genuine negative)
   and apply bounded repair authority; two failed attempts on the same problem escalate.
10. If no workload is active: execute `ACTIVE_STATE.NEXT_SAFE_ACTION`.
11. After any stage/family finishes: continue to the next dependency-safe canonical work.
12. Continue: Stage 4A + Stage 4B → Stage 5 → Stage 6 → Stage 7 → qualification →
    adversarial review → PR → merge → main → post-merge proof.
13. Release the lease at iteration end
    (`scripts/dev/marathon_execution_lease.py release`). Terminate the recurring task only
    when canonical completion is proven or a genuine ALL-scope hard blocker exists.

## Invariants that never lapse

- Zero-idle-burn: no paid RUNNING pod without verified active workload
  (`docs/marathon/RUNPOD_ZERO_IDLE_BURN.md`); RunPod reconciliation is FIRST on every resume.
- No post-hoc rule relaxation; predeclare before launch; register before launch;
  telemetry never promotes — gameplay evaluation is the arbiter.
- Do not duplicate capacity, experiment arms, adjudications, Git writes, or
  ACTIVE_STATE mutations (lease enforces).
- Preserve unique evidence; cost is telemetry, not the optimisation target.
