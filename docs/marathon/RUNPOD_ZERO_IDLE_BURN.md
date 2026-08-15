# RUNPOD ZERO-IDLE-BURN INVARIANT

Status: ACTIVE operator amendment `RUNPOD-ZERO-IDLE-BURN-2026-08-15`
(supplements `RUNPOD-SPEND-2026-08-15`). Provenance: the SH-R2/A100 idle
incident (EV-0028) showed end-of-round cleanup alone is insufficient.

**Invariant:** no paid RunPod GPU resource may remain RUNNING without a
verified active workload (or a short bounded startup/setup window). PAY FOR
COMPUTE WHILE IT IS COMPUTING — never for idle shells, completed jobs,
forgotten retries, local analysis time, or experiments awaiting attention.

## Operating rules

1. **Ownership ledger.** Every paid resource has a record in
   `experiments/marathon/runpod_resources.json`: pod id, owner experiment/run,
   expected workload/command, start time, last verified heartbeat, expected
   completion, billing rate, disposition after completion. A RUNNING pod with
   no active Marathon workload is an ORPHAN and must be stopped after
   verification.
2. **Verify the workload, not the pod status.** RUNNING ≠ working. Verify via
   trainer PID, advancing telemetry, GPU utilisation, output growth. A RUNNING
   pod whose workload completed/crashed/stalled is IDLE.
3. **Automatic watchdog.** While any paid pod is RUNNING, reconcile at bounded
   cadence without waiting for the human:
   `python scripts/dev/runpod_idle_watchdog.py` (report) or
   `--stop-idle` (capture → judge completed/failed → preserve artefacts →
   stop → billing-log → repair/resume only when needed).
4. **Completion means STOP.** On budget reached / elimination / success /
   unrecoverable failure: flush telemetry → save checkpoint → transfer and
   verify artefacts → STOP POD → adjudicate locally. No lengthy analysis
   while the GPU idles.
5. **Transfer never justifies burn.** Fetch results promptly; if analysis is
   slow and persistent storage already holds the outputs, stop the compute
   first. GPU compute and persistent storage are different resources.
6. **Bounded startup window.** A freshly provisioned pod may idle only while
   validation/sync/compile/restore/smoke is actively occurring; stalled setup
   → stop.
7. **No accidental duplicate capacity.** Before any start/create, query all
   resources; never two paid accelerators for one single-GPU workload unless
   parallel execution was deliberately predeclared. If a fallback takes
   ownership while the preferred resource later frees, choose one owner and
   stop the unused one immediately.
8. **Fallback retries cancel themselves.** Touch
   `var/marathon_takeover/stop_pod_retries` when a fallback owns the workload;
   `runpod_start_retry_loop.py` exits on the marker (exit 4).
9. **Stage gating includes lifecycle.** A bounded task is not complete while
   unexplained paid compute runs, finished workloads are unstopped, billing
   events are unrecorded, or ACTIVE_STATE misstates remote workloads.
10. **Stop-gate integration.** `.qoder/hooks/marathon_stop_gate.py` blocks a
    session stop while the resource ledger lists RUNNING/UNVERIFIED resources.
11. **Session recovery order.** After any interruption/restart, FIRST:
    query RunPod → enumerate paid resources → correlate with ledger/registry →
    heartbeat-check workloads → stop orphans/finished → preserve active ones →
    THEN continue research. Never assume pre-interruption state is current.
12. **Cost is telemetry, not the target.** Do not shrink scientifically
    justified training to save money; the spend policy remains
    `RUNPOD-SPEND-2026-08-15`.

Evidence: EV-0028 (incident reconciliation), billing log
`var/marathon_takeover/runpod_billing_log.jsonl`.
