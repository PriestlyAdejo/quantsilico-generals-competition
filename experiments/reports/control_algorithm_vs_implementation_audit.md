# Control Algorithm vs Implementation Audit

**Kind:** `CONTROL_ALGORITHM_VS_IMPLEMENTATION_AUDIT`  
**Created:** 2026-08-06  
**Task scope:** Write-only audit and plan amendment. No training, benchmarking, packaging, controller experiments, GPU jobs, overnight execution, uploads, or Phase 10.

## Verdict

**Yes — Stages 13–16 of the frozen V4.3 daytime-to-package plan are vulnerable to false controller rejection** if a slow prototype’s deadline faults or p99 failures are treated as evidence that the controller logic is strategically bad.

**This does not block Stages 0–12.** Stage 12 freezes the uncontrolled base (`BASE_DAYTIME_UPLOAD_READY` → immutable `qualified.json`) before optional controls. A controller failure must leave that base eligible for final recommendation.

**This audit does not automatically modify the running V4.3 programme.** The addendum supersedes only Stages 13–16 after an explicit `CONTROL_STAGE_PLAN_RESOLUTION_GATE`. If the amendment is not VALID at Stage 13, the correct action is `SKIP_ACTION_CHANGING_CONTROLS_KEEP_BASE` — not the parent’s thin controller stages.

## Audited identity (Option A — frozen baseline)

```text
audited_source_commit = 2df3c6a33db3a9d9f31b55a93ee6685985f85b97
audited_plan_path     = plans/v4_3_daytime_to_package_5333a24e.plan.md
audited_plan_sha      = 7d277bfd61853cfad827443ced1db925e9dd6cfc5d300f91b99ec34c67e62feb
source_manifest_sha   = 7977e753e2efad1257d58d0a1fa2f58006c418f74696d96d54863acdb9e276ff
env_implementation_hash = 4d8660fde2da633629b3bf6f40cb3d313a96bd3c30054571236ace5e6d985353
```

Sources: `experiments/manifests/competition_native_jax_v4_3_programme_state.json`, `experiments/manifests/competition_native_jax_v4_3_source_snapshot.json`.

Note: the V4.3 snapshot records `dirty_worktree: true`. This audit inspects committed plan artefacts and the control stub at the recorded commit identity; it does not claim the entire dirty worktree is bit-identical to that commit.

## Amendment artefacts produced

| Artefact | Path |
|---|---|
| Amended Stages 13–16 plan | `plans/v4_3_control_two_verdict_c7e2f91a.plan.md` |
| Cursor copy | `C:\Users\pries\.cursor\plans\v4.3_control_two_verdict_c7e2f91a.plan.md` |
| Amendment manifest | `experiments/manifests/control_algorithm_vs_implementation_amendment.json` |
| This report | `experiments/reports/control_algorithm_vs_implementation_audit.md` |

Parent plan **not edited**.

```text
amendment_plan_id     = v4_3_control_two_verdict_c7e2f91a
amendment_plan_sha256 = 01c42ecc7779d009641bda3dd1097e1b1758b086ed25ca2f11773ed99cdbcf89
```

### Authoritative relationship

```text
v4.3_daytime_to_package_5333a24e.plan.md
  → governs Stages 0–12 and Stages 17–19

v4_3_control_two_verdict_c7e2f91a.plan.md
  → supersedes only Stages 13–16
```

---

## Nine required answers

### 1. Is the existing plan vulnerable to false controller rejection?

**Yes.** Parent Stages 13–17 state only: passive noninterference first; action-changing if evidence gate PASS; if control fails, keep qualified base. There is no separation of:

- controller decision quality (policy benefit), from
- package latency / deadline / size feasibility (implementation).

A strategically useful controller first implemented as slow Python could lose games via deadline faults or fail p99 and be summarised as “control failed,” conflating algorithm and implementation.

Additional conflation risk: daytime evaluation protocol `tie_break` includes `lower_package_p99` (`experiments/manifests/competition_native_jax_daytime_evaluation_protocol_v1.json`).

### 2. Which controller implementations currently use Python loops, NumPy, JAX, or another backend?

| Component | Backend | Status |
|---|---|---|
| `control_off` | NumPy identity | Stub in `src/generals_bot/controls/__init__.py` |
| `apply_static_risk` | NumPy logit bias / PASS boost | Stub; not wired into package inference |
| `passive_telemetry_noninterference` | NumPy; records stats only | Unit-tested; not in live policy path |
| PI / barrier / bounded MPC / CONTROL_ON | — | **Not implemented** |
| Heuristic `phase_controller` / `SurvivalShield` | Pure Python | Separate always-on heuristics; **not** Stage 13 control ladder |

No JAX controller exists. No full 3,970-action Python-loop controller exists yet (families not implemented). Package policy remains NumPy CPU; `jaxlib_packaged: false`.

### 3. Measured or currently available latency evidence

| Source | Evidence |
|---|---|
| `competition_native_jax_early_deployment_gate.json` | Teacher NumPy steady p50/p95/p99 ≈ 3.06 / 4.16 / 4.16 s; cold ≈ 4.33 s; `DEPLOYMENT_REQUIRES_DISTILLATION` |
| `competition_native_jax_early_student_feasibility.json` | Teacher emb192 p99 ≈ 3.42 s (TOO_SLOW); student emb96_d2_h4 p99 ≈ 31 ms (OK) |
| `competition_native_jax_deployment_limits.json` | Official ordinary deadline 150 ms; promotion/target p99 **100 ms**; `headroom_frozen_before_candidate_latency: true` |
| Controller-specific `L_total` / `ΔL_controller` | **Absent** — not yet measured |

### 4. Recommended research backend (Stage C2)

- **Default for STATIC_RISK / PI / barrier:** vectorised NumPy (parity with deployment math).
- **MPC research screening:** batched JAX **or** vectorised NumPy, offline only.
- **Alternative:** precomputed controller-decision replay from frozen policy outputs.
- **Never:** unoptimised package play as the first strategic verdict.

### 5. Recommended deployment backend (Stage C3)

Vectorised NumPy, bounded scalar Python with preallocated buffers, lookup tables, precomputed geometry, or compact top-K.

**Default: no JAX/jaxlib in the ZIP.** Do not assume JAX wins for single-turn CPU inference.

### 6. Exact plan amendments

Separate addendum `plans/v4_3_control_two_verdict_c7e2f91a.plan.md` replaces thin Stages 13–16 with:

1. `CONTROL_STAGE_PLAN_RESOLUTION_GATE`
2. C1 passive noninterference
3. C2 policy-benefit (deadline-fault-free)
4. C3 implementation optimisation + latency decomposition + research/deployment parity
5. Controlled package qualification only if both verdicts PASS
6. Locked statuses including `CONTROL_ALGORITHM_PROMISING_IMPLEMENTATION_BLOCKED`
7. Decision matrix and family rules (STATIC_RISK / PI / barrier / MPC)

Parent `5333a24e` left frozen.

### 7. Files that change (this task)

- `experiments/reports/control_algorithm_vs_implementation_audit.md` (this file)
- `experiments/manifests/control_algorithm_vs_implementation_amendment.json`
- `plans/v4_3_control_two_verdict_c7e2f91a.plan.md`
- Cursor mirror of the amendment plan

No edits to `plans/v4_3_daytime_to_package_5333a24e.plan.md`, env/learner/package source, or `src/generals_bot/controls/` beyond inspection.

### 8. Tests that would be required (later Stage 13+; not implemented here)

1. C1 exact noninterference (actions, support, recurrent, RNG, protocol)
2. C2 paired benefit; prototype deadline faults classified as implementation, not benefit FAIL
3. Research↔deployment parity fixtures
4. Latency decomposition CONTROL_OFF vs ON
5. Promotion matrix classification (all six rows)
6. No-JAX-in-ZIP for controlled package
7. Handoff gate: VALID / NOT_PRESENT / PARENT_MISMATCH / INVALID

### 9. Does this issue block daytime-to-package execution?

**No** for Stages 0–12.

**Yes for promoting a controlled package** until the amendment is installed and `CONTROL_AMENDMENT_VALID`.

**If amendment missing/invalid at Stage 13:** skip action-changing controls; keep immutable qualified base; continue to one recommendation. Do **not** run the old underspecified control stage.

---

## CONTROL_STAGE_PLAN_RESOLUTION_GATE

At end of Stage 12, the running programme must resolve the amendment manifest and verify parent/amendment SHA match.

| Outcome | Meaning |
|---|---|
| `CONTROL_AMENDMENT_VALID` | Execute amended Stages 13–16 (C1→C2→C3) |
| `CONTROL_AMENDMENT_NOT_PRESENT` | Amendment missing |
| `CONTROL_AMENDMENT_PARENT_MISMATCH` | Parent SHA mismatch |
| `CONTROL_AMENDMENT_INVALID` | Manifest/plan corrupt or incomplete |

```text
if CONTROL_AMENDMENT_VALID:
    execute amended Stages 13–16
else:
    SKIP_ACTION_CHANGING_CONTROLS_KEEP_BASE
    continue to Stage 17 recommendation with qualified base
```

Creating the amendment files alone does not adopt them into a running agent’s context. Before Stage 13: copy/cherry-pick approved artefacts into the programme worktree, record hashes in programme-state at a checkpoint boundary, then resolve the gate. Do not change model, environment, or learner source files when integrating.

---

## Concurrent execution with running V4.3

| Concern | Guidance |
|---|---|
| GPU / compute | Audit is write-only; negligible interference |
| Same worktree | Not ideal (manifest races, snapshot contamination, hash drift) |
| Preferred | Separate Git worktree or clone for audit production |
| Integration | Before Stage 13 only; planning/manifest files only |

Example:

```powershell
cd C:\Users\pries\Documents\Projects\quantsilico-generals-competition
git worktree add ..\quantsilico-control-audit -b audit/control-two-verdict
```

---

## Required conceptual separation (locked)

### A. Controller policy-benefit verdict

Does the controller select better actions than `CONTROL_OFF`? Evaluated without avoidable prototype overhead dominating the result.

### B. Controller implementation-feasibility verdict

Can the same logic run in the real CPU package within latency, memory, size, and parity limits?

Promotion only when both PASS (and package qualification PASS).

Important classification:

```text
CONTROL_ALGORITHM_PROMISING_IMPLEMENTATION_BLOCKED
```

Means: improved decisions under paired evaluation, but current implementation missed deployment limits. Not “the controller is not useful.”

### Decision matrix

| Policy benefit | Implementation | Result |
|---|---|---|
| Pass | Pass | Controlled package may enter confirmation |
| Pass | Fail | Promising algorithm, implementation blocked; keep base |
| Pass | Not optimised | Optimise within frozen budget; keep base |
| Fail | Pass | Fast but strategically useless; reject controller |
| Fail | Fail | Reject controller |
| Inconclusive | Any | Do not promote; keep base |

---

## Base-package protection (confirmed)

Parent Stage 12 freezes uncontrolled base before controls. Control optimisation must use a separate candidate/build path. Failed control must not erase ZIP/SHA/`qualified.json`.

---

## Explicit non-goals completed as non-goals

- No PI / barrier / MPC implementation
- No edits to frozen parent plan
- No training, packages, uploads, overnight, or Phase 10
- No inventing controller latency numbers
