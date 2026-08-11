# Phase 9FU Super-Prompt Source Brief

**Purpose:** Hand this entire document to another AI so it can generate one authoritative Cursor Agent prompt (or a sequenced set of prompts).  
**Constraint for that AI and for any executor:** This file is evidence + instructions for prompt generation only. Do **not** treat reading this brief as permission to implement, kill processes, upload, train, or mutate the portal.

**Repo snapshot time (local):** 2026-08-05 ~15:48+01:00  
**Workspace:** `C:\Users\pries\Documents\Projects\quantsilico-generals-competition`

---

## 0. One-sentence mission for the prompt author

Produce a Cursor Agent prompt that: (1) correctly stops or safely finishes the current opaque paired evaluator according to the decision rule below, (2) records Tactical V2 portal rejection + local diagnostic status, (3) repairs `scripts/phase9fu_paired_eval.py` for progress/checkpoint/resume without changing frozen protocol semantics, (4) prioritises Hybrid BC for Stage 3 recommendation, (5) keeps `UPLOAD_THIS` = no candidate until gates pass, (6) encodes the full post-recovery Phase 9G order including mandatory pre-overnight daytime PPO, and (7) never auto-uploads or starts overnight/PPO/control without explicit later gates.

---

## 1. Identity model (must appear in any generated prompt)

```text
P9FU / candidate ID  = what was built (technical identity)
PUBLIC version ID    = what was uploaded and deployed
Runtime policy key   = code constructor name inside the package
Build hash           = first 16 hex of package SHA-256; selects exact ZIP
Role JSON            = pointer (live / recommended / rejected / research), not a bot
```

Upload rule:

```text
Upload candidate ZIPs.
Record successful uploads as public versions.
Upload only the exact path + SHA in submission/UPLOAD_THIS.md
  AND submission/roles/recommended.json.
```

Never conflate:

- `best_overall` / `safe_fallback` / `upload_ready` / `portal` with candidate IDs
- `QS-PUBLIC-V00N` with a second package that must be built separately

Lifecycle example:

```text
QS-P9FU-HYBRID-BC-V1  --(manual upload of package.zip)-->  QS-PUBLIC-V002
```

---

## 2. Authoritative current position (repo-verified)

| Item | Status | Evidence |
|------|--------|----------|
| V001 | Live, qualified, strategically weak | `submission/roles/live.json`, freeze gate PASS |
| UPLOAD_THIS | `NO_CANDIDATE_CURRENTLY_RECOMMENDED` | `submission/UPLOAD_THIS.md` |
| recommended.json | same | `submission/roles/recommended.json` |
| rejected.json | empty (`NONE`) | `submission/roles/rejected.json` — **needs portal rejection write** |
| Hybrid BC | Packaged + BEHAVIOURAL_PASS; local paired eval **in progress** | packages + behavioural gate + `_paired_eval.log` |
| Tactical V2 | Packaged + BEHAVIOURAL_PASS; **local direct finished**; **portal NOT_QUALIFIED** | package + log + user portal evidence |
| Paired evaluator | Running, opaque until candidate completes | script + process list |
| Throughput | ~4 TPS, non-blocking docs only | `phase9fu_throughput_lane.json` |
| Challengers result file | **Does not exist yet** | no `phase9fu_v001_vs_challengers.json` |

### 2.1 Live public baseline (do not re-upload)

```text
public_version:     QS-PUBLIC-V001
stable_candidate:   QS-P9F-PORTAL-V0
runtime_policy:     heuristic_v2f_plus_planner_terminal_force
build_hash:         e1237f77dee46993
sha256:             e1237f77dee469935fc3a60811b9a34522b83dd37bf4d76fa2555e6107a8edfa
package:            submission/packages/QS-PUBLIC-V001/e1237f77dee46993/package.zip
also at:            submission/packages/heuristic_v2_preppo_8f7405fe9834161c_packaged.zip
upload_local:       2026-08-04T00:37:00+01:00
status:             TECHNICALLY_QUALIFIED_STRATEGICALLY_WEAK_BASELINE
do_not_reupload:    true
```

Related non-live Windows repackage (not live):

```text
sha256: 898b37b104545fa6217877dd2db2af7c6e8810f41b4ba1f79cc8530b798d558e
```

### 2.2 Challenger packages

**Hybrid BC**

```text
candidate_id:   QS-P9FU-HYBRID-BC-V1
architecture:   hybrid_bc
build_hash:     5152a08eb774cf0e
sha256:         5152a08eb774cf0e29167e9469422834b0a6e40392a6035ccc0f830d50674b9f
path:           submission/packages/QS-P9FU-HYBRID-BC-V1/5152a08eb774cf0e/package.zip
size:           1222953 bytes
behavioural:    BEHAVIOURAL_PASS
qualification:  structural PACKAGED; windows_validation PENDING; linux_parity NOT_RUN; official_upload_ready false
role:           RESEARCH_CANDIDATE
```

Pipeline (from builder report):

```text
generate_proposals → canonicalize → legal_mask → one BC forward →
rank → confidence → SurvivalShield → else full-set SurvivalShield → verify legal
```

BC checkpoint used by local evaluator factory:

```text
experiments/phase9f_cnn_ranker_v1/checkpoints/bc/model.json
```

Note in builder report: Hybrid confidence defaults are **provisional**; `HYBRID_CONFIDENCE_CALIBRATION_GATE` mentioned as needed before challenger behavioural seeds — behavioural gate still recorded PASS.

**Tactical V2**

```text
candidate_id:     QS-P9FU-HEURISTIC-TACTICAL-V2
runtime_policy:   heuristic_v2f_tactical_attack_v2
build_hash:       22a6d05b7160d86f
sha256:           22a6d05b7160d86ff8d278da28d327d88a6236a4655d90a8bf99894473531770
path:             submission/packages/QS-P9FU-HEURISTIC-TACTICAL-V2/22a6d05b7160d86f/package.zip
behavioural:      BEHAVIOURAL_PASS
qualification:    structural PACKAGED; windows_validation PENDING; linux_parity NOT_RUN; official_upload_ready false
role:             RESEARCH_CANDIDATE
```

### 2.3 Portal evidence for exact Tactical V2 hash (user-verified upload)

Genuine new bot (`create_ablation("heuristic_v2f_tactical_attack_v2")`), not old V001.

```text
portal_upload_timestamp:  2026-08-05T14:57:00+01:00 (approx from user)
portal_faults:            0
portal_vs_expander:       DRAW, WIN, DRAW
portal_vs_hunter:         LOSS, LOSS, LOSS
portal_verdict:           NOT_QUALIFIED
```

Compare V001 historical portal (user):

```text
Expander: WIN, DRAW, WIN
Hunter:   LOSS, WIN, LOSS
verdict:  QUALIFIED
```

Consequences (must be encoded):

```text
PORTAL_REJECTED
INELIGIBLE_FOR_V002
DO_NOT_REUPLOAD
add to submission/roles/rejected.json
never put 22a6d05b... in UPLOAD_THIS.md
Elo rise on leaderboard is almost certainly still V001, not this package
```

### 2.4 Frozen evaluation protocol v1 (do not silently change)

File: `experiments/manifests/phase9fu_evaluation_protocol.json`

```text
protocol_version: v1
frozen: true
seeds: 101..116 (16 seeds)
seats: swap_within_pair
direct_vs_v001: 16 pairs / 32 games
per critical baseline: 8 pairs / 16 games
critical_baselines:
  official_expander, official_hunter, heuristic_aggressive, heuristic_defensive
castle_oriented: heuristic_castle when available
scoring: win=1, draw=0.5, loss=0
thresholds (key):
  min_direct_score_rate_improvement_vs_v001: 0.05
  max_opponent_suite_score_rate_regression: 0.05
  max_draw_rate_increase: 0.15
  max_protocol_faults: 0
```

Approx game budget per challenger: ≥96 games (32 + 4×16), more if castle baseline included.

If only observability/checkpointing changes → keep `protocol_version: v1` + record implementation amendment.  
If candidate scope or counts change → create `phase9fu_evaluation_protocol_v2.json` (never silently mutate v1).

### 2.5 Paired evaluator defects (code-verified)

File: `scripts/phase9fu_paired_eval.py`

Observed behaviour:

1. Prints only `evaluating {cid} ...` then one JSON summary **after the entire candidate** finishes.
2. No per-pair logging.
3. No atomic checkpoint / partial JSON / resume.
4. No `--candidate`, `--opponent`, `--resume-from`.
5. No max wall-time / heartbeat / SIGINT flush.
6. Writes `phase9fu_v001_vs_challengers.json` only at end.
7. Candidate order in factories: **Tactical V2 first, then Hybrid BC**.
8. Uses in-process policy factories (not necessarily the packaged ZIP runtime) — note for local-vs-portal divergence analysis.

### 2.6 Live process state at brief time

```text
PID 20748  parent launcher  ~3.6 MB  tiny CPU
  cmd: .venv-training\Scripts\python.exe -u scripts/phase9fu_paired_eval.py
PID 2040   worker           ~1.0 GB  CPU climbing (~1595s CPU)
  cmd: Python312\python.exe -u scripts/phase9fu_paired_eval.py
Started: 2026-08-05 14:38:38 local
```

`_paired_eval.log` content at ~15:48:

```text
evaluating QS-P9FU-HEURISTIC-TACTICAL-V2 ...
{"cid": "QS-P9FU-HEURISTIC-TACTICAL-V2", "direct": {"games": 32, "pairs": 16,
 "wins": 13, "draws": 6, "losses": 13, "score_rate": 0.5, "draw_rate": 0.1875},
 "suite_mean": 0.671875}
evaluating QS-P9FU-HYBRID-BC-V1 ...
```

Interpretation:

- Tactical V2 **local direct vs V001 = 0.5** → does **not** meet +0.05 improvement → not V002_ELIGIBLE under frozen thresholds (even before portal).
- suite_mean 0.671875 is strong vs baselines, but insufficient alone.
- Hybrid BC evaluation **started**; no Hybrid summary yet; no final manifest yet.
- Worker still doing useful work (CPU increasing), but remaining Hybrid run remains opaque for another long stretch.

### 2.7 Stage completion map (recovery plan)

Completed / present:

```text
✓ V001 historical correction + freeze
✓ submission/ cutover (layout gate artefacts present)
✓ Hybrid BC packaged
✓ Tactical V2 packaged
✓ Per-candidate behavioural gates PASS for both
✓ Throughput lane documented non-blocking (~4 TPS)
✓ Evaluation protocol frozen v1
~ Paired evaluation PARTIAL (Tactical summary logged; Hybrid in flight; no result file)
✗ Recommendation / UPLOAD_THIS still NO_CANDIDATE
✗ rejected.json not yet updated for portal failure
```

Not part of this recovery finish line (defer):

```text
full repo naming cleanup
Tier-1 ≥100 TPS
meaningful repaired PPO
PI / barrier / MPC
overnight
Phase 10 execution
```

---

## 3. Decision conflicts the prompt author must resolve explicitly

Earlier chat advice said **let the opaque run finish**.  
Later authoritative recovery advice said **stop only the evaluator**, classify abort, repair, prioritise Hybrid BC.

**Repo-updated decision rule for the generated prompt:**

```text
IF Hybrid BC has not yet printed its candidate summary
AND user still prefers not to wait blindly for a second opaque multi-hour block
THEN:
  stop evaluator process tree (paired_eval only)
  preserve logs/process metadata
  record ABORTED_OPAQUE_NO_CHECKPOINT for any incomplete candidate
  BUT also preserve the completed Tactical V2 summary from _paired_eval.log
    as DIAGNOSTIC_LOCAL_PARTIAL (not a competitive recommendation)
  repair evaluator
  resume Hybrid BC only with --candidate QS-P9FU-HYBRID-BC-V1

ELSE IF user chooses to let Hybrid finish because Tactical already completed:
  do not kill
  after terminal report, apply portal amendment + recommendation rules
  still repair evaluator before any future long eval
```

Default recommended by latest human recovery instruction: **stop → repair → Hybrid-first resumable eval**.  
Do not classify incomplete Hybrid results as competitive.  
Do not treat Tactical local summary as upload justification (portal already rejected exact hash; local direct = 0.5).

---

## 4. Immediate Stage 3 recovery actions (for the Cursor prompt)

### 4.1 Preserve then stop evaluator (if stop path chosen)

```powershell
New-Item -ItemType Directory `
  .\experiments\aborted_runs\phase9fu_paired_eval_20260805 `
  -Force

Copy-Item .\experiments\manifests\_paired_eval.log `
  .\experiments\aborted_runs\phase9fu_paired_eval_20260805\ `
  -ErrorAction SilentlyContinue

Copy-Item .\experiments\manifests\_paired_eval.err `
  .\experiments\aborted_runs\phase9fu_paired_eval_20260805\ `
  -ErrorAction SilentlyContinue

Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine |
  Out-File .\experiments\aborted_runs\phase9fu_paired_eval_20260805\processes.txt
```

Then kill only paired-eval tree (example PIDs from snapshot; **re-resolve before kill**):

```powershell
# Re-resolve PIDs; do not blindly use stale IDs
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*scripts/phase9fu_paired_eval.py*' } |
  ForEach-Object { taskkill /PID $_.ProcessId /T /F }
```

Do not kill unrelated Python (conda/Anaconda helpers observed on machine).

### 4.2 Record abort + portal rejection artefacts

Write:

```text
experiments/reports/phase9fu_paired_eval_abort.md
experiments/manifests/phase9fu_paired_eval_abort.json
experiments/reports/phase9fu_evaluator_recovery.md
experiments/reports/phase9fu_tactical_v2_portal_result.md
experiments/manifests/phase9fu_tactical_v2_portal_result.json
```

Abort classification for incomplete Hybrid (and for any incomplete suite):

```text
run_status: ABORTED_OPAQUE_NO_CHECKPOINT
reason: No per-pair progress / no resumable checkpoint / excessive unobservable runtime
local_candidate_result: NOT_EVALUATED_TO_COMPLETION   # for Hybrid if killed mid-run
```

For Tactical V2, record separately:

```text
local_diagnostic_from_log:
  direct_vs_v001 score_rate 0.5 (13/6/13), draw_rate 0.1875, suite_mean 0.671875
local_competitive_class: NOT_V002_ELIGIBLE under protocol v1 (+0.05 rule)
portal_verdict: NOT_QUALIFIED
final: INELIGIBLE_FOR_V002 / DO_NOT_REUPLOAD / PORTAL_REJECTED
```

Update:

```text
submission/roles/rejected.json  # add exact Tactical V2 hash
```

Ensure `UPLOAD_THIS.md` never references `22a6d05b7160d86f...`.

### 4.3 Evaluator repair requirements

Modify `scripts/phase9fu_paired_eval.py` to provide:

1. Per-pair progress: candidate, opponent, seed, pair index/total, seat, result, turns, elapsed
2. Atomic checkpoint after every completed pair
3. Partial files: `experiments/manifests/phase9fu_eval_<candidate_id>.partial.json`
4. `--resume-from <checkpoint>`
5. `--candidate <candidate_id>`
6. `--opponent <opponent_id>`
7. Max game wall time + deterministic timeout classification
8. Max candidate wall time
9. Heartbeat ≤60s: completed games/pairs, current game, RSS, elapsed
10. Graceful SIGINT: flush completed-pair checkpoint
11. **No change** to frozen seeds, maps, seats, thresholds, scoring

Add tests for resume/idempotency/progress logging.

### 4.4 Prioritisation after repair

```text
Tactical V2 exact hash = portal-rejected → do not spend main qualification budget on it
Hybrid BC = main possible V002 candidate
Run Hybrid with repaired evaluator --candidate QS-P9FU-HYBRID-BC-V1
Persist after each pair
```

Hybrid eligibility checklist (prompt must verify, not invent):

```text
[x] behavioural gate PASS (repo)
[?] clean-package gate — windows_validation still PENDING in qualification_report.json
[?] CPU/memory qualification — not evidenced as PASS in current qualification_report
[?] model parity — not evidenced in phase9fu hybrid qual report
[?] recurrent execution — hybrid uses BC CNN; confirm package/runtime gates separately
[?] candidate-set compatibility — both listed stage3_eligible
```

If any required package/CPU/parity gate is still PENDING, the prompt must either run those gates first or record BLOCKED — do not fake PASS.

If Hybrid passes preregistered thresholds:

```text
exactly one recommendation
update submission/roles/recommended.json
update submission/UPLOAD_THIS.md with exact ZIP + SHA
recommend QS-PUBLIC-V002
do not upload automatically
```

If Hybrid fails or under-sampled:

```text
keep NO_CANDIDATE_CURRENTLY_RECOMMENDED
```

Tactical may later get a **diagnostic-only** local eval for local-vs-portal divergence; never recommend exact `22a6d05b...`.

### 4.5 Post-run portal evidence amendment (if local Tactical results retained)

If local favours Tactical despite portal fail → classify `LOCAL_PORTAL_EVALUATION_DIVERGENCE` and investigate:

```text
simulator parity, official Hunter version, map generation, turn cap,
castle rules, observation semantics, belief-state, packaging vs local runtime,
deterministic reset
```

Do not resolve divergence by ignoring portal.

Current local evidence already does **not** favour Tactical for V002 (direct score_rate 0.5).

---

## 5. Upload / V002 rules

Right now:

```text
Upload nothing new.
Leave weak V001 live.
Do not re-upload V001.
Do not re-upload Tactical V2 22a6d05b...
```

Valid upload signal only when both agree on one exact package:

```powershell
Get-Content .\submission\UPLOAD_THIS.md
Get-Content .\submission\roles\recommended.json
```

Expected successful shape:

```text
Candidate: QS-P9FU-HYBRID-BC-V1   # (or a future eligible candidate — not current Tactical hash)
Package: submission/packages/<ID>/<BUILD_HASH>/package.zip
SHA-256: <full>
Recommended public version: QS-PUBLIC-V002
```

After manual upload → freeze as `QS-PUBLIC-V002` pointing at that exact candidate/hash.

---

## 6. Intended programme order (full roadmap for prompt sequencing)

```text
NOW: Phase 9FU Stage 3 recovery
  stop/repair opaque evaluator (decision rule §3)
  portal-reject exact Tactical V2
  resumable Hybrid BC eval
  recommend V002 only if thresholds pass
  user uploads manually if recommended
  freeze V002

THEN (only after 9FU hard stop + merges):
  optional repository normalisation (separate prompt)
  Phase 9G tracks in isolation:
    A passive telemetry
    B Tier-1 throughput ≥100 valid PPO TPS
    C1 mandatory daytime bounded repaired-PPO + package release gate
    C repaired PPO / replay (meaningful only after TIER1_PASS)
    D evidence-gated control (STATIC_RISK → PI → barrier → MPC)
  only then overnight readiness discussion
  Phase 10 proposal only; no execution without user approval
```

Release opportunities:

```text
Release 1: Hybrid BC (or other eligible challenger) → QS-PUBLIC-V002
Release 2: daytime repaired PPO package → QS-PUBLIC-V003
Overnight: only after both recovery package attempt AND daytime training gate
```

---

## 7. Mandatory TRACK C1 — pre-overnight bounded training (must be in Phase 9G prompt)

Insert after TIER1_PASS and before any overnight readiness:

```text
TRACK C1 — PRE-OVERNIGHT BOUNDED TRAINING AND RELEASE GATE

Mandatory after TIER1_PASS. Cannot skip to overnight.

Prerequisites:
- valid PPO learning transitions/sec >= 100
- repaired PPO correctness PASS
- canonical BC restart checkpoint
- invalid overnight PPO checkpoints excluded
- frozen public baseline
- frozen evaluation protocol
- package builder for learned candidate
- CPU/memory/package qualification available

Always restart from canonical valid BC checkpoint.
Never continue Phase 9F invalid overnight PPO / PASS-collapsed / unsupported trainer semantics.

Stage A correctness: ≤1024 transitions, ≤4 updates, ≤15 min
Stage B smoke: ≤4096 transitions, ≤16 updates, ≤30 min
Stage C daytime short: ≤25000 transitions, ≤90 min (supervised)
Stage D daytime medium: ≤100000 transitions, ≤4 h (supervised; not overnight)

Stop early on PASS/entropy collapse, support/recurrent mismatch, illegal, NaN/Inf,
BC competence loss, survival regression, package CPU failure.

Checkpoint compare vs: BC, frozen public, Tactical, Hybrid, Expander, Hunter,
aggressive/defensive baselines; frozen maps/seeds/seats.

PRE_OVERNIGHT_TRAINING_GATE: PASS|PARTIAL|FAIL|BLOCKED
PRE_OVERNIGHT_RELEASE_GATE: READY|BLOCKED

On PASS package QS-P9G-REPAIRED-PPO-DAY-V1 under submission/packages/...

OVERNIGHT_READINESS cannot be READY unless:
TIER1_PASS + repaired correctness + C1 training gate + C1 release gate +
resume verified + stop conditions + monitoring/recovery written +
explicit separate user approval.
```

Hard anti-pattern to forbid in prompts:

```text
100 TPS reached → start overnight   # FORBIDDEN
```

Required:

```text
100 TPS → 90 min proof → ≤4 h proof → package/evaluate → then discuss overnight
```

---

## 8. Phase 9G full track summary (post-9FU only)

### Precondition gate

Read UPLOAD_THIS, live/recommended roles, package/candidate registries, 9FU terminal report, evaluation protocol, v001_vs_challengers, behavioural + package quals, throughput + repaired-PPO manifests.

Classify: `PHASE9G_INPUT_READY | BLOCKED | PARTIAL`.

### Branches

```text
research/phase9g-passive-telemetry-v1
research/phase9g-tier1-throughput-v1
research/phase9g-repaired-ppo-v1
research/phase9g-control-interventions-v1
```

### Track A — Passive telemetry

Non-decision-changing metrics; `PASSIVE_NONINTERFERENCE_GATE` exact equality CONTROL_OFF with telemetry on/off.

### Track B — Tier-1 throughput

From ~4 → ≥100 valid PPO learning transitions/sec. Profile env/obs/support/belief/heuristics/neural/PyTorch/JAX bridge/rollout/GAE/PPO/optim/logging/ckpt. Optimisation order: remove Python hot path → padded tensors → vectorised env → vmap/jit/scan → batched encode/infer → larger batches → pinned/DLPack → device-resident buffers → less logging → async ckpt. Do not full-JAX port unless sync ≥20% after simpler batching. Pause during competitive measurements.

### Track C — Repaired PPO / replay

Only after TIER1_PASS (+ C1 daytime gate before overnight). Restart from canonical BC. Revalidate support/recurrent/burn-in/reset/terminal/policy-version/mixture/step-zero/resume/BC retention. Replay partitions with no game-ID overlap. Scale model only on under-capacity evidence (≈1M→3M→5M) with CPU/package gates.

### Track D — Control

No action-changing control without `ACTION_CHANGING_CONTROL_EVIDENCE_GATE`. Ladder: OFF → STATIC_RISK → PI → BARRIER → BOUNDED_MPC → COMBINED. Passive telemetry yes before overnight; one justified STATIC_RISK possible; PI/barrier/MPC probably after more evidence. Never claim formal safety without proof.

### Phase 10

Write readiness proposal only; classify READY/BLOCKED/NOT_EVALUATED; do not execute.

Hard stops everywhere:

```text
No auto-upload
No portal delete/mutate
No invalid overnight PPO continuation
No meaningful PPO below Tier 1
No controller without evidence
No post-hoc threshold changes
No rental / overnight without explicit approval
No Phase 10 execution
```

---

## 9. Post-run repository normalisation (separate later prompt)

Only after 9FU agents/evals/builders stopped and branches merged.

Canonical terms:

```text
Public: QS-PUBLIC-V###
Candidates: QS-P9FU-HYBRID-BC-V1, QS-P9FU-HEURISTIC-TACTICAL-V2, QS-P9G-REPAIRED-PPO-DAY-V1, ...
Runtime keys: hybrid_bc_v1, heuristic_tactical_v2, repaired_ppo_v1
Historical V001 impl: heuristic_v2f_plus_planner_terminal_force
Deprecated typos: terminal_form, terminal_force (transcription/typo — keep only as aliases)
```

Note: live V001 identity string in repo is `heuristic_v2f_plus_planner_terminal_force` — preserve in historical artefacts; prevent deprecated aliases from spreading.

Stages: inventory → directory ownership → candidate registry/aliases → source naming enforcement → safe deletion → builder path enforcement → verification gates.

Do not delete unknowns; do not rewrite public-version history; no portal mutation; no eval reruns as part of rename.

---

## 10. Elo interpretation (for prompt context, not a task)

Elo slightly above 1000 ≈ beating some weaker opponents; not dominance. Fresh versions provisional. Current climb almost certainly from live QUALIFIED V001, not NOT_QUALIFIED Tactical V2.

---

## 11. Artefacts the generated Cursor prompt must create/update

Immediate recovery:

```text
experiments/aborted_runs/phase9fu_paired_eval_20260805/*   # if stop
experiments/reports/phase9fu_paired_eval_abort.md
experiments/manifests/phase9fu_paired_eval_abort.json
experiments/reports/phase9fu_evaluator_recovery.md
experiments/reports/phase9fu_tactical_v2_portal_result.md
experiments/manifests/phase9fu_tactical_v2_portal_result.json
submission/roles/rejected.json
scripts/phase9fu_paired_eval.py (+ tests)
experiments/manifests/phase9fu_eval_QS-P9FU-HYBRID-BC-V1.partial.json  # during resume
experiments/manifests/phase9fu_v001_vs_challengers.json                 # on completion
submission/UPLOAD_THIS.md / roles/recommended.json                     # only if pass
```

Later Phase 9G (not now):

```text
experiments/reports/phase9g_preovernight_training.md
experiments/manifests/phase9g_preovernight_training_gate.json
experiments/manifests/phase9g_preovernight_checkpoint_comparison.json
experiments/reports/phase9g_preovernight_package_qualification.md
+ passive/throughput/control artefacts listed in §8
```

---

## 12. Suggested output structure for the other AI

Generate **three** Cursor prompts (do not merge into one mega-run if context limits bite):

### Prompt A — NOW (Agent mode): Stage 3 recovery

Title: `PHASE 9FU STAGE 3 RECOVERY — STOP/RECORD/REPAIR/HYBRID`

Include: decision rule §3, preserve/kill, abort+portal artefacts, evaluator repair checklist, Hybrid-first resumable eval, recommendation rules, hard stops (no PPO/overnight/portal/upload).

### Prompt B — AFTER 9FU terminal + merges: repo normalisation

Title: `PHASE 9FU POST-RUN REPOSITORY NORMALISATION`

### Prompt C — AFTER 9FU + INPUT_READY: Phase 9G

Title: `PHASE 9G CONTROL + TIER-1 + C1 PRE-OVERNIGHT + PPO`

Must include Track C1 mandatory daytime gate; overnight only after C1; passive telemetry allowed; action-changing control evidence-gated.

Also emit a short **operator checklist** the human can tick:

```text
[ ] Re-resolve paired_eval PIDs; stop only if decision rule says stop
[ ] Preserve _paired_eval.log (contains Tactical summary)
[ ] Record portal rejection for 22a6d05b...
[ ] Repair evaluator observability/resume
[ ] Verify Hybrid package/CPU/parity gates before treating as uploadable
[ ] Run Hybrid-only resumable paired eval
[ ] Recommend V002 only if thresholds pass
[ ] Manual upload only from UPLOAD_THIS
[ ] Freeze V002
[ ] Do not start overnight / control / rename storm yet
```

---

## 13. Hard stops for every generated prompt

```text
Do not implement from this brief alone without user pasting an execution prompt.
Do not auto-upload.
Do not portal mutate/delete.
Do not re-upload V001 or Tactical V2 22a6d05b...
Do not mark incomplete opaque runs as competitive wins.
Do not silently edit phase9fu_evaluation_protocol.json v1.
Do not start PPO overnight, rental, Phase 10, or action-changing control in Stage 3 recovery.
Do not invent metrics; only record measured or user-provided portal evidence.
Do not expose active tactics in public artefacts beyond what packaging already requires.
```

---

## 14. Key file index (absolute-ish repo paths)

```text
plans/phase9fu_competitive_recovery.md
plans/phase9fu_super_prompt_source_brief.md          # this file
scripts/phase9fu_paired_eval.py
scripts/phase9fu_behavioural_gates.py
scripts/phase9fu_package_hybrid_bc.py
scripts/phase9fu_package_tactical_v2.py
experiments/manifests/phase9fu_evaluation_protocol.json
experiments/manifests/phase9fu_behavioural_gates.json
experiments/manifests/phase9fu_tactical_v2_package.json
experiments/manifests/phase9fu_throughput_lane.json
experiments/manifests/phase9fu_v001_reclassification.json
experiments/manifests/phase9fu_submission_upload_freeze_gate.json
experiments/manifests/_paired_eval.log
experiments/manifests/_paired_eval.err
experiments/reports/phase9fu_candidate_b_builder.md
experiments/reports/phase9fu_behavioural_gates.md
experiments/reports/phase9fu_v001_strategic_failure.md
experiments/reports/phase9fu_throughput.md
submission/UPLOAD_THIS.md
submission/roles/{live,recommended,rejected,research_candidates}.json
submission/packages/QS-PUBLIC-V001/e1237f77dee46993/
submission/packages/QS-P9FU-HYBRID-BC-V1/5152a08eb774cf0e/
submission/packages/QS-P9FU-HEURISTIC-TACTICAL-V2/22a6d05b7160d86f/
submission/public_versions/QS-PUBLIC-V001/upload_freeze.json
submission/manifests/package_registry.json
experiments/phase9f_cnn_ranker_v1/checkpoints/bc/model.json
```

---

## 15. Bottom line for the prompt author

```text
V001: live weak baseline — keep, do not re-upload
Tactical V2 exact build: portal NOT_QUALIFIED + local direct 0.5 — reject, do not re-upload
Hybrid BC: main V002 hope — packaged, behavioural PASS, eval incomplete/opaque
Paired evaluator: must gain progress + checkpoint + resume
UPLOAD_THIS: remain NO_CANDIDATE until Hybrid (or future eligible) proves thresholds
After V002 attempt: Tier-1 → daytime PPO package → only then overnight
Control engineering: evidence-gated; not now
```

End of brief.
