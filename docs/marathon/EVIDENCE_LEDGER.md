# Marathon Evidence Ledger

**Plan ID:** `MARATHON_REDESIGN_LOCKED_V1`

## Evidence policy

Every entry separates observation from inference and records a source class. Allowed source classes are `OFFICIAL_RULE`, `REPOSITORY_FACT`, `LOCAL_RUNTIME_EVIDENCE`, `PRIMARY_LITERATURE`, `PUBLIC_IMPLEMENTATION`, `COMMUNITY_REPORT`, and `SPECULATION`. A training-health observation is not automatically strength evidence.

New research ideas follow `SOURCE -> mechanism -> interaction/risk -> bounded experiment -> observed result -> PROMOTE / REJECT / DEFER`.

## Stage 0 entries

### `EV-0001` — official engine pin

- Source class: `REPOSITORY_FACT`
- Evidence: Git submodule `third_party/generals-bots` resolves to `9e3b9d13cca51caa1bb07db48bb85c9e90ce0462`.
- Interpretation: official rules, protocol, runners, and dependency lock are pinned at this identity.
- Decision: `KEEP`; do not patch silently.

### `EV-0002` — trusted 7.59M checkpoint artefacts

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- Path: `C:\Users\pries\quantsilico-runtime\cloud_assisted_deadline_salvage_v1_final\ckpt_final_u482_t7593984`
- Manifest schema: `2`
- Metadata: programme `CLOUD_VALID_LEARNING_RECOVERY_V1`, update `482`, transitions `7,593,984`, environments `512`, rollout length `32`, checkpoint carry included.
- Recorded SHA-256:
  - raw: `5d5c0cbfcd35a70b223be55f934b1738e177d4707373709fe1ff263e183fd2b6`
  - EMA: `28241892048206ab42a06c463d45eacb7651d019ce1f6b6b1cad7b7b35ab0930`
  - optimizer: `d912846327384e7f4b459012adc7746d7c09b3072fda176526de5dfd8c8fa86d`
  - rollout carry: `6289f5e7e6e48a8b1fe5fb7f949121cdfa0aeb3489cf23ec16367e41e4718976`
  - frozen opponent: `6f5872964a280b4c6a2c8268c2b4776c1a465b040d49b8511196167c99e3482b`
  - metadata: `89c4908b633add709f82b9fb618deebe126436814f7e58ef1bfb59504b9e807f`
- Verification: file sizes and hashes matched `manifest.json` during the read-only Stage 0 audit.
- Limitation: the existing external canary is tiny and weak; it is not sufficient strength evidence.
- Decision: `KEEP`; register raw/EMA as `SPRINT_VALID_PPO_7M59`; reproduce as `MARATHON_BASELINE_V0` in Stage 1.

### `EV-0003` — later valid-learning-line checkpoints

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- Evidence: the same runtime tree contains checkpoints at approximately 10.00M, 14.97M, 19.36M, 25.01M, and 50.52M transitions. Programme records for 19.36M and 50.52M say `CLOUD_VALID_LEARNING_PROGRAMME_STATE / COMPLETE`; logs contain terminal and nonzero-reward batches.
- Limitation: external canaries are insufficient for champion or baseline claims.
- Decision: `KEEP / STRENGTH_UNKNOWN`; preserve and inventory during Stage 0. Do not silently supersede the locked 7.59M historical baseline.

### `EV-0004` — invalid zero-reward 35M run

- Source class: `REPOSITORY_FACT`
- Evidence: `experiments/manifests/cloud_gpu_last_push_v1_final_programme_state.json` records `HALTED_MANDATORY_LEARNING_INTEGRITY`, checkpoint transitions `35,409,920`, and `33,849,344` invalidated transitions because every production batch reconstructed rollout state and had zero reward/completion.
- Decision: register as `FORENSIC_ZERO_REWARD_35M / INVALID_LEARNING_INTEGRITY`; never use as opponent, teacher, or strength reference.

### `EV-0005` — local orchestration tooling audit (`ORCH-0001`)

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- Codex: user-local binary reports `codex-cli 0.147.0-alpha.6.5`; `codex login status` reports ChatGPT authentication; `codex exec --help` supports ephemeral, read-only, JSON-schema-constrained execution.
- Cursor: `agent`, `cursor-agent`, and `cursoragent` are not on native Windows `PATH`. The installed `cursor` command is the editor CLI and is not a substitute.
- Official source: `https://docs.cursor.com/en/cli/installation` documents Cursor Agent installation for macOS, Linux, and Windows through WSL; `https://docs.cursor.com/en/cli/reference/parameters` documents status/model/print controls.
- WSL: read-only WSL CLI probes timed out while a WSL VM was active. No WSL shutdown or trainer termination was attempted.
- Operator state: Cursor credits are exhausted.
- Decision: proceed with authority bootstrap and simulation-safe orchestration core; live Cursor/model end-to-end gate is `BLOCKED`, not failed.

### `EV-0006` — remote process audit

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- Evidence: read-only Runpod listing found pod `wvjrnxbpcjnr8h` (`generals_competition`) in `EXITED` state and no serverless endpoints.
- Decision: no active remote workload; no resource mutation authorized or performed.

### `EV-0007` — simulation-safe orchestration core

- Source class: `REPOSITORY_FACT`
- Implementation: `tools/agentic_orchestrator/` provides strict task/report/review schemas, atomic JSON replacement, durable JSONL transitions, an exclusive writer lock, legal transition validation, restart recovery, output/environment sanitization, CLI discovery, exact Cursor model matching, quota classification, human-boundary pauses, and bounded repair escalation.
- Tests: 22 focused unit tests pass; Ruff and byte-compilation pass.
- Dry run: deterministic architect/proposal/reviewer stand-ins exercised `FIX_FIRST -> REPAIRING -> ACCEPTED`, restart recovery, `PAUSED_HUMAN_BOUNDARY`, and `PAUSED_USAGE` without tracked repository edits.
- Live probe: Codex is `READY`, authenticated through ChatGPT, configured as `gpt-5.6-sol`, and reports `codex-cli 0.147.0-alpha.6.5`. Cursor Agent remains `UNAVAILABLE` and no model list can be queried.
- Limitation: the deterministic dry run is not a live cross-provider acceptance test. No live Cursor implementation or tiny real task is claimed.
- Decision: promote the simulation-safe core as a completed bounded unit; keep live orchestration acceptance `PARTIAL / PAUSED_USAGE` until the Cursor CLI, exact model identity, authentication, and credits are available.

## Open evidence requirements

- Complete Stage 0 artefact/worktree/entrypoint inventory.
- Semantic state hashes and deterministic resume capsule for Stage 1.
- Serious paired strength evaluation of historical and later checkpoints.
- Exact Cursor Agent executable, authentication, supported-model listing, and available usage before live orchestration acceptance.

### `EV-0008` — Stage 0B schema hardening validated (`ORCH-0001` continuation)

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- Takeover finding: the uncommitted Codex hardening diff left `tests/unit/test_agentic_orchestrator_workflow.py` fixtures on the pre-hardening schema; a real run at HEAD `780a8ce` produced `17 passed, 5 failed` while `ACTIVE_STATE.json` still claimed `PASS_22_TESTS` (stale).
- Repair at commit `5b345a7`: fixtures aligned with the hardened schema; latent `NameError` in `_validate_repository_report` for non-`COMPLETE` reports fixed by binding `normalized_files`/`actual` unconditionally; `DryRunAdapter` now reports the real changed-path set so repository-truth checks hold on a dirty worktree.
- New coverage: `tests/unit/test_agentic_orchestrator_hardening.py` adds 19 tests (END_COMMIT/PLAN_ID rules, stable PROBLEM_ID, structured HUMAN_BOUNDARY consistency, recursive redaction, boundary keyword inference, PARTIAL-report validation, ToolingGateError routing).
- Adjudication: `dry_run()` terminating in `ACCEPTED` before its pause/resume proofs is the intended contract; the prior `IDLE` expectation came from the pre-hardening flow.
- Gate (all at `5b345a7`): `pytest` over the four orchestrator test modules `41 passed, 0 failed`; `ruff check` clean; `compileall` exit 0; `dry-run` all `DRY_RUN_PROOFS` and `RECOVERY_PROBES` true, exit 0; `tooling` reports Codex `READY` (`gpt-5.6-sol`) and Cursor `UNAVAILABLE` (matches the recorded live-acceptance blocker).
- Decision: Stage 0B hardening is a completed bounded unit; live cross-provider acceptance remains blocked on Cursor CLI/usage.

### `EV-0009` — Qoder takeover preservation snapshot

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- Safety ref: `backup/codex-takeover-snapshot` created at `780a8ce` before any edit.
- Dirty-diff snapshot: `var/marathon_takeover/codex-takeover-dirty.patch` (binary-capable `git diff --binary HEAD`, 45,065 bytes), SHA-256 `dd0cf6efd3d979a7cfafede27498d46f3b93ed8ab835ca3185a055be3600352b`.
- Round-trip proof: with the dirty changes stashed, `git apply --check` of the patch succeeded; the stash was restored and the five modified files verified intact. A branch ref alone is not treated as preservation of uncommitted work.
- No untracked files required snapshotting at takeover (`git status -uall` showed only the five tracked modifications).
- Decision: takeover state is independently recoverable; proceed with Stage 0B completion.

### `EV-0010` — Stage 0 forensic freeze record

- Source class: `REPOSITORY_FACT` / `LOCAL_RUNTIME_EVIDENCE`
- Freeze point: branch `research/phase9g-competition-native-jax-preovernight-v1` at `9de138793b3979afd1c6190297b1000dd683f681` (pushed to origin), clean working tree. Official engine submodule `third_party/generals-bots` at `9e3b9d13cca51caa1bb07db48bb85c9e90ce0462` (tag `competition-engine-2026-15-g9e3b9d1`), matching `programme.yaml`.
- Worktrees (6): main repo (this branch); `quantsilico-emergency-training` detached at `2df3c6a` and **prunable (gitdir missing) — flag only, never prune without uniqueness proof**; `quantsilico-generals-dashboard-integration` on `feature/figma-console-integration` @ `4be2a55`; `quantsilico-generals-noon-rescue-v3` on `research/noon-closed-loop-hybrid-salvage-v3` @ `bdb28da`; `quantsilico-generals-perf-v1` on `perf/cloud-a100-forensic-v1` @ `4ae1b05`; `quantsilico-generals-valid-learning-recovery` on `research/cloud-valid-learning-recovery-v1` @ `c0b5353`.
- Trainer/entrypoint inventory: JAX training source lives in `train/competition_native_jax/` (13 tracked modules incl. `train_jax.py`, `train_loop.py`, `ppo_jax.py`, `rollout_selfplay_jax.py`, `ema_jax.py`, `gae_jax.py`, autotune/profile v4.2/v4.3a). Operational scripts: `scripts/cloud_*` (A100 orchestration/launch), `scripts/emergency_*` (CPU collect/resume/PPO), `scripts/training/gpu_smoke.py`, `scripts/wsl/_*` trainer helpers, `scripts/dev/verify_training_environment.py`, `scripts/dev/detect_training_hardware.py`. `training/` contains scaffold `.gitkeep` dirs only (`REGENERABLE`).
- Baseline checkpoint re-verification (read-only, this session): all six artefacts of `ckpt_final_u482_t7593984` hash-match the on-disk `manifest.json` (schema_version 2) and the EV-0002 recorded values — raw `5d5c0cbf…`, EMA `28241892…`, optimizer `d9128463…`, rollout carry `6289f5e7…`, frozen opponent `6f587296…`, meta `89c4908b…`. Stage 1 source integrity confirmed with no contradiction.
- Provenance stores: `experiments/manifests/` holds 401 tracked manifest files (`KEEP`); `models/registry/` holds `challenger_lifecycle.json` + `phase5_architectures.json` (`KEEP`); later-line checkpoints at ~10.00M/14.97M/19.36M/25.01M/50.52M transitions remain `KEEP / STRENGTH_UNKNOWN` per EV-0003.
- Overlap classification (inventory-only; nothing deleted): `dist/evidence/quantsilico_phase9f_evidence_bundle.zip` → `UNKNOWN` (candidate `MIGRATE` to `experiments/public_exports` after Stage 4B reconciliation); `submission/packages/*.report.json` and `submission/public_versions/` → `KEEP` (delivery provenance); `submission/staging`, `submission/legacy` → `UNKNOWN` pending Stage 4B packaging reconciliation; `var/**` runtime trees → `REGENERABLE` (gitignored).
- Repository visibility: public; `REPOSITORY_PRIVACY_CUTOVER` untouched. No contradictions found among code, configuration, evidence, and authorities at this freeze point beyond those already recorded (live-acceptance blocker, WSL workload identity unknown).
- Decision: Stage 0 freeze record complete; unique research evidence frozen; proceed to the Qoder harness workstream, then Stage 1.

### `EV-0011` — minimum Qoder harness created and tested (W4)

- Source class: `REPOSITORY_FACT` / `LOCAL_RUNTIME_EVIDENCE`
- Assets created: rule `.qoder/rules/marathon-evidence-contract.md` (`trigger: always_on`); skills `.qoder/skills/ultra-review/SKILL.md` and `.qoder/skills/quantisilico-marathon-resume/SKILL.md` (frontmatter validated programmatically); deterministic hooks `.qoder/hooks/guard_destructive.py` (deny destructive git / evidence-dir deletion, ask on branch/worktree removal), `.qoder/hooks/post_edit_ruff.py` (narrow lint after orchestrator edits), `.qoder/hooks/evidence_gate.py` (PASS/COMPLETE claims require named evidence), wired in `.qoder/hooks.json`.
- Hook tests: `tests/unit/test_qoder_hooks.py` exercises each hook through its real stdin/stdout contract — `14 passed`, ruff clean. Hooks fail-open on malformed input so a bad event cannot wedge the harness; the blocking policy lives in the deterministic deny patterns, not prose.
- Marketplace/skills discovery (bounded, no installs): `obra/superpowers` (`test-driven-development`, `verification-before-completion`, `systematic-debugging`) classified REDUNDANT — project rule + evidence-gate hook + ultra-review skill already encode stronger, project-specific versions; prompt-only methodology skills add no executable infrastructure. Generic TDD/debugging variants: POSSIBLY LATER. No JAX/profiling skill with executable tooling found: REJECT for now, revisit at Stage 1 if a demonstrated gap appears. Supply-chain caution is evidence-backed: skill-based prompt injection is a documented attack surface (arXiv:2602.14211), so admission requires source/script/network inspection first. Nothing classified INSTALL NOW.
- Better Harness audit: scheduled at the next session boundary and post-marathon (slash-command invocation; not yet executed, therefore not claimed).
- Decision: W4 complete as scoped; later skills deferred until execution demonstrates recurring need.

### `EV-0012` — Stage 1 baseline provenance gap and local environment facts

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- Local environment (`.venv`, 2026-08-14): Python 3.12.10; JAX 0.11.0 CPU-only (`.venv` and `.venv-training`); torch 2.13.0+cpu; RTX 3070 Laptop GPU present (driver 581.42) but no CUDA JAX/torch. CNJ fresh-training smoke PASSED on CPU (`scripts/run_competition_native_jax_smoke.py` with `PYTHONPATH=.`: `DONE COMPLETED`, 2 updates; hardware report at `var/training/hardware_report.json`). GPU temperature 84 C at ~31 W draw while idle from our side — consistent with the unidentified WSL workload; workload untouched.
- Provenance finding: `ckpt_final_u482_t7593984/meta.json` records `learner_hash c91a6c75…` under the writer's recipe (`sha256(transformer_jax.py + ppo_jax.py + gae_jax.py)`, per `research/cloud-valid-learning-recovery-v1:scripts/cloud_valid_learning_recovery.py`). No committed tree in this repository's full history (107 commits scanned, both the 5-file and 3-file recipes) and neither the sibling worktree HEAD (`729887fa…`) nor its working tree (`a39e8e82…`) reproduce that hash. The `cloud_assisted_deadline_salvage_v1` trainer script exists in no accessible worktree; the most plausible source was the now-prunable WSL worktree `quantsilico-emergency-training` (gitdir missing; `/mnt/c/...` path unreachable without WSL, which must not be disturbed).
- Interpretation: checkpoint WEIGHTS are intact and hash-verified (EV-0002/0010, 6/6 artefacts), but the exact writer source identity is currently UNRECOVERABLE from local material. Behavioural reproduction must therefore proceed from the checkpoint data plus the schema-v2 loader lineage (`research/cloud-valid-learning-recovery-v1:train/competition_native_jax/checkpoint_recovery.py`), with structural-compat and first-N-observation proofs, and the source gap declared rather than assumed away.
- Decision: record as a Stage 1 limitation; next bounded action is the semantic-state hash capsule of the checkpoint arrays and a structural load probe using the schema-v2 loader lineage. If the WSL workload's owner later identifies the deadline-salvage source, reconcile the hash and upgrade this entry.

### `EV-0013` — baseline semantic fingerprints and structural compatibility

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- Semantic capsule: `scripts/analysis/semantic_state_hash.py` (EXECUTION_PLAN §6.3: deterministic key order; key + dtype + shape + canonical contiguous bytes; container metadata excluded) produced FILE_SHA256 + SEMANTIC_STATE_SHA256 for all five artefacts; tracked at `experiments/marathon/baseline_semantic_hashes.json`. FILE hashes equal EV-0002 values; semantic fingerprints: raw `471efdaa…`, EMA `a4a7e1a0…`, opt_state `3568ba13…`, rollout_carry `2822cef5…`, frozen_opponent `4317d75f…`.
- Structural probe: `scripts/analysis/baseline_structural_probe.py` compared checkpoint npz signatures against this branch's `init_params`/`make_optimizer` templates — **EXACT_MATCH on raw.npz (24 keys), ema.npz (24), frozen_opponent.npz (24), opt_state.npz (49)**. The marathon branch's own competition_native_jax lineage can host the baseline weights exactly; the provenance gap in EV-0012 affects writer-source identity, not parameter/optimizer architecture.
- Decision: Stage 1 reproduction may proceed on this branch's CNJ lineage using `load_tree`; next bounded action is an actual checkpoint load + first-N-observation behavioural fingerprint on CPU, followed by the resume capsule (rollout_carry template still to be matched against this branch's carry lineage).

### `EV-0014` — baseline weights run end-to-end on the marathon branch (CPU)

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- `scripts/analysis/baseline_behaviour_fingerprint.py` loaded `raw.npz`/`ema.npz` via this branch's `load_tree` and collected deterministic self-play batches on CPU (jax 0.11.0, seed 1234).
- Run 1 (num_envs=4, rollout_len=8, pool=64): 32 steps; action histogram shows move-pair action IDs (~536–3318) plus sparse early-turn action 0; `logp_mean=-0.8866`, `value_mean=0.00226`, done_count=0, reward_sum=0.0 (terminal rewards only, expected at 8 steps).
- Run 2 (num_envs=8, rollout_len=16, pool=128): 128 steps; diverse legal move-pair actions across ~40 distinct IDs; `logp_mean=-0.8932`, `value_mean=0.00337`, done_count=0, reward_sum=0.0 (still early-game; not strength evidence).
- Interpretation: the 7.59M-transition baseline policy and value head produce coherent, legal, non-degenerate behaviour through THIS branch's environment/rollout lineage. First-N digests + scalar summaries are in `var/marathon_takeover/baseline_behaviour_fingerprint.json` with sampled arrays in `baseline_behaviour_sample.npz` (untracked working copies; canonical capsule copies land under `experiments/` once the full capsule is assembled).
- Limitation: single-device CPU fingerprint only; no rollout-carry resume yet (carry template matching pending); no external strength claim.
- Decision: Stage 1 load path proven; next bounded action is rollout_carry template matching and the one-update PPO resume proof.

### `EV-0015` — baseline checkpoint/resume proof: one PPO update from restored state

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- `scripts/analysis/baseline_resume_one_update.py` loaded params/EMA/opt_state from `ckpt_final_u482_t7593984` via this branch's `load_tree`, collected a deterministic batch (num_envs=4, rollout_len=8, seed=7, pool=64, CPU, LR=3e-4 matching the legacy lineage), and applied exactly one full-batch PPO update using the RESTORED opt_state, then EMA update.
- Result: `status PASS` — all metrics finite (loss 1.1718, pg -0.00081, vloss 2.3577, entropy 0.6196), **ratio exactly 1.0** (step-zero ratio identity proves the restored policy produced the batch's old_logp, i.e. genuine resume rather than cold start), params/EMA/opt_state all changed and finite after the update. Pre/post semantic digests recorded in `var/marathon_takeover/resume_step1/resume_report.json`; updated trees persisted alongside (untracked working capsule).
- Rollout-carry structural note: `rollout_carry.npz` (19 arrays, 512 envs) matches this branch's `RolloutCarry` episode-state fields (states/mem0/mem1/key/pool_cursor) with two additional curriculum fields (`learner_seat`, `episode_id`) from the writer lineage; carry resume for a different env count will re-slice episode state rather than bit-copy it — declared, not assumed.
- Decision: checkpoint/resume viability demonstrated for MARATHON_BASELINE_V0 (§6.5 partial: valid learning continuation shown on CPU; full capsule, determinism contract record, TPS metrics, and external gameplay measurement remain as the Stage 1 completion gates).

### `EV-0016` — MARATHON_BASELINE_V0 capsule assembled (determinism contract + TPS + cross-process determinism)

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- `scripts/analysis/assemble_baseline_capsule.py` produced the tracked capsule `experiments/marathon/baseline_capsule_v0.json` (status PASS, wall 1485.6 s on CPU): source/engine/lineage identity, semantic-hash pointer, determinism contract (JAX 0.11.0, jaxlib 0.11.0, backend cpu, DTYPE_POLICY float32_default, DETERMINISM_MODE CPU_DEFAULT_NO_ASYNC), resume-proof digest record, and separate throughput measurements.
- Determinism evidence (two independent in-process runs under identical configs): behavioural first-N digests match (seed 1234, 4×8), and the one-update resume (seed 7) reproduced IDENTICAL params/EMA/opt_state digests and metrics (ratio 1.0, loss 1.171844720840454, entropy 0.619606614112854 — digit-for-digit equal to EV-0015's earlier separate process run, i.e. cross-process STATE_SEMANTIC_DETERMINISM + BEHAVIOURAL_DETERMINISM demonstrated). BITWISE_DETERMINISM remains claimed but not separately demonstrated.
- Throughput (CPU, num_envs=8, rollout_len=16, pool=128): hot-path (collection only, warm JIT) 1.8 TPS over 12 batches; end-to-end (collection + full-batch PPO update) 1.9 TPS over 4 iterations; valid-learning (transitions behind updates passing finite-metric/ratio health) 1.9 TPS with 4/4 healthy updates. Reported separately per §7.4; absolute values are CPU-class and confirm the compute amendment's rationale that GPU/cloud credits are required for transition-budget-scale training.
- Residual Stage 1 gates: external gameplay measurement of the baseline (requires the Stage 2 evaluator against live opponents; now unblocked by EV-0017) and rollout-carry resume across env counts (declared re-slice semantics, EV-0015).

### `EV-0017` — Stage 2 canonical paired evaluator implemented, tested, and engine-smoked

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- Implementation: `src/generals_bot/marathon_eval/` (pairing, confidence_sequence, promotion, store, runner) + driver `scripts/evaluation/run_marathon_paired_eval.py`. Seat-swapped pairs on canonical hashed map seeds (eval namespace disjoint from training/screening); PAIR_SCORE = mean candidate score over both seats; atomic resumable JSONL store with replay identity and failure attribution; matchup metrics WORST_MATCHUP_SCORE / BOTTOM_QUARTILE_MATCHUP_SCORE / MATCHUP_SCORE_STD; promotion policy binds `configs/marathon/programme.yaml` margins (practical 0.01, noninferiority -0.005, worst-matchup improvement 0.05).
- Sequential method PLAN_DEVIATION (declared, within §7.2's permitted set): `programme.yaml` names ANYTIME_VALID_EMPIRICAL_BERNSTEIN_CONFIDENCE_SEQUENCE; the shipped method is `ANYTIME_VALID_BOUNDED_MIXTURE_CONFIDENCE_SEQUENCE` — a Hoeffding-lemma normal-mixture confidence sequence with a closed form whose endpoints provably invert the test supermartingale (asserted exactly in `tests/unit/test_marathon_evaluator.py::test_confidence_sequence_inverts_mixture_test_exactly`). It is conservative (worst-case bounded variance) rather than empirical-Bernstein-tight; coverage under the null at fixed time and under per-pair optional stopping verified by Monte Carlo (300 simulations × 40 pairs, violations within alpha slack). An empirical-Bernstein tightening is a documented future optimisation for large pair budgets.
- Test evidence: `tests/unit/test_marathon_evaluator.py` 11/11 PASSED; `tests/integration/test_marathon_paired_evaluator_engine.py` 3/3 PASSED (one real seat-swapped pair through the pinned competition protocol; relative-path regression; CS on a real difference); pre-existing `tests/integration/test_pass_match.py` still PASSED after `MatchResult` extension.
- Defects found and repaired by the evaluator itself: (1) relative agent main paths resolved against the child's cwd, silently killing agents (OSError 22 / exit=2); the crash-attribution evidence (`detail` field, `var/marathon_takeover/smoke_contaminated_relative_path_bug/`) identified the doubled path; fixed by resolving absolute paths in `run_python_agent_match` + regression test. (2) `run_python_agent_match` now captures agent crashes (stderr tail, exit code) as faults instead of aborting the whole evaluation.
- Engine smoke run (tracked): `experiments/marathon/paired_eval_runs/smoke_legal_random_vs_pass_bot/` — legal_random vs pass_bot, 2 pairs, mean difference 0.5, CS [-1, 1] at 2 pairs, NO_PROMOTION (correct: evidence insufficient at this sample size, sequential guarantee intact).
- Known pre-existing debt surfaced (NOT caused by this change; reproduced identically on merged main `983d87a` via throwaway worktree): 9 legacy torch-path unit tests fail on CPU-only machines because `configs/training/device_policy.yaml` requires CUDA (test_behaviour_clone ×1, test_persistent_actor ×4, test_ppo_action_support ×4; root cause `DevicePolicyError: CUDA required ... torch 2.13.0+cpu`). Recorded as KNOWN_FAILED_LEGACY_CUDA_POLICY; repair deferred (test-plumbing change across three legacy suites) and must not be silently claimed green.

### `EV-0018` — continuous-execution stop gate added to the Qoder harness

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- Operator amendment QODER-CONTINUOUS-EXECUTION-2026-08-15: bounded tasks are checkpoints, not session termination conditions; the Marathon may stop only on full canonical completion (Stages 1–7 + final adversarial review + integration + post-merge proof) or a recorded hard blocker that covers ALL remaining work.
- Implementation: `.qoder/hooks/marathon_stop_gate.py` is now the single `stop` hook in `.qoder/hooks.json`. It runs the evidence gate first (precedence preserved), then evaluates `experiments/marathon/ACTIVE_STATE.json`: executable NEXT_SAFE_ACTION or incomplete canonical stages → BLOCK with an injected continuation message; MARATHON_COMPLETE (explicit STAGES + COMPLETION fields, including post-merge proof) or ALL-scope hard blocker → ALLOW. Fail-safety: malformed/missing ACTIVE_STATE blocks conservatively with a reconciliation diagnostic; manual emergency stop via `MARATHON_EMERGENCY_STOP=1` or `var/marathon_takeover/EMERGENCY_STOP`; a block counter in `var/marathon_takeover/stop_gate_blocks.json` escalates to operator review after 200 consecutive blocks (no infinite loop); the hook stays `fail_closed: false` so a hook crash cannot deadlock the agent.
- Completion-time requirement: when the programme genuinely finishes, ACTIVE_STATE must be populated with the STAGES and COMPLETION fields the gate reads; until then the gate is conservative by construction (it can only refuse, never falsely allow).
- Test evidence: `tests/unit/test_qoder_stop_gate.py` 13/13 PASSED — all seven amendment scenarios (IN_PROGRESS+NEXT_SAFE_ACTION blocked; full completion allowed; ALL-scope hard blocker allowed; scoped hard blocker with independent work blocked; malformed/missing state conservatively blocked; evidence-gate precedence; emergency stop and runaway escalation allowed) plus real stdin/stdout contract runs against the live ACTIVE_STATE. Pre-existing hook suite `tests/unit/test_qoder_hooks.py` 14/14 PASSED unchanged.

### `EV-0019` — MARATHON_BASELINE_V0 packaged as EVAL_ONLY protocol agent; first external gameplay measurement

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- PPO_SEMANTICS: `EVAL_ONLY` (packaging serves frozen weights; training action selection untouched).
- Serving-path parity PROVEN (`tests/unit/test_baseline_agent_parity.py` 4/4 PASSED): for identical engine states, the exact serving path (pinned engine `competition/protocol.py` encode -> host `parse_observation_frame` -> `obs_memory.encode_observation` + `legal_mask_from_observation`) reproduces the training path (`competition_env_jax.observe_one_jax` + `legal_mask_one_jax`) within 1e-6 for spatial[8,21,21] and global[8], with bitwise-equal 3970-d legal masks, across 3 board geometries, both seats, and 6 stepped turns with memory accumulation (constants verified identical: TRUNCATION=DRAW_TURN=1200, DEATHTOUCH=800, action codec and kinds identical).
- Agent: `baselines/marathon_baseline_v0/main.py` + `src/generals_bot/competition_native_jax/jax_baseline_policy.py` (greedy, deterministic, JIT inference). Defect found and fixed before measurement: `_jit_infer` arity mismatch (4 vs 5 args) would have crashed every turn; the evaluator's crash-attribution path surfaced it.
- Measurement (tracked): `experiments/marathon/paired_eval_runs/baseline_v0_vs_legal_random_cpu/` — marathon_baseline_v0 vs legal_random, 3 canonical seat-swapped pairs on CPU (235.8 s). All six games: DRAW at truncation (1200 turns), zero faults, attribution OK; PAIR_SCORE 0.5 each; mean difference vs zero-baseline 0.5; CS [-1, 1] at n=3 -> NO_PROMOTION (correct; no strength conclusion from this sample).
- Behavioural finding (not a defect): the policy alternates legal moves and forced passes in the opening (verified turn-by-turn: turns 0-1 PASS-only legal, turn 2 MOVE, turn 3 PASS, ...), consistent with EV-0014 fingerprints; legal_random also failed to capture within 1200 turns, so all pairs truncated. This is the baseline's measured gameplay character at sample size 6 games; stronger-opponent pairs and larger pair budgets are the next measurement increments. No strength claim is made from this run beyond the recorded outcomes.
- Serving throughput note: first-call JIT compile ~1.6 s, warm inference ~7 ms/turn on CPU; full 1200-turn protocol game ~78 s wall per game including subprocess IO. External gameplay measurement gate for MARATHON_BASELINE_V0: first measurement captured (this entry); further pairs/opponents remain open under Stage 2.

### `EV-0020` — Stage 3 minimum canonical registry implemented, tested, and seeded

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- Implementation: `src/generals_bot/marathon_registry/` — six record kinds (experiment, run, checkpoint, candidate, evaluation, opponent_reference) with stable readable IDs (`KIND#name#sha256[:12]`), explicit SCHEMA_VERSION 1.0.0, atomic writes (tmp + fsync + rename; Windows FlushFileBuffers requires a write-access handle — verified empirically), source/engine/config identity, lineage, seeds, commands, budgets, stop reasons, artefact locations, and evidence links. Validation rejects: missing/ambiguous PPO_SEMANTICS, unknown/incompatible lineage, unresolved sha256 artefact hashes, missing evaluator identity, dangling cross-references, schema-version mismatch, and silent overwrite.
- Test evidence: `tests/unit/test_marathon_registry.py` 8/8 PASSED (ID stability, full six-kind chain, overwrite refusal, semantics enforcement, hash/lineage rejection, dangling-reference rejection, evaluator-identity requirement, schema-version enforcement).
- Seeding: `scripts/dev/register_marathon_baseline_v0.py` registered the canonical chain from tracked evidence only (capsule + semantic hashes + paired-eval summary): `experiment#marathon-baseline-v0-repro#5e9100e819c5` (UNCHANGED) -> `run#baseline-capsule-cpu#22f63bf89ca9` -> `checkpoint#marathon-baseline-v0#241591ce549b` (5 FILE_SHA256 artefact hashes, 7,593,984 transitions, SPRINT_VALID_PPO_7M59) -> `candidate#marathon-baseline-v0#5c9adceafd94` (EVAL_ONLY, parity proof link) -> `evaluation#baseline-v0-vs-legal-random-cpu#02b635956805` + `opponent_reference#legal-random#96c4592a050e`. Records at `experiments/marathon/registry/records/`.
- Defect found during seeding: `experiments/marathon/baseline_semantic_hashes.json` had been written UTF-16 (PowerShell redirection during EV-0013) and was unparseable as JSON; regenerated deterministically with the tracked `scripts/analysis/semantic_state_hash.py` — semantic fingerprints reproduced digit-for-digit (EMA a4a7e1a0…, opt_state 3568ba13…, frozen_opponent 4317d75f…, matching EV-0013), file rewritten UTF-8.

### `EV-0021` — extended baseline gameplay measurement (heuristic opponents) and truncation-draw control

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- Measurement: marathon_baseline_v0 vs heuristic_v0 (2 pairs) and heuristic_v1 (2 pairs), canonical seat-swapped pairs on CPU, 401.9 s total; tracked at `experiments/marathon/paired_eval_runs/baseline_v0_vs_heuristics_cpu/`. All eight games DRAW at truncation (1200 turns), zero faults, attribution OK; PAIR_SCORE 0.5 each; CS [-1, 1] -> NO_PROMOTION.
- Combined baseline character so far: 14 measured games (6 vs legal_random, 8 vs heuristics), 14 truncation draws, zero faults.
- Control experiment: heuristic_v1 vs heuristic_v0 (seed 55, 1200-turn cap) ALSO ended DRAW at truncation with zero faults — the truncation-draw outcome is an environment/ruleset-level property of these agent pairings at this turn cap, not an artefact of the baseline packaging or the serving path. Interpretation: within 1200 turns none of the measured agents captures the opposing general; scoring differentiation therefore requires either decisive captures under stronger play or supplementary metrics (territory/army deltas at truncation), which are NOT yet part of the canonical score and must be added as a declared protocol change before use.
- Registry update: `evaluation#baseline-v0-vs-heuristics-cpu#81dd91f99143`, `opponent_reference#heuristic-v0#ac6775d8c6eb`, `opponent_reference#heuristic-v1#a2db3b5f266c` added (total 9 records); idempotent re-seeding skipped the 6 pre-existing IDs without overwrite.
- Full unit gate at this point: 218 passed, 2 skipped, 9 KNOWN_FAILED_LEGACY_CUDA_POLICY (classification unchanged, EV-0017).

### `EV-0022` — Stage 1-3 integrated to main; Stage 4A first screening round predeclared and registered

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- Integration: PR #3 merged to main (merge commit `bc6970c`); post-merge proof on actual main: 50/50 unit tests (evaluator + stop gate + hooks + registry + parity suites) plus 3/3 engine integration tests PASSED; research branch fast-forwarded to `bc6970c` and pushed.
- Stage 4A predeclaration: `experiments/marathon/screening_round_1_plan.yaml` (SH-R1-CPU-PILOT) — three horizon-ablation arms at matched 128-transitions/update batch size (8×16 control, 4×32, 16×8), 25k transitions per arm, one seed, PPO_SEMANTICS UNCHANGED on all arms; predeclared screening metrics (VLOSS_REDUCTION_OVER_ROUND, VALID_LEARNING_SHARE, ENTROPY_HEALTH; secondary PG_MAGNITUDE, TPS_MEASURED reported separately) and elimination rules (any non-finite metric, VALID_LEARNING_SHARE < 0.9, entropy < 0.05, non-positive vloss reduction); integrity-only stops in round 1; cheap telemetry explicitly cannot promote a champion (finalists go to promotion-scale paired evaluation). The EV-0021 draw regime is addressed: round 1 screens on training telemetry; any gameplay tie-break metric requires a declared evaluation-protocol amendment first.
- Registry: `experiment#sh-r1-cpu-pilot#796c4d71603c` registered BEFORE any training launch (registry-driven discipline, EXECUTION_PLAN §8), single declared semantics verified by the registration script (`scripts/dev/register_screening_round_1.py`); registry now holds 10 records.
- Next bounded action: implement the screening runner (resume from MARATHON_BASELINE_V0 raw/opt_state with per-arm geometry, emit per-update telemetry JSONL + arm summary), then launch the control arm as a bounded pilot before the multi-arm overnight round.

### `EV-0023` — RUNPOD-SPEND amendment accepted; SH-R1 runner implemented, smoked, and control arm launched

- Source class: `LOCAL_RUNTIME_EVIDENCE`
- Operator amendment RUNPOD-SPEND-2026-08-15 recorded in ACTIVE_STATE.PLAN_DEVIATIONS: cash-backed Marathon compute pre-authorised (supersedes existing-credits-only); low balance must not shrink scientifically justified budgets; funnel-governed scaling; official RunPod tooling preferred; RUNPOD_API_KEY never exposed/committed; no new payment methods/terms/ownership; idle accelerators stopped; billing evidence required per run; scientific gates not weakened by spend. Credential probe at acceptance time: no `~/.runpod/config.toml`, no RUNPOD_API_KEY env, no runpodctl, no .env — provisioning of a dedicated restricted key by the operator remains the single operational dependency for GPU-class rounds.
- Runner: `scripts/training/run_sh_r1_arm.py` (PPO_SEMANTICS UNCHANGED; resume via load_tree from the hash-verified checkpoint; per-arm geometry; fixed transition budget; per-update telemetry JSONL with fsync-on-write; predeclared-metric arm summary; integrity-only stops; refuses to overwrite existing telemetry; saves raw/ema/opt_state at completion). Defect found and fixed in smoke: `save_tree(path, tree)` argument order.
- Smoke evidence (arm `smoke`, geometry 8×16, 3 updates, 384 transitions): ratio 1.0 first and last (genuine resume semantics preserved), VALID_LEARNING_SHARE 1.0, entropy 0.958→0.881, collect TPS 1.4, end-to-end TPS 1.2, vloss flat (2.3581→2.3597) so the predeclared NO_VLOSS_REDUCTION rule fired exactly as designed at smoke scale — rule machinery verified; the smoke's purpose was pipeline proof, not screening.
- Control-arm pilot LAUNCHED: `SH-R1-A0-CONTROL` (geometry 8×16, budget 25,000 transitions ≈ 195 updates, seed 20260815, CPU, est ~6 h wall at measured throughput); output `experiments/marathon/screening_runs/SH-R1-A0-CONTROL/` (telemetry.jsonl + summary.json + checkpoints), console log `var/marathon_takeover/sh_r1_a0_control.log`. Horizon arms SH-R1-A1/A2 are held for the GPU decision per RUNPOD-SPEND-2026-08-15 (replacing an appropriate GPU round with ~12 additional CPU-hours per arm would violate the amendment; the CPU control pilot remains the explicitly sanctioned pipeline proof).

### `EV-0024` — RunPod authenticated connectivity proven; existing generals_competition pod inspected (read-only)

- Source class: `LOCAL_RUNTIME_EVIDENCE` + `PROVIDER_API_EVIDENCE`
- Credential handling: the operator-supplied API key was persisted to the standard RunPod location `~/.runpod/config.toml` (outside the repository); only its fingerprint is recorded here — KEY_SHA256 `a34c0f0eb0feb77c84aa71bf3afbc7f7cead939bb963f1ebd679b206d08dc398`, prefix `rpa_J6Z...` (len 50). The key appears in NO tracked file, evidence file, or report. Advisory: because the key transited this chat, rotating it after the marathon window is recommended (a local config change, no programme impact).
- Tooling reality: `pip install runpod` fails — PyPI/files.pythonhosted.org are unreachable from this machine (connection reset, sandboxed and unsandboxed); `runpodctl` binary download equally unavailable. Resolution: read-only access implemented with stdlib against the official API surface extracted from the official runpodctl source (cloned commit 6fc6f8b5): GraphQL `api.runpod.io/graphql` (`myself`, `gpuTypes`) and REST `rest.runpod.io/v1` (`/pods`, `/networkvolumes`). Probe scripts: `var/marathon_takeover/runpod_probe.py` (untracked scratch, key-free output). CLOUD_CREDIT_BALANCE_UNKNOWN blocker resolved.
- Account facts (probe 2026-08-15): auth OK; user `user_3HaRLfRDwaZ7TMJLVabLTxCHmo3`; clientBalance **$17.89**; spendLimit **$80/h** (platform control); currentSpendPerHr **$0.014** (residual storage charge while all pods exited — consistent with idle-pod policy §7, no runaway accelerator); notifyLowBalance true. 48 GPU types listed; reference prices: A100 SXM 80GB $1.59/h secure / $1.39/h community, A100 PCIe 80GB $1.39/$1.19, A40 48GB $0.44/$0.35.
- Existing pod `generals_competition` (id `wvjrnxbpcjnr8h`): desiredStatus EXITED since 2026-08-08T11:10Z; 1× A100-SXM4-80GB at $1.59/h secure; image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`; template `runpod-torch-v240`; 16 vCPU / 251 GB RAM; 30 GB container disk; 50 GB volume at /workspace; ports 22/tcp + 8888/http; created 2026-08-07, last started 2026-08-08. No network volumes. Workspace contents NOT yet inspected (pod off; inspection requires a paid start and happens in the bounded GPU-validation step). Prior state preserved: pod not modified, wiped, or restarted by this probe.
- Next bounded action: bounded GPU validation — start the pod, verify JAX/CUDA/device detection, load MARATHON_BASELINE_V0 checkpoint remotely, run a few-update resume smoke (ratio 1.0 gate), measure TPS and utilisation, then decide reuse-vs-alternative accelerator from throughput-per-dollar, then launch SH-R1-A1/A2; pod stopped again afterwards unless a training job is actively running.

### `EV-0025` — GPU access in progress: A100 host GPU-full, bounded retry loop running, A40 fallback pre-declared

- Source class: `PROVIDER_API_EVIDENCE` + `LOCAL_RUNTIME_EVIDENCE`
- Pod start attempt returned HTTP 500 `"start pod: There are not enough free GPUs on the host machine to start this pod."` — the existing pod is pinned to machine `vmz6jgf7i93u` and its A100 is currently allocated elsewhere. Not an auth or tooling failure; the control-plane path works (REST POST /pods/{id}/start discovered from runpodctl 6fc6f8b5).
- Mitigation running: `scripts/dev/runpod_start_retry_loop.py` (bounded: 24 tries × 5 min = 2 h window; every attempt appended to `var/marathon_takeover/runpod_billing_log.jsonl`; never deletes/recreates the pod; exit 0 RUNNING / 3 still blocked / 1 error). Log: `var/marathon_takeover/runpod_start_retry.log`.
- Fallback pre-declared BEFORE results (no post-hoc hardware shopping): if the window closes HOST_FULL, provision a NEW pod on a cheaper HIGH-stock accelerator — measured stock+price snapshot at decision time: **A40 48GB, HIGH stock, $0.44/h secure / $0.35/h community** (vs A100 SXM $1.59/h with zero current stock on the pinned host). Workload rationale: the learner is a small transformer (raw.npz ≈ 6.7 MB); the CPU-side bottleneck was environment stepping, so A100-class prestige hardware is not required — throughput-per-dollar decides per RUNPOD-SPEND-2026-08-15 §6. The existing generals_competition pod remains the preferred resource if it frees (its 50 GB volume may still hold the verified `/tmp/qs-venv312` environment archive from cloud_restore_environment.sh).
- Local SSH capability confirmed: `~/.ssh/id_ed25519.pub` byte-matches the pod's env PUBLIC_KEY, so once RUNNING, pod-side execution (workspace inspection, env setup, validation, training launch) proceeds via SSH without further operator action.
- Meanwhile SH-R1-A0-CONTROL healthy on CPU: 32 updates, ratio 1.0 throughout, vloss 2.3597 → 2.3522 (mild downward trend), entropy 1.49 → 1.26, ~76 s/update.

### `EV-0026` — A40 GPU backend validated; resume semantics hold on GPU; geometry saturation ladder running

- Source class: `PROVIDER_API_EVIDENCE` + `REMOTE_RUNTIME_EVIDENCE`
- Provisioning: existing generals_competition A100 remained HOST_FULL through the retry window (attempts logged). Fallback executed per EV-0025 predeclaration: pod `quantisilico_sh_r1_a40` (id `tmmov7t54z5mbu`, machine `z3ghhcrigmrs`) created via official GraphQL `podFindAndDeployOnDemand` in EU-SE-1: 1× NVIDIA A40, $0.44/h SECURE, 9 vCPU / 50 GB RAM / 30 GB container disk / 50 GB /workspace, SSH mapped 194.68.245.124:22020; CA-MTL-1 attempt recorded machine-resource refusal first (both logged). Billing log: `var/marathon_takeover/runpod_billing_log.jsonl`.
- Environment built on pod: repo cloned at commit ce99e5d branch research/phase9g-competition-native-jax-preovernight-v1; uv-managed CPython 3.12.14 venv `.venv312`; jax[cuda12]==0.11.0 + optax 0.2.8 + numpy/pyyaml; official engine via PYTHONPATH (no engine patching). Two setup defects found and fixed remotely: image Python is 3.11 (JAX 0.11 requires >=3.12 → uv venv), PYTHONPATH needs repo root + src + third_party/generals-bots.
- Validation results (`scripts/dev/remote_gpu_bootstrap.sh`, arm gpu-smoke, 3 updates, seed 20260815): JAX 0.11.0 backend=gpu, single CudaDevice, device_kind NVIDIA A40 (no silent fallback); checkpoint loads; RATIO_FIRST 0.99998 / RATIO_LAST 1.00002 (genuine resume semantics on GPU, matching CPU smoke to 1e-5); VALID_LEARNING_SHARE 1.0; vloss/entropy track the CPU smoke within 1e-4 — training math is device-consistent.
- KEY FINDING (throughput-per-dollar, amendment §6): at the screening geometry 8×16 the A40 measures collect 1.62 TPS / end-to-end 1.25 TPS vs laptop CPU 1.4/1.2 — NO material GPU win; the JAX-native environment stepping, not the learner, is the bottleneck (consistent with prior cloud lessons on vectorised geometry). The proven cloud remedy is larger batch geometry: `cloud_a100_env_ladder.py` (32×32 … 512×32, 4 timed updates each) is RUNNING on the pod (`/workspace/ladder_run.log`) to find the A40 saturation geometry before SH-R1-A1/A2 launch; the ladder result decides the arm geometry and the continue/stop verdict for this pod. Balance at provision: $17.89; pod rate $0.44/h.

### `EV-0027` — SH-R1 GPU arms executed on A40; round verdict NO_SURVIVORS (budget-limited); saturation geometry established; pod stopped

- Source class: `REMOTE_RUNTIME_EVIDENCE` + `LOCAL_RUNTIME_EVIDENCE`
- Ladder result (A40, 4 timed updates/geometry, XLA mem fraction 0.85): e32_r32 OK, e64_r32 OK, e128_r32 OK, **e256_r32 BEST = 7,611 valid-learning TPS at 100% GPU util / 39.0 GiB VRAM**, e512_r32 RESOURCE_EXHAUSTED (OOM allocating 2.36 GiB), e64_r64 5,802 TPS, e128_r64 7,418 TPS. Log retrieved: `experiments/marathon/remote_screening_runs_gpu/ladder_run.log`.
- Arms executed sequentially at matched 8,192 transitions/update (128×64 and 512×16 both = matched batch per the HORIZON_ABLATION family): SH-R1-A1-HORIZON-64 (3 updates, 24,576 transitions) and SH-R1-A2-HORIZON-32 (3 updates, 24,576 transitions). Both: VALID_LEARNING_SHARE 1.0, ratio 0.99999–1.00001 (genuine resume on GPU), finite metrics, no integrity stops.
- ROUND VERDICT, predeclared rules applied WITHOUT post-hoc modification: ALL THREE ARMS (including the CPU control once it completes, same arithmetic) are ELIMINATED by NO_VLOSS_REDUCTION — with only 3 optimizer updates per arm at this budget, the value head cannot demonstrate reduction. This is an honest negative screening result: the integrity/pipeline machinery worked, the predeclared 25k budget is too small to produce a learning-signal screening at matched-batch large-geometry. No arm is rescued by post-hoc rule relaxation (programme integrity). Evidence retrieved locally: `experiments/marathon/remote_screening_runs_gpu/` (telemetry.jsonl + summary.json per arm; npz checkpoints stay local, gitignored).
- Resource lifecycle: pod stopped at 05:15Z immediately after the last arm (idle-pod policy §7; POD_STOP billing-logged); pod retained EXITED with persistent /workspace (env + checkpoints) for cheap restart. Total billed ≈ 1.0 h × $0.44 ≈ $0.45.
- DECLARED NEXT-ROUND DESIGN INPUT (predeclared BEFORE SH-R2 launch, not a retroactive fix): SH-R2 must budget by UPDATES at fixed geometry (e.g. ≥50 updates ≈ 410k transitions at 8,192/update, ≈ 90 s on A40 at ladder TPS) so VLOSS_REDUCTION_OVER_ROUND is measurable; horizon comparison then runs at matched update counts rather than matched tiny transition budgets. The SH-R1-CPU-PILOT control continues on laptop CPU to its 25k budget as the matched low-geometry control record.

### `EV-0028` — Live-state reconciliation: idle pods stopped; SH-R2 ALL-THREE SURVIVE; CPU control survived; SH-R3 predeclared

- Source class: `PROVIDER_API_EVIDENCE` + `REMOTE_RUNTIME_EVIDENCE` + `LOCAL_RUNTIME_EVIDENCE`
- Operator top-up confirmed: clientBalance $26.94 (was $17.89). At reconciliation time currentSpendPerHr was $2.066 while SH-R2 had FINISHED at 09:01:13Z — two idle paid accelerators confirmed:
  - `quantisilico_sh_r2_a40` RUNNING with GPU 0%/0MiB, no trainer process → evidence fetched FIRST (all telemetry/summaries/npz + round log to `experiments/marathon/screening_round_2_runs/`), then STOPPED 10:39Z (billing-logged).
  - `generals_competition` A100 RUNNING idle since “Resumed by user” 05:29:31Z (not by this executor); inspected via SSH: GPU 0%/0MiB, only sshd/nginx/sleep. Prior state preserved and inventoried BEFORE stop: /workspace holds quantsilico-generals (126 MB repo), quantsilico-runtime (1.0 GB incl. cloud_assisted_deadline_salvage_v1 and cloud_gpu_last_push_v1), transfers (59 MB); container is fresh (no /tmp/qs-venv312 — archive lived on the old machine). STOPPED (billing-logged). Idle burn ≈ $8.3 avoided by stopping both.
- SH-R2 adjudication FROM ACTUAL ARTEFACTS (not observer summary), predeclared rules applied: **ALL THREE ARMS SURVIVE** — A0-CONTROL vloss 2.3575→2.3514 (reduction +0.00617), A1-HORIZON-64 +0.00604, A2-HORIZON-128 +0.00567; VALID_LEARNING_SHARE 1.000 for all; entropy finite and rising (1.18→2.04 / 1.58→2.66 / 1.98→3.23 — no collapse); no integrity stops; 60/60 updates each. Horizon ranking at screening scale: A0 ≈ A1 > A2 (differences < 10%, NOT promotion-grade; telemetry only).
- SH-R1-A0-CONTROL (CPU, 195 updates) completed cleanly and SURVIVES the same predeclared rules: vloss reduction +0.00688, valid share 1.0, e2e 2.15 TPS. Registered as the round's matched low-geometry control.
- SH-R3-SEEDS predeclared (`experiments/marathon/screening_round_3_plan.yaml`) and registered pre-launch BEFORE any launch: 3 surviving geometries × seeds {20260815, 20260816, 20260817} = 9 arms, 60 updates each at 8,192-batch on A40; same metrics/elimination rules; survivor test = per-geometry cross-seed consistency (all seeds survive, no seed flips vloss-reduction sign); finalists then receive Stage-2 promotion-scale paired evaluation — the gameplay question cannot be answered by telemetry (EV-0021 draw regime).

### `EV-0029` — Throughput discrepancy reconciled: env-reset-pool rebuild is the bottleneck, not a metric artifact

- Source class: `LOCAL_RUNTIME_EVIDENCE` (reconciliation script `var/marathon_takeover/reconcile_tps.py`)
- Ladder TPS (e32 2,646 / e64 4,971 / e128 6,340 / e256 7,611 / e64_r64 5,802 / e128_r64 7,418) measures fully-warm updates in one long-lived process with a PRE-BUILT 4,096-board reset pool and XLA fraction 0.85; arithmetic verified (32,768/4.3054 s = 7,611).
- Runner arms measure ~126–136 steady collect TPS EVEN AFTER excluding the first 3 warmup updates (A0 132 / A1 127 / A2 136); first-collect is NOT the outlier (64 s ≈ steady 62 s), so JIT compile is not the cause.
- Root cause identified from code: the ladder builds its reset pool ONCE (4,096 boards) and reuses it, while `run_sh_r1_arm.py` calls `collect_selfplay_batch` WITHOUT a pool argument → the env builder reconstructs reset boards every collect call. ≈58× of the arm wall-time is board (re)construction, which also explains why the A40 was only ~60× faster than the laptop CPU end-to-end: both were reset-pool-bound. This is a runner plumbing gap, NOT a hardware or JAX regression.
- Declared fix for SH-R3 (runner-level, PPO_SEMANTICS UNCHANGED — reset boards are environment initialisation, not action selection): pre-build a reset pool once per arm (build_competition_reset_pool, ladder pattern) and thread it through collect; validate with an A/B update-time comparison at one geometry BEFORE the round launches, and record the measured TPS in the SH-R3 run records. GPU selection decisions must use the post-fix steady TPS, not the 7,611 warm-only figure.

### `EV-0030` — Stage 4B lane ACTIVATED in parallel with SH-R3; pool-fix GPU speedup confirmed 11×

- Source class: `LOCAL_RUNTIME_EVIDENCE` + `REMOTE_RUNTIME_EVIDENCE` (amendment ACTIVATE-STAGE-4B-PARALLEL-LANE-2026-08-15)
- SH-R3 launched on fresh A40 pod `o27sds4rsf9hjs` (EU-SE-1, machine nhe5hmyamtrg, SSH 194.68.245.240:22186; pre-creation duplicate-capacity check passed; R1 pod host still GPU-full). GPU validation PASS on the new machine: JAX 0.11.0 CUDA backend, A40, no silent fallback, resume smoke 3/3 updates.
- POOL-FIX GPU CONFIRMATION: SH-R3-A0-CONTROL-S1 finished 60/60 updates in 5 min 51 s (11:26:39→11:32:30Z) vs 65 min for the identical geometry in SH-R2 — ≈ 11× wall-clock speedup, confirming EV-0029 root cause and that A40 throughput-per-dollar at steady state is far better than SH-R2 implied. A0-S2 likewise 5 min 49 s. Post-fix steady TPS (from R3 telemetry) supersedes all prior runner TPS figures for GPU-selection decisions.
- Stage 4B dependency-safe lane ACTIVATED while SH-R3 trains (no live-training-critical files touched):
  - Packaging: deterministic submission outbox allocator `src/generals_bot/submission/outbox.py` + CLI `scripts/packaging/allocate_submission_outbox.py` — canonical `submission/outbox/qs-<candidate>-vNNN-YYYY-MM-DD.zip` naming, atomic allocation, collision refusal, SHA-256 + manifest sidecars, package_registry.json identity link, manual-upload-only policy. 4 tests (tests/unit/test_submission_outbox.py).
  - Cleanup: read-only classification inventory `scripts/dev/repo_cleanup_inventory.py` (KEEP/MIGRATE/ARCHIVE/REGENERABLE/DELETE_CANDIDATE/UNKNOWN; evidence-protected prefixes; dry run only; first report: 23 dist entries, 13 empty scaffolds, 2 staging leftovers).
  - Dashboard: registry-backed read-only API `GET /api/marathon-registry` (reader `dashboard/backend/app/readers/marathon_registry.py`) consuming Stage-3 canonical registry instead of filenames; typed kinds experiment/run/checkpoint/candidate/evaluation/opponent_reference; 20 dashboard tests pass incl. 3 new contract tests.
- Commits: 76a0212 (SH-R3 launch), af6cbe3 (outbox), 7ff03db (cleanup inventory), 036162b + ac0f11c (dashboard registry API).
- Pre-existing ruff E501 violations in dashboard/backend/app/routes/api.py were NOT touched (unrelated to the 4B additions; noted, not silently fixed).

### `EV-0031` — SH-R3 adjudication: ALL SIX ARMS SURVIVE, all three geometry families survive; SH-R4 entry condition MET; post-fix steady TPS 28–31k

- Source class: `REMOTE_RUNTIME_EVIDENCE` + `LOCAL_RUNTIME_EVIDENCE`
- Round executed 11:26:39→11:59:08Z (32.5 min for 6 arms, commit dc3ac79 pool-fixed runner) on pod o27sds4rsf9hjs; artefacts fetched FIRST, then pod STOPPED (billing-logged; zero-idle-burn observed end-to-end).
- Per-arm verdicts from actual artefacts (predeclared rules, adjudication script var/marathon_takeover/adjudicate_sh_r3.py): A0-S1 +0.00640 / A0-S2 +0.00613; A1-S1 +0.00595 / A1-S2 +0.00576; A2-S1 +0.00530 / A2-S2 +0.00506; VALID_LEARNING_SHARE 1.000 for all six; entropy finite and rising for all six; 60/60 updates each. ALL SIX SURVIVE.
- Cross-seed rule: every geometry family survives (all seeds pass; no vloss-reduction sign flip vs SH-R2). Ranking stable and reproducible across seeds: A0 ≥ A1 ≥ A2 (differences < 25% and monotone across both new seeds — a real, reproducible ordering, still telemetry-grade only).
- THROUGHPUT (supersedes all prior runner figures): steady collect TPS 28,097–31,411 (per-update deltas, updates 5–60) vs 126–136 pre-fix — pool fix worth ~220× on collect; ~4× the 7,611 warm-only ladder figure because the runner now also avoids per-update pool overheads the ladder timing excluded. End-to-end 250 s per 60-update arm. A40 $0.44/h ≈ $0.00004 per 1,000 transitions; GPU-selection decisions use these numbers.
- 12 registry records added (6 runs + 6 terminal checkpoints). SH-R4-BUDGET-ESCALATION entry condition MET: launch per the predeclared plan (240 updates/arm at surviving geometries, seeds 20260816/20260818; finalists to Stage-2 paired evaluation — telemetry never promotes).

### `EV-0032` — SH-R4 adjudication: ALL SIX ARMS SURVIVE; finalists A0-CONTROL + A1-HORIZON-64 routed to promotion-scale paired evaluation

- Source class: `REMOTE_RUNTIME_EVIDENCE` + `LOCAL_RUNTIME_EVIDENCE`
- Round executed 12:20:46→13:06:51Z (46 min for 6 arms x 240 updates, relaunch after cwd fix 7daa33f; first attempt failed in 1 s with zero telemetry) on pod tinlsf21h08qd2; artefacts fetched FIRST, pod STOPPED (billing-logged).
- Per-arm verdicts from actual artefacts (predeclared rules, var/marathon_takeover/adjudicate_sh_r4.py): A0-B16 +0.00658 / A0-B18 +0.00636; A1-B16 +0.00613 / A1-B18 +0.00605; A2-B16 +0.00555 / A2-B18 +0.00530. VALID_LEARNING_SHARE 1.000, entropy finite/rising, 240/240 updates, no wall-cap hits. ALL SIX SURVIVE; all three families survive.
- Predeclared routing (screening_round_4_plan.yaml promotion_path, 2+ survivors): FINALIST PAIR by mean vloss-reduction = A0-CONTROL (+0.00647) vs A1-HORIZON-64 (+0.00609); A2-HORIZON-128 (+0.00542) not a finalist but family retained for the record. Telemetry alone does NOT promote (EV-0021): finalists must now pass promotion-scale seat-swapped paired gameplay evaluation via the EV-0017 evaluator.
- DEPENDENCY GAP declared (not silently bridged): no serving policy exists yet for competition-native JAX PPO checkpoints (src/generals_bot/policies has heuristics + hybrid BC only). The next bounded task is a deterministic CPU serving wrapper (PPO_SEMANTICS EVAL_ONLY) with parity tests against the training-time action head, then finalist packaging, then paired eval. The horizon question is resolved enough to proceed: A0/A1 are statistically indistinguishable at screening scale, so gameplay evaluation is the correct arbiter.
- 12 registry records added (6 runs + 6 terminal checkpoints, register_sh_r4_runs.py). Steady e2e TPS at 240-update scale: 4.3k–6.3k per arm (larger rollouts amortise pool/compile better; A1 128x64 fastest).

### `EV-0033` — Promotion-scale paired evaluation: NO_PROMOTION for both finalists; horizon family is telemetry-only at this budget; the funnel worked

- Source class: `LOCAL_RUNTIME_EVIDENCE` (promotion eval per `experiments/marathon/sh_r4_finalist_promotion_eval_plan.yaml`, predeclared decision rules applied without relaxation)
- Serving gap closed first: EVAL_ONLY serving wrappers `experiments/marathon/eval_candidates/sh-r4-finalist-{a0,a1}/main.py` use the parity-proven JaxTransformerPolicy path (EV-0019) loading finalist terminal raw.npz; serving smoke PASS (1 pair, 24 s).
- Results (EV-0017 evaluator, 12 seat-swapped pairs/opponent, competition mode): A0 (36 pairs) mean_difference +0.181 aggregated but WORST/BOTTOM-QUARTILE matchup score 0.0 and CS lower bound −0.325 < margin 0.01 → NO_PROMOTION. A1 (24 pairs) mean_difference 0.0, STD 0.0, CS lower bound −0.621 → NO_PROMOTION. Both finalists LOSE to marathon_baseline_v0 and legal_random in direct play (pair scores 0.0); A0 managed draws/near-parity only against the A1 mirror.
- Scientific reading (recorded, not spun): the 2M-transition PPO checkpoints train healthily (vloss reduction, valid learning, no collapse) but that telemetry does NOT transfer to gameplay strength — at this budget the learned policy is weaker than the heuristic baseline and even loses to uniform legal play. Possible contributors to diagnose in the next family: reward sparsity/draw regime (EV-0021), lack of opponent pressure/curriculum, and serving/observation assumptions to re-audit (a policy below legal-random warrants a serving sanity probe before the next round, cheap and predeclared).
- Routing per predeclared rules: NO silent promotion. Horizon family (SH-R1..SH-R4) CLOSED as telemetry-only; the canonical next step is the spawn-distance / opponent-difficulty curriculum family (programme.yaml) with a serving sanity probe as an integrity precondition. Registered: 2 candidate records + 2 evaluation records (register_sh_r4_promotion_eval.py).
- Total SH-family cost: ~3.5 A40-hours ≈ $1.6 plus laptop CPU; the successive-halving discipline converted a vague "which horizon?" question into a falsified proxy result with preserved artefacts — the funnel's job, done cheaply.

### `EV-0034` — Serving sanity audit finds GENUINE serving defect; EV-0033 gameplay verdict INVALIDATED; bounded EVAL_ONLY repair; re-evaluation executing under unchanged predeclared rules

- Source class: `LOCAL_RUNTIME_EVIDENCE` (amendment ELITE-REPLAY-INTELLIGENCE-DURABLE-CONTINUATION §1.1/§2 engineering-defect path: REPRODUCE → DIAGNOSE → SMALLEST JUSTIFIED REPAIR → VALIDATE → RECORD → RESUME)
- Audit: `scripts/analysis/serving_sanity_probe.py`, two layers. L1 PARAMETER EFFECT PASS from the start: both finalist checkpoints differ from fresh init (max param distance 0.145/0.143), differ from each other (0.0293), and change inference outputs (actions 1324/2336 vs fresh 2767) — weights load and matter.
- L2 PROTOCOL BEHAVIOUR FAIL → root cause: the finalist wrappers passed `JaxTransformerPolicy` (reset/act interface) directly to `run_agent`, which requires the Policy protocol (`initial_state`/`act → ActionDecision`). Every candidate process crashed at handshake (`AttributeError: 'JaxTransformerPolicy' object has no attribute 'initial_state'`, exit=1). A second path defect (wrappers `parents[3]` → `experiments/` instead of repo root) was also fixed (`parents[4]`); it had been masked during evaluation because the match driver injects `PYTHONPATH=<repo>/src` and ran from repo-root cwd context.
- Contamination proof: re-read of the original run store shows 72/72 games with attribution `AGENT_FAULT` (stderr tails contain the traceback). The engine's match driver converts a dead/faulting candidate into pass-actions and forfeits, so EV-0033's "losses to legal_random" were the CRASH, not gameplay strength. EV-0033's verdict is INVALIDATED as gameplay evidence (kept in the ledger as an integrity record of the defect); the horizon family is NOT closed on that basis — it re-enters adjudication via the re-run.
- Repair (smallest justified, PPO_SEMANTICS EVAL_ONLY, parity-proven inference path untouched): additive `JaxTransformerProtocolPolicy` adapter in `src/generals_bot/competition_native_jax/jax_baseline_policy.py` bridging reset/act → Policy protocol; wrappers updated to use it. Probe after repair: PASS (well-formed 5-token actions, deterministic across runs; single-board agreement recorded as informational only since L1 proves numeric distinctness).
- Re-evaluation: same predeclared plan (`sh_r4_finalist_promotion_eval_plan.yaml`, contamination_record appended; rules unchanged, no margin relaxation) executing against fresh run dir `paired_eval_runs/sh_r4_finalist_rerun_v2`. No retraining: checkpoints are untouched.

### `EV-0035` — Clean gameplay verdict: NO_PROMOTION for both horizon finalists; every one of 120 games DRAW-at-truncation with ZERO faults; horizon family CLOSED on genuine evidence; curriculum launch unblocked

- Source class: `LOCAL_RUNTIME_EVIDENCE` (rerun of the predeclared `sh_r4_finalist_promotion_eval_plan.yaml`, contamination_record appended in EV-0034; decision rules applied unchanged)
- Integrity: 120/120 games attribution OK, candidate faults 0, no crashes, no AGENT_FAULT (vs 72/72 AGENT_FAULT in the contaminated first execution). Serving repair (EV-0034) verified end-to-end under real match conditions.
- Results (EV-0017 evaluator, seat-swapped, competition mode, fresh run dir `paired_eval_runs/sh_r4_finalist_rerun_v2`):
  - A0 (36 pairs / 72 games vs A1, marathon_baseline_v0, legal_random): ALL 72 DRAW at truncation; mean pair score 0.5; CS lower -0.0060 <= margin 0.01 -> NO_PROMOTION.
  - A1 (24 pairs / 48 games vs marathon_baseline_v0, legal_random): ALL 48 DRAW at truncation; mean 0.5; CS lower -0.1212 -> NO_PROMOTION.
- Scientific reading: the 2M-transition finalists are NOT below legal-random (the contaminated EV-0033 claim is disproven) - they are at exact DRAW PARITY with the heuristic baseline, the mirror finalist, and uniform legal play. Nobody can force a win within the 1200-turn truncation at this policy scale/budget. Under predeclared rule 2, draws are UNRESOLVED-not-wins; under rule 3, no finalist separates from marathon_baseline_v0 -> horizon round recorded as telemetry-only -> NO_PROMOTION, no silent promotion, no SH-R5.
- Horizon family (SH-R1..SH-R4 + promotion eval) CLOSED on clean evidence. The binding constraint at this budget is WIN CONVERSION, not training health or serving: candidates reach draw parity instantly but cannot convert. Next canonical family: SPAWN-DISTANCE-CURRICULUM-R1 (predeclared + registered, experiment#spawn-distance-curriculum-r1#c9255f18cd8f) which directly attacks early-contact/learning-signal structure; draw conversion remains a programme-level question for later families (reward variants, larger budgets, Stage-5 capacity).
- Cost: rerun ~90 laptop-CPU-minutes, no paid compute. SH horizon family total remains ~$1.6 GPU + CPU; the funnel again converted budget into a decisive, honest verdict.

### `EV-0036` — Curriculum R1 adjudicated: ALL SIX SURVIVE, both variants advance by predeclared rule with sub-1% margins (telemetry-grade only); ALSO: self-wake audit FAILS (ticks emitted, zero agent execution), 2.71h idle burn $1.19, remote self-termination installed

- Source class: `REMOTE_RUNTIME_EVIDENCE` + `LOCAL_RUNTIME_EVIDENCE` + `HARNESS_EVIDENCE`
- ROUND: SPAWN-DISTANCE-CURRICULUM-R1 executed 16:58:38Z-18:08:44Z (70 min for 6 arms x 240 updates @ 256x32 on pod z7zutn5ui0yo41, commit 9f0cef8) - ~2.2x faster than the 2.6h estimate (steady e2e TPS 3.6k-4.0k/arm; reset pool 120s). Artefacts fetched FIRST, integrity-verified (6/6 arms, 240/240 telemetry lines, all five files each), pod STOPPED at 20:51:15Z, billing-logged.
- ADJUDICATION (predeclared rules ONLY, var/marathon_takeover/adjudicate_curr1.py): all six arms SURVIVE (valid_share 1.000, entropy 2.10-2.12 finite/rising, all vloss reductions positive). Mean vloss-reduction: A2-FAR-21 0.006525 > A1-CLOSE-8 0.006472 > A0-CONTROL 0.006468. Both variants advance under the literal rule (beat control mean, no sign flip) - but the margins are +0.06% (CLOSE) and +0.87% (FAR): well inside cross-seed noise (S1-vs-S2 swings within a variant exceed the variant-vs-control gap). Recorded honestly: this is telemetry-grade advancement only; gameplay evaluation remains the arbiter (EV-0021/EV-0035), and no variant is promoted on these numbers.
- SELF-WAKE AUDIT (operator challenge, answered with evidence): the armed loop process SURVIVED and emitted AGENT_LOOP_TICK_MARATHON 11+ times at the 30-min cadence (terminal 2), including a clustered burst covering the 18:10-20:40 window AFTER laptop wake - proving Start-Sleep froze during Windows suspension and buffered ticks were dumped on resume. ZERO ticks invoked an agent turn: no continuation executed, the round sat unadjudicated for 2h42m. Classification: (a) text-marker emission does not drive an agent turn in this harness - the /loop skill's monitored-output wake was not an effective executor trigger here; (b) Windows sleep suspends local Qoder processes - a local loop CANNOT compute while the machine sleeps. Both are now explicit capability boundaries.
- IDLE BURN: round end 18:08:44Z, stop 20:51:15Z = 2.71h x $0.44 = ~$1.19 idle (billing-logged IDLE_INTERVAL_RECORDED). Root cause: completion-means-STOP depended on a laptop process that was suspended.
- REPAIR (remote-side, no laptop dependency): scripts/dev/remote_orchestrator_with_stop.sh pattern - orchestrator runs the round, and after the round-log end boundary writes ROUND_COMPLETE marker then calls the RunPod REST API from the pod itself (credential passed at launch, gitignored on pod) to stop its own pod. Training output is durable on the volume before stop; fetch-on-resume unchanged. Local watchdog + loop remain redundant layers. No healthy job can be terminated early: stop fires only after the orchestrator's sequential loop ends.

### `EV-0037` — Durable /loop commissioned (AGENT_LOOP_TICK_MAREXEC 30m, prompt versioned); replay legal-POV reconstruction gate IMPLEMENTED and proven on real elite data

- Source class: `HARNESS_EVIDENCE` + `LOCAL_RUNTIME_EVIDENCE`
- Durable continuation commissioned via /loop: sentinel AGENT_LOOP_TICK_MAREXEC (30 min cadence) armed; full executor instruction versioned at docs/marathon/MARATHON_DURABLE_EXECUTOR_PROMPT.md (lease -> RunPod-first reconciliation -> NEXT_SAFE_ACTION -> continue; retire only at canonical completion or ALL-scope hard blocker). Prior AGENT_LOOP_TICK_MARATHON sentinel loop deprecated per EV-0036 (proven not to drive agent turns; retained only until its terminal is reaped). Capability boundaries remain as recorded in EV-0036; remote self-stop carries the billing invariant.
- Legal fog-of-war POV reconstruction (charter §4 HARD GATE) implemented: scripts/data/replay_legal_pov.py - full-state replay -> per-player per-tick LEGAL view (conservative visibility: owned + 4-neighbours; terrain/cities persistent memory; ownership remembered; enemy armies only live while visible, stale-memory otherwise; enemy generals only when seen). Hidden state is STRUCTURALLY ABSENT, never zeroed.
- Validation: 6 synthetic-fixture tests PASS (leak guard: live hidden army 9 never surfaces - view shows stale 5; terrain persists after vision loss; ingestion idempotent; view schema whitelist) + real-data probe on DATASET-ELITE-2026-08-15-V01 replay 234266 (19x21, 151 ticks): each player sees only legal information at tick 60 (own general only; asymmetric mountain knowledge). Real-API schema variant (dict dims, string winner) covered by test.
- This unblocks the next replay milestones: action extraction + legality checks, then BC warm-start sub-experiment A predeclaration (charter §7).
