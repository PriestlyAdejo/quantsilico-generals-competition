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
