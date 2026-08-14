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

## Open evidence requirements

- Complete Stage 0 artefact/worktree/entrypoint inventory.
- Semantic state hashes and deterministic resume capsule for Stage 1.
- Serious paired strength evaluation of historical and later checkpoints.
- Exact Cursor Agent executable, authentication, supported-model listing, and available usage before live orchestration acceptance.
