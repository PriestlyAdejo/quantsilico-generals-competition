# Marathon Design Authority

**Plan ID:** `MARATHON_REDESIGN_LOCKED_V1`

**Authority status:** Active Stage 0 bootstrap

## Purpose

This file records canonical architecture, invariants, resolved amendments, and known repository facts. It does not claim unperformed validation. `EXECUTION_PLAN.md` controls programme order; `EVIDENCE_LEDGER.md` controls evidence provenance; `CONTROL_ENGINEERING.md` controls action-modifying semantics.

## Repository facts frozen on 2026-08-14

- Handoff base commit: `4c84cb77d726aab638b68c7d8f1b37256569c4ac`.
- Official engine submodule: `third_party/generals-bots` at `9e3b9d13cca51caa1bb07db48bb85c9e90ce0462`.
- The official engine is read-only to Marathon work.
- The repository is intentionally public during early development. Active tactics are not public artefacts. Visibility changes require `REPOSITORY_PRIVACY_CUTOVER` and human approval.
- Stage 0 authority files were absent at the architect handoff and are being bootstrapped before training/evaluation/control changes.

## Canonical programme invariants

1. Reproduce and fingerprint `MARATHON_BASELINE_V0` before moving the JAX trainer or removing legacy entrypoints.
2. File hashes and semantic state hashes are separate evidence.
3. Behavioural, semantic-state, and bitwise determinism are distinct claims.
4. Evaluation uses common-map/seed seat-swapped pairs and sequentially valid inference over paired differences.
5. Candidate identity, checkpoint lineage, evaluation, and promotion are registry-driven, not inferred from filenames.
6. Platform work cannot block research after the shared baseline, evaluator, and minimum registry pass.
7. Raw and EMA are tracked from the same compatible run.
8. Architecture, observation, critic, optimizer, and capacity changes are isolated where practical.
9. Scaling uses successive halving, adaptive reference budgets, and external learning velocity.
10. The objective is the strongest qualified deployable student, not merely the strongest teacher.
11. Paid resources, visibility, destructive evidence removal, pushes without authorization, and competition uploads remain human-controlled.

## Historical references

The permanent diagnostic reference set is:

- `HEURISTIC_V2_FALLBACK`
- `SPRINT_BC_STEP450`
- `SPRINT_VALID_PPO_7M59` raw
- `SPRINT_VALID_PPO_7M59` EMA

`FORENSIC_ZERO_REWARD_35M` is invalid-learning evidence only. It is never a strength opponent or teacher.

The local `ckpt_final_u482_t7593984` checkpoint is the locked historical Stage 1 baseline source. Later checkpoints at approximately 10.00M, 14.97M, 19.36M, 25.01M, and 50.52M transitions are `KEEP / STRENGTH_UNKNOWN`. Their existence and nonzero terminal telemetry do not promote them or supersede the locked baseline without serious paired evaluation.

## Resolved amendment `ORCH-0001`

### Conflict

The operator added automatic Codex-to-implementer-to-Codex orchestration after the locked plan was persisted. The existing protocol used reviewer verdict `SHIP`; the amendment requires `ACCEPT` and adds `HUMAN_BOUNDARY`. The protocol escalated two failed attempts at the same substantive problem while the amendment supplied an absolute three-cycle repair cap.

### Resolution

The operator instruction is accepted as a bounded Stage 0B infrastructure amendment. `ACCEPT` is canonical for new automation records and is semantically equivalent to historical `SHIP`. `HUMAN_BOUNDARY` is a non-failure pause. Three cycles are an absolute ceiling; the existing two-failures rule applies earlier when the substantive-problem fingerprint repeats. This amendment changes orchestration acceptance and schemas, not the research architecture or promotion evidence.

### Current tooling fact

Codex CLI `0.147.0-alpha.6.5` is executable from the user-local Codex application binary and authenticated with ChatGPT. The WindowsApps shim is not directly executable from this shell. Cursor Agent is not installed/discoverable natively, its official Windows route is WSL, WSL inspection timed out, and the operator reports exhausted Cursor credits. Therefore simulation-safe orchestration work may proceed, but live Cursor end-to-end acceptance must remain unclaimed until the exact CLI/model/authentication/usage gates pass.

## Supersession rule

A future amendment identifies the superseded statement, repository evidence, consequence, replacement, affected acceptance criteria, decision owner, and commit. No code-only change silently supersedes this authority.
