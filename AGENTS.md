# AGENTS.md

Cross-tool instructions for coding and research agents working in this Generals.io competition repository.

## Before editing

1. Confirm the Git root is `quantsilico-generals-competition`, not `priestlyadejo-portfolio` or the main `quantsilico` product repository.
2. Inspect the relevant repository paths and the official engine submodule.
3. Read the Marathon authorities listed below when the work touches Marathon training, evaluation, deployment, control, research operations, packaging, or repository structure.
4. Prefer the smallest bounded change that advances the current stage and has explicit acceptance evidence.

## Marathon authority contract

Read each of these files where it exists, in this order:

1. `docs/marathon/EXECUTION_PLAN.md`
2. `docs/marathon/AGENTIC_EXECUTION_PROTOCOL.md`
3. `docs/marathon/DESIGN_AUTHORITY.md`
4. `docs/marathon/EVIDENCE_LEDGER.md`
5. `docs/marathon/CONTROL_ENGINEERING.md`
6. `configs/marathon/programme.yaml`
7. `experiments/marathon/ACTIVE_STATE.json`

`docs/marathon/EXECUTION_PLAN.md` is the architecturally locked programme. Repository files, not chat history, are shared memory. A fresh agent must be able to act safely using repository state alone.

The Stage 0 authority files that do not yet exist must be created and validated before Marathon training, evaluation, or action-control behaviour is changed. Do not silently treat a missing authority as approved or complete.

If code, configuration, evidence, and documentation disagree, do not silently choose one. Record a `PLAN_CONFLICT` using `docs/marathon/AGENTIC_EXECUTION_PROTOCOL.md`, determine repository truth, obtain the required decision, and update the authorities and acceptance criteria together.

Every session that changes Marathon state must update `experiments/marathon/ACTIVE_STATE.json` before stopping. Unknown state is written as `UNKNOWN` with a reason; an empty list means the field was audited and nothing was found.

## Roles and escalation

- Codex GPT-5.6 Sol is the architect, research planner, conflict resolver, and fresh independent reviewer.
- Cursor Grok 4.6 is the default bulk implementer.
- Implementers may identify defects but may not silently redesign the programme.
- Two failed attempts at the same substantive problem require architect escalation instead of uncontrolled retries.
- Implementation work may not report silent success, an empty diff, or test-free completion. Use the status and evidence contract in `docs/marathon/AGENTIC_EXECUTION_PROTOCOL.md`.

## Source of truth

- Official rules, protocol, runners, and sandbox dependency lock live in `third_party/generals-bots`.
- Do not silently patch the official engine.
- Official runtime versions come from `third_party/generals-bots/competition/requirements.txt`.
- Do not invent results, metrics, provenance, or opponent claims.
- Record experiments under `experiments/` with stable identity, seeds, commits, commands, budgets, stop reasons, and evidence.

## Implementation discipline

- Implement one bounded task at a time in each worktree. After the shared baseline, evaluator, and minimum registry prerequisites pass, independent Stage 4A research and Stage 4B platform tasks may proceed concurrently only in separate non-conflicting scopes or worktrees.
- Add tests for every behavioural change and run the acceptance checks appropriate to the risk.
- Anything that changes action selection must declare exactly one `PPO_SEMANTICS` value: `UNCHANGED`, `PRE_SAMPLING_MASK`, `OFF_POLICY_AUXILIARY`, or `EVAL_ONLY`. Ambiguous semantics fail validation before training.
- Preserve unique checkpoints, manifests, datasets, replays, hashes, and research evidence. Initial cleanup classes are `KEEP`, `MIGRATE`, `ARCHIVE`, `REGENERABLE`, `DELETE_CANDIDATE`, and `UNKNOWN`.
- Do not implement RL or strategy during bootstrap-only or documentation-only tasks.
- Do not expose active tactics in public artefacts. The repository is currently intentionally public; any visibility change requires the human-controlled `REPOSITORY_PRIVACY_CUTOVER` decision.
- Do not create paid resources, remove potentially unique evidence, change repository visibility, push, or upload a competition submission unless explicitly authorized by the human owner.
- Dashboard APIs never receive arbitrary shell, browser-supplied filesystem, Git mutation, paid-resource, visibility-change, or competition-upload authority.
- Do not modify the portfolio repository from this worktree.

## Bootstrap notes

- Prefer Python 3.12.10.
- Full bootstrap installs official requirements, then the engine with `--no-deps`, then `.[dev]`.
- `-ScaffoldOnly` / `--scaffold-only` is for recovery after a recorded official-dependency failure. It is not environment parity.
- `verify_repository.py` does not require a clean tree by default; use `--require-clean` after commit/push.
- Preserve LF and Git executable bits on `*.sh`, especially `run.sh` and `build.sh`.
- Invoke Git Bash via `C:\Program Files\Git\bin\bash.exe`, never an ambiguous generic `bash` from PowerShell.
