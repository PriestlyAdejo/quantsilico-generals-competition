# AGENTS.md

Instructions for coding agents working in this private competition repository.

## Before editing

1. Inspect the relevant paths and the official engine submodule.
2. Confirm you are not inside `priestlyadejo-portfolio` or the main `quantsilico` product repository.
3. Prefer the smallest change that advances one stage.

## Source of truth

- Official rules, protocol, runners, and sandbox dependency lock live in
  `third_party/generals-bots`.
- Do not silently patch the official engine.
- Official runtime versions come from
  `third_party/generals-bots/competition/requirements.txt`.

## Implementation discipline

- Implement **one stage at a time**.
- Add tests for every behavioural change.
- Record experiments under `experiments/` with seeds, commits, and commands.
- Do **not** invent results, metrics, or opponent claims.
- Do **not** expose active tactics in public artefacts.
- Do **not** push unless the task explicitly permits it.
- Do **not** upload competition submissions unless explicitly asked.
- Do **not** modify the portfolio repository from this worktree.
- Do **not** implement RL/strategy during bootstrap-only tasks.

## Bootstrap notes

- Prefer Python 3.12.10.
- Full bootstrap installs official requirements, then the engine with
  `--no-deps`, then `.[dev]`.
- `-ScaffoldOnly` / `--scaffold-only` is for recovery after a recorded
  official-dependency failure. It is not environment parity.
- `verify_repository.py` does not require a clean tree by default; use
  `--require-clean` after commit/push.
- Preserve LF + Git executable bits on `*.sh`, especially `run.sh` / `build.sh`.
- Invoke Git Bash via `C:\Program Files\Git\bin\bash.exe`, never an ambiguous
  generic `bash` from PowerShell.
