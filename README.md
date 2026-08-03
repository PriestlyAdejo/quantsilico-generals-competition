# quantsilico-generals-competition

Private QuantSilico repository for the **EquiLibre × ÚFAL Generals Competition**.

This tree holds the private bot, evaluation harness, training system, model
weights, experiment logs, and active tactics. It is **not** a public portfolio
project and must stay separate from:

- `priestlyadejo-portfolio`
- `quantsilico`
- other QuantSilico competition repositories

## Repository boundaries

- Private GitHub repository: `PriestlyAdejo/quantsilico-generals-competition`
- All rights reserved (see `LICENSE`)
- Official engine retained as a Git submodule under `third_party/generals-bots`
- No active tactics are published in the portfolio
- No fabricated results

## Official engine submodule

Source of truth: https://github.com/strakam/generals-bots

Bootstrap pin and rules: [`third_party/README.md`](third_party/README.md)

Do not silently modify the official engine.

## Current implementation status

**Bootstrap scaffold only.**

Implemented:

- repository layout
- private Python package stub (`generals_bot`)
- development bootstrap / verification scripts
- architecture and research notes

Not implemented:

- bot strategies
- reinforcement learning
- submission packaging / uploads
- HGB-PSRO learning components (documented as planned only)

## Planned tracks

### Heuristic track

Deterministic / rules-conforming heuristics under `baselines/` and
`submission/sprint_heuristic/` (future work).

### HGB-PSRO research track

Hierarchical Graph-Belief Policy + PPO best-response + lightweight PSRO.

See [`docs/architecture/0003-hgb-psro-research-direction.md`](docs/architecture/0003-hgb-psro-research-direction.md).
All learning components are **planned, not implemented**.

## Bootstrap on Windows

Requirements:

- Python **3.12.10**
- Git
- Git Bash at `C:\Program Files\Git\bin\bash.exe` for shell agents / smoke matches

```powershell
Set-Location 'C:\Users\pries\Documents\Projects\quantsilico-generals-competition'
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\dev\bootstrap.ps1
```

If official competition dependencies fail on native Windows, capture the
failure, then:

```powershell
.\scripts\dev\bootstrap.ps1 -ScaffoldOnly
```

Scaffold-only mode installs the private package and development tools only.
It sets `environment_parity: false` and must not be described as a verified
competition environment.

## Bootstrap under WSL / Linux

A real Linux distro is preferred for sandbox parity. Then:

```bash
cd /mnt/c/Users/pries/Documents/Projects/quantsilico-generals-competition
./scripts/dev/bootstrap.sh
```

Missing WSL does not block repository work; it only means Linux sandbox parity
is not yet established.

## Validation

```powershell
.\.venv\Scripts\python.exe scripts\dev\verify_environment.py
.\.venv\Scripts\python.exe scripts\dev\verify_repository.py
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m pytest
```

After a clean commit/push:

```powershell
.\.venv\Scripts\python.exe scripts\dev\verify_repository.py --require-clean
```

## Official smoke match (Windows / Git Bash)

Only after a successful full bootstrap (not scaffold-only):

```powershell
$gitBash = 'C:\Program Files\Git\bin\bash.exe'
& $gitBash -lc @'
cd /c/Users/pries/Documents/Projects/quantsilico-generals-competition
source .venv/Scripts/activate
python third_party/generals-bots/competition/matchup.py \
  third_party/generals-bots/competition/agents/expander_python/run.sh \
  third_party/generals-bots/competition/agents/expander_python/run.sh \
  --mode competition \
  --seed 0
'@
```

This is a Windows/Git-Bash smoke test, not proof of exact Linux sandbox parity.
