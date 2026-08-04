# Phase 9 / 10 terminal report — overnight readiness BLOCKED

Working branch: `fix/console-and-phase9-phase10`
Feature branch (pre-finalisation): `feature/full-research-platform-v0` @ `3c80b5c`
Recovery tag: `dashboard-before-phase9-automatic-run-3c80b5c` → `3c80b5c`

## Sequence executed

1. Safeguard G run-state maintained throughout.
2. Console correction + required MDX — integrity gate PASS, arena smoke PASS.
3. Working-branch recovery pushed; feature branch **not** mid-run FF'd.
4. DEVELOPMENT eight-arm audit → `cnn_bc_init_seed11`, `graph_bc_init_seed7`.
5. `INITIAL_READINESS_GATE=READY`.
6. Durable campaign telemetry + Training observer wired.
7. Adaptive INITIAL completed (both candidates `PLATEAU`).
8. `OVERNIGHT_READINESS_GATE=BLOCKED` — no overnight, no holdout, no package.

## Adaptive INITIAL (recorded)

| Candidate | Stop | Env steps | Best score rate | Wins/Draws/Losses | Protocol faults |
| --- | --- | --- | --- | --- | --- |
| cnn_bc_init_seed11 | PLATEAU | 1536 | 0.5 | 0/2/0 | 200 |
| graph_bc_init_seed7 | PLATEAU | 1536 | 0.5 | 0/2/0 | 200 |

Primary metric was validation score rate. Both candidates plateaued on draw-only score_rate with high protocol faults during learned-checkpoint vs expander validation. Holdout seeds were never opened.

## Overnight blockers

- best validation wins = 0
- protocol_faults > 0
- draw-dominated score_rate with zero wins

## Terminal outcome

`OVERNIGHT_READINESS_BLOCKED` — integrate the blocked-readiness evidence commit when tests pass; no ZIP; active portal bot remains `heuristic_v2f_plus_planner_terminal_fix`.

## Artefacts

- `experiments/manifests/development_arm_audit.json`
- `experiments/manifests/initial_readiness_gate.json`
- `experiments/manifests/adaptive_initial_campaign.json`
- `experiments/manifests/overnight_readiness_gate.json`
- `var/dashboard/campaigns/initial_*.json`
- `configs/training/initial/adaptive_initial_v1.yaml`
