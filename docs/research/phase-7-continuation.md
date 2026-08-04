# Phase 7 continuation (post-dashboard)

Status after Figma console integration on `feature/figma-console-integration`.

Do **not** start this work from the dashboard integration branch’s training controls.
Continue research on:

`feature/full-research-platform-v0`

## Sequence

1. Competition-size graph latency on 18×18, 18×21, 21×18, 21×21 boards.
2. One-core CPU p50 / p95 / p99 for full recurrent inference.
3. Include legal-mask and survival-shield latency.
4. Profile encoder, message passing, recurrent state, and action heads separately.
5. Optimise or reduce the graph architecture when p99 is not safely below 150 ms.
6. Run the CNN control under the same conditions.
7. Validate PFSP / population scheduling.
8. Reproduce payoff matrices; normalise meta-strategy.
9. Population opponents: Expander, Hunter, `heuristic_v1`, `heuristic_v2f_plus_planner_terminal_form`.
10. Bounded DEVELOPMENT training only.
11. No promotion-holdout seeds.
12. Generate explainability records from frozen checkpoints.
13. Learned promotion only after competitive + packaging gates.
14. No INITIAL / OVERNIGHT / MARATHON campaigns yet.
15. No neural upload until promotion and packaging pass.

## Explicit non-claims

- The ~139 ms blank 8×8 graph smoke measurement is a **risk**, not a passed latency gate.
- BC/PPO smoke accuracies are not competitive win rates.
- `HEURISTIC_DEVELOPMENT_GATE` remains FAIL until discovery evidence changes.
- `LEARNED_PROMOTION_GATE` remains NONE until a real promotion record exists.
