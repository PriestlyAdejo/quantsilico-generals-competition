---
name: quantisilico-elite-replay-research
description: Operational rules for elite/top-player replay research in the QuantSilico marathon - snapshot acquisition, immutable provenance, legal fog-of-war POV reconstruction, anti-overfit splits, and BC/clone/scenario consumption. Use when fetching generals.bot leaderboard/replay data, building or validating replay datasets, or designing replay-augmented training experiments.
---

# Elite Replay Research (QuantSilico Marathon)

Authority: `docs/marathon/ELITE_REPLAY_AUGMENTATION.md` + `EXECUTION_PLAN.md` §9.5.
This skill is operational knowledge only; the canonical programme wins conflicts.

## Acquisition

- Public API: `https://www.generals.bot/api/leaderboard`
  (listing; `?matches=1&player=NAME`; `?replay=ID` full state timeline).
- Use `scripts/data/fetch_elite_replays.py` (stdlib, bounded workers, 403 backoff,
  resume-in-place while unsealed). Never hammer the API; pilots stay tiny.
- Dataset id `DATASET-ELITE-YYYY-MM-DD-VNN` under `experiments/datasets/elite_replays/`.
- Raw payloads are IMMUTABLE once sealed by `manifest.json`; derived datasets get
  new identities. Register snapshots in the Stage-3 registry as dataset records.

## Hard legality gate (fog of war)

Replay timelines contain FULL state. Before ANY replay-trained policy enters
the funnel, prove legal-POV parity against the canonical competition
observation path (visible/hidden tiles, armies, legally observable features,
perspective, action mask, history/memory). Full-state data may only label
OFF_POLICY_AUXILIARY analysis that never reaches the deployed policy.

## Splits (never row-random)

Player-disjoint, time-disjoint, seed/map-disjoint, and style-stressed holdouts;
per-player caps; track contribution per player. No usernames fed to the policy
unless separately predeclared. A ResBot-specialist that regresses broadly does
not promote.

## Consumption mechanisms (never replay-vs-live self-play)

BC warm start; auxiliary replay loss (declared OFF_POLICY_AUXILIARY);
scenario curriculum; opponent imitation clones (`ELITE_CLONE_*`, reactive);
disagreement mining (selection signal, not superiority proof).

## Evidence requirements

Promotion needs generalised gains beyond matched controls (frozen Stage-2
strength, held-out elite/player/style/seed, efficiency, regression resistance)
via the EV-0017 evaluator — never imitation accuracy or one named opponent.
