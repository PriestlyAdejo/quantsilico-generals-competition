---
name: quantisilico-elite-replay-research
description: Operational rules for elite/top-player replay research in the QuantSilico marathon - snapshot acquisition, immutable provenance, legal fog-of-war POV reconstruction, anti-overfit splits, and BC/clone/scenario consumption. Use when fetching generals.bot leaderboard/replay data, building or validating replay datasets, or designing replay-augmented training experiments.
---

# Elite Replay Research (QuantSilico Marathon)

Authority: `docs/marathon/ELITE_REPLAY_AUGMENTATION.md` + `EXECUTION_PLAN.md` §9.5.
This skill is operational knowledge only; the canonical programme wins conflicts.

## Competition authority (EV-0042 — read before any replay/action work)

Action validity / legality / observation semantics come from the COMPETITION
ruleset, NOT generic present-day generals.io. Order: (1) the pinned engine
`third_party/generals-bots` (submodule SHA is the executable source of truth
and must be recorded in every derived dataset + audit report); (2) the
official generals.bot competition rules/docs; (3) QuantSilico parity
contracts; (4) the replay payload; (5) generic generals.io only where the
competition inherits it.

Operational rules:

- No single `is_legal_action` boolean. Classify: PROTOCOL_VALID /
  ENGINE_EXECUTED / ENGINE_SILENT_PASS / RUNNER_FAULT / PROCESS_FORFEIT,
  plus a SEPARATE observation-legality / leakage question. Silent pass is a
  well-formed-but-invalid action, NOT a runner fault; runner faults and
  forfeits are not reconstructable from replay payloads — say so explicitly.
- Engine-oracle-first: use the pinned engine's primitives as the oracle
  (`scripts/data/replay_engine_oracle.py` wraps state reconstruction +
  engine-exact move/build classification; `game.step` / `build_castles` /
  `deathtouch` / competition composition builds→deathtouch→step). Any
  duplicate validator needs DIFFERENTIAL parity tests vs the engine
  (`tests/unit/test_replay_engine_oracle.py`) before it filters data.
- Honour competition specifics: rectangular 18–21 boards; player-built
  castles with dynamic crowding price (captured castles count as own);
  simultaneous resolution (chasing > reinforcing > smaller army;
  build-before-move; strict `>` combat favours defender); simultaneous
  general capture = DRAW; deathtouch from turn 800 (`general_positions`
  static); 1200-turn draw cap.
- Keep TRUE_COMPETITION_STATE (oracle/analysis) separate from
  LEGAL_PLAYER_OBSERVATION (policy input). Hidden destination properties are
  recorded as visible/known/unknown, never BC features.
- Silent-pass expert actions are NOT automatic BC labels; predeclare handling.
- Prove replay timing alignment (engine-step-from-tick trace) before large
  audits: `scripts/data/replay_alignment_trace.py`.
- Audits use absolute player indices; test BOTH seats; real dims vs 21×21
  training padding are distinguished (OUT_OF_BOUNDS, not "mountain").

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
