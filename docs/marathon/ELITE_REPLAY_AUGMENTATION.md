# ELITE_REPLAY_AUGMENTATION — Stage 4A research family charter

Operator amendment `ELITE-REPLAY-INTELLIGENCE-DURABLE-CONTINUATION-2026-08-15`.
ADDITIVE to the canonical Stage 4A programme; it replaces no existing family.
This family is evidence-gated like every other family: PREDECLARE → REGISTER →
PILOT → health gate → successive halving → gameplay evaluation → promote/reject.

## 0. Authority: the COMPETITION ruleset defines action/observation semantics (operator amendment 2026-08-16, EV-0042)

Action validity, legality, observation semantics and replay interpretation are
defined against the QuantSilico competition's pinned ruleset and engine —
NEVER generic/present-day generals.io behaviour. Authority order:

1. the PINNED competition engine in `third_party/generals-bots` (submodule
   SHA recorded in every derived dataset and audit report) — executable
   source of truth;
2. the official generals.bot competition rules/docs
   (`https://www.generals.bot/docs`, `https://www.generals.bot/rules`);
3. existing QuantSilico competition parity tests/contracts;
4. the replay payload;
5. generic generals.io behaviour ONLY where the competition inherits it.

Binding consequences:

- Single "legal/illegal" booleans are BANNED for action semantics.
  Classification distinguishes at minimum: PROTOCOL_VALID (five-int
  `kind row col dir split`), ENGINE_EXECUTED, ENGINE_SILENT_PASS
  (well-formed but invalid under competition rules; NOT a runner fault),
  RUNNER_FAULT and PROCESS_FORFEIT (not reconstructable from replay
  payloads — audits must say so explicitly), and OBSERVATION_LEGALITY /
  information-leakage as a SEPARATE question.
- Engine-oracle-first: reuse the pinned engine's transition/validity
  primitives (`game.step` validity predicates, `build_castles.build_cost_grid`
  and `apply_build_actions`, `deathtouch.step`, the competition composition
  builds→deathtouch→step) as the oracle; any duplicate validator must pass
  DIFFERENTIAL tests against the engine before it may filter data.
- Competition specifics that MUST be honoured: rectangular 18–21 boards;
  player-built castles with dynamic crowding pricing (captured castles count
  as own structures); simultaneous-turn resolution (chasing > reinforcing >
  smaller army; build-before-move; strict `>` combat favours the defender);
  simultaneous general capture resolves as a DRAW; deathtouch from turn 800
  (`general_positions` is static — the touch target is the general's tile);
  hard 1200-turn draw cap; invalid actions are silent passes, malformed/late
  replies are runner faults, crashes are forfeits.
- TRUE_COMPETITION_STATE (full state, oracle/analysis only) and
  LEGAL_PLAYER_OBSERVATION (`replay_legal_pov`, policy inputs only) stay
  separate objects; hidden destination properties are recorded as
  visible/known/unknown, never used as BC features.
- Silent-pass expert actions are NOT automatic BC labels; their handling
  rule is predeclared per §7A before any large-scale training.
- Replay timing alignment must be PROVEN (obs → attempted actions →
  resolution → true state T+1 trace against the engine) before large audits;
  mismatches are reported, never papered over.
- Replays use absolute player indices; perspective-relative encoding
  (1=ME/2=OPPONENT) is the protocol layer's job; audits test BOTH seats.
  Real board dims are distinguished from 21×21 training padding (out of real
  bounds = OUT_OF_BOUNDS, never "mountain").
- Derived datasets record: engine SHA, ruleset/mode, parser/POV/extraction/
  audit versions, classification policy, silent-pass handling, content
  hashes. Engine/rules changes create NEW derived versions, never silent
  reinterpretation.

## 1. Hypothesis

Elite replay augmentation improves GENERALISED external gameplay strength,
strategy diversity and/or sample efficiency beyond matched pure-self-play
controls, without overfitting to individual leaderboard opponents and without
leaking hidden information. It is NOT "download the #1 player's games and
train until we beat that player."

## 2. Source and provenance

- Source input: operator-supplied gist `https://gist.github.com/mrinmoy2developer/c5ee19453a77947d7941deb61318f5fb` (`fetch_top_replay.py` concept). Inspect the live public API before implementation; the remote source may evolve.
- If verbatim reuse has licensing/provenance ambiguity, implement equivalent behaviour from the observed public API contract instead of vendoring.
- Respect endpoint throttling with backoff; bounded parallel workers; resumable downloads.
- Raw snapshots are immutable; provenance is preserved end-to-end.

## 3. Dataset identity (minimum contract)

`DATASET-ELITE-YYYY-MM-DD-VNN` — record at minimum: dataset ID; fetch
timestamp; source endpoint; source implementation/version; leaderboard
snapshot hash; selected players; ratings/ranks where available; replay IDs;
match seeds; player pair; match timestamp; turn count; result; dedup policy
(seed + player pair collapses the two seat-order representations); replay
payload hashes; observation-reconstruction version; action-extraction
version; data split; content hash; filtering parameters. Derived datasets
receive separate identities. Never overwrite a snapshot because the
leaderboard moved.

## 4. Fog-of-war legality gate (HARD)

Replays contain full state timelines. The deployed policy must NEVER receive
privileged information. Pipeline: full replay → player-specific perspective →
exact legal observation reconstruction → training sample. Before any
replay-trained policy enters the funnel, prove replay-observation parity
against the canonical competition observation path: visible/hidden tiles,
public features, army values, legally observable generals/cities/castles,
perspective, turn/time scalars, action mask, history/memory semantics.
Full-state data may be retained only for explicitly OFF_POLICY_AUXILIARY
analysis labels that never reach the deployed policy.

## 5. Anti-overfitting splits (no row-level random splits)

- Player-disjoint holdout: some elite identities fully absent from training.
- Time-disjoint holdout: later replays held out from earlier snapshots.
- Seed/map-disjoint holdout: no closely-related games across the split.
- Style-disjoint/stressed holdout where clustering evidence supports it.
- Per-player caps or balanced sampling; track effective contribution per
  player. The corpus must not silently become "95% one-player imitation".
- No leaderboard usernames fed to the policy unless identity-conditioned
  policies become a separately predeclared experiment.

## 6. Legitimate usage mechanisms (never "self-play against a recording")

Historical actions are not an interactive opponent. Allowed mechanisms:
behavioural cloning / pretraining; auxiliary replay loss; replay-derived
curriculum/scenarios; strategy/style analysis; opponent-policy imitation
clones (which DO react to novel states); DAgger/disagreement-state mining;
MPC/counterfactual state selection.

## 7. Sub-experiments (each predeclared and registered before launch)

- A. BC warm start: canonical self-play init vs elite BC warm start → same
  PPO continuation, matched RL budget. Measure BC accuracy, held-out-player
  accuracy, learner health, sample efficiency, external strength,
  generalisation, forgetting. BC accuracy NEVER promotes; gameplay is authority.
- B. Auxiliary replay loss during RL: `L_total = L_PPO + beta(t)*L_replay`,
  predeclared beta schedules; semantics declared OFF_POLICY_AUXILIARY.
- C. Elite opponent clones: named `ELITE_CLONE_<dataset/player/style/version>`,
  fidelity measured on held-out legal observations, robustness tested
  off-distribution; included in population work player-balanced; never train
  exclusively against one clone.
- D. Replay-derived scenario curriculum (rush, deficit defence, advantage
  attack, threatened general, fog uncertainty, overextension, counterattack,
  late-game pressure, castle decisions, comebacks). Replay-prefix
  reconstruction only if legal and reliable; otherwise synthetic states
  matched to replay-derived features. No future information to the policy.
- E. Elite disagreement/failure mining: champion vs elite actions on legal
  held-out observations; disagreement is a state-selection signal, NOT proof
  the elite action is better.

## 8. Promotion gate

Promotion requires improvement beyond matched controls on an appropriate
combination of: frozen Stage-2 benchmark strength; held-out elite strength;
unseen-player/style performance; unseen-seed performance; sample efficiency;
compute efficiency; regression resistance. NOT for: imitation accuracy alone,
beating one named opponent, prettier curves, or one short leaderboard window.

## 9. External-strength telemetry (supplement, not authority)

Persist timestamped leaderboard snapshots via the public API (bot/player,
exposed rating/rank, matches, W/L/D where exposed). Do not fabricate metrics
the API does not expose. Derive rate metrics only where observations suffice
(rating change per match/day, rank velocity, strength gain per M transitions
and per GPU-hour). The internal frozen Stage-2 evaluator remains the primary
scientific comparison; the public ladder tracks ecosystem health.

## 10. Generalisation report

Versioned multi-metric report for serious candidates: frozen-baseline
strength, champion strength, held-out elite strength, held-out player/style
gaps, unseen-seed strength, replay-vs-control gap, forgetting penalty, public
rating trajectory where available. A scalar `GENERALISATION_SCORE` may only
appear later with explicitly declared, validated weighting.

## 11. Stage reuse

- Stage 5: teacher research considers replay pretraining where Stage-4
  evidence warrants; a negative result on the small learner does not condemn
  the dataset for larger teachers (record architecture-specific conclusions).
- Stage 6: legal replay states feed castle decision analysis, counterfactual
  state selection, successor-value supervision, targeted DAgger, belief
  modelling, MPC/Expert Iteration, residual/reranking, risk-state discovery.
  Historical full-state data must never silently enter the policy observation.
- Stage 7: qualified clones may extend the population (champions, baselines,
  heuristics, Stage-4/5/6 candidates, elite clones, style-cluster clones,
  historical snapshots, counter specialists) without dominating it; fixed
  broad population first, adaptive sampling (e.g. PFSP) only afterwards.

## 12. Rotating datasets

Versioned snapshots over time (different train/holdout player sets and time
windows). Previous snapshots are never destroyed.

## 13. Tests (minimum)

Leaderboard parsing; match selection; seat-order dedup; resume; retries/
backoff; corrupt-replay rejection; content hashes; immutable manifest; legal
observation parity; action extraction + competition-engine action
classification with DIFFERENTIAL parity tests against the pinned engine
(§0); timing-alignment trace before large audits; no hidden-state leakage;
player/time-disjoint splits; seed dedup; per-player caps; deterministic
derived shards; registry record validation. Fixtures/mocked responses; never
hammer the public API in unit tests.
