# Competition-Native JAX Pre-Overnight Master Specification

**Candidate family:** `competition_native_jax`  
**Authorisation:** `PRE_OVERNIGHT_REBUILD` only (overnight/upload/portal/rental/Phase 10 = false)  
**Branch:** `research/phase9g-competition-native-jax-preovernight-v1`  
**Parent closure commit:** recorded in programme state

This file is the self-contained execution specification. It must not depend on
inaccessible chat context. Official competition simulator and frozen package
bytes override this document when they conflict.

## Evidence precedence

1. Frozen package bytes / official `GeneralsEnv(mode="competition")`
2. Frozen evaluation protocols
3. Current repository commit
4. This master specification + authorised plan
5. Paper / pinned external research (provenance only)
6. Older prose

## Deployment limits (verify at execution; official docs win)

Expected (from https://generals.bot/docs#environment and local mirrors in
`src/generals_bot/rules.py`, `submission/builder.py`):

| Limit | Value |
|-------|-------|
| CPU cores | 1 |
| Process memory | 2 GB |
| Ordinary action deadline | 150 ms |
| First-action / model-load | 10 s |
| Compressed package | ≤ 50 MB |
| Unpacked package | ≤ 512 MB |
| File count | ≤ 10_000 |
| Network | unavailable |
| GPU at match | unavailable |
| Fault budget | 50 |

Resolved values are written to
`experiments/manifests/competition_native_jax_deployment_limits.json`.

Operational headroom: p99 ≪ 150 ms; RSS ≪ 2 GB; ZIP < 50 MB; cold load < 10 s.

## Action space

Training pad: 21×21. Deployment: exact rectangular H×W with H,W ∈ [18,21].
Padding cells are known mountains and are never playable.

\[
|A| = 1 + 9HW = 3970 \quad (H=W=21)
\]

Layout (canonical):

- index 0: PASS
- for cell flat `i = r*21+c`, local offsets `0..7` are moves (4 directions × 2 splits),
  offset `8` is BUILD
- absolute index = `1 + 9*i + local`

Illegal logits never enter entropy, KL, top-action ranks, or controller stats.

## Castle pricing

\[
C(x) = 35 + \sum_{s\in S_{\mathrm{own}}}\max(0, 14-2d_1(x,s))
\]

Build consumes the turn, resolves before movement, pays army from the cell,
may leave zero army, produces under official growth timing, remains a castle on
capture, and updates ownership for subsequent prices. Enemy structures do not
affect own price.

### Required price examples

- no nearby surcharge → 35
- d=6 → +2; d=5 → +4; d=4 → +6; d=3 → +8; d=2 → +10; adjacent → +12
- two structures at d=2 → 35+10+10 = 55

### Required castle tests

1. Price map parity vs official simulator  
2. Stacked surcharge  
3. Exact-price zero-army  
4. Insufficient army invalid  
5. Build on general invalid  
6. Build on existing castle invalid  
7. Enemy/neutral/fog invalid  
8. Padding invalid  
9. Production timing  
10. Build-before-move  
11. Captured castle ownership/price  
12. Enemy structures ignored  
13. Two simultaneous builds no conflict  
14. Source/package/stdio parity  

## Deathtouch / horizon

- Activates at turn 800  
- Hard cap 1200 (draw unless already terminated)  
- Simultaneous general capture = draw per competition rule  
- After activation, move onto enemy general wins regardless of defending army
  (official chase semantics)  
- Do not reimplement transitions outside the official simulator  

Fixtures: pre-800 capture; turn-800 boundary; one-unit lethal touch; successful
chase; counterattack-from-general not chase; both generals same turn; turn-1200
draw.

## Patch → cell mapping

21×21 → 7×7 = 49 patch tokens. Each patch yields 3×3×8 move + 3×3×1 build logits,
unpatchified to `move_logits[21,21,8]` and `build_logits[21,21]`. PASS separate.
Tests in `tests/competition_native_jax/test_patch_cell_mapping.py`.

## Architecture

Transformer policy/value (neutral name): depth 4, emb 192, heads 4 (initial).
HL-Gauss value: 128 bins on [−1,1], σ=0.04. EMA τ=0.999.
Deterministic observation memory (not opaque RNN as primary).

## Self-play

Symmetric: shared raw parameters both seats; independent RNG; player-specific
obs/memory; gradients from both trajectories; terminal rewards +1/−1/0.
EMA for evaluation/selection only.

## PPO contract

Collection and update must share identical mask and action support.
\[
\rho_t = \exp(\log\pi_\theta(a_t|x_t,M_t)-\log\pi_{\mathrm{old}}(a_t|x_t,M_t))
\]
Zero-update ⇒ ρ≈1. Clipped PPO; GAE γ=1, λ=0.9. Default `control_mode=OFF`.
When a controller modifies actions at collection, ratios use controlled π̃.

## Curriculum 0–4

| Stage | Content |
|-------|---------|
| 0 | Small separation; small maps; castle off only if clean switch exists |
| 1 | Larger spawn distance; affordable castles; short horizon |
| 2 | 18–21 maps; full castle prices; pre-Deathtouch |
| 3 | Full Deathtouch + 1200 cap + rectangular distribution |
| 4 | Exact competition distribution |

## Training ladder

| Run | Limits |
|-----|--------|
| Correctness | ≤8192 transitions, ≤4 updates, ≤15 min post-compile |
| Smoke | ≤100k transitions, ≤30 min |
| Short | ≤90 min wall + budget = ⌊0.85×TPS×seconds⌋ |
| Medium | ≤4 h, only if short passes |

Canonical lineage trains from scratch unless EXTERNAL_CHECKPOINT_RULE is met.
Replay/DAgger is not on the critical path.

## Daytime release sequence

1. Freeze eval protocol before reading competitive results  
2. Select base (raw vs EMA)  
3. Package/qualify `QS-P9G-COMPETITION-POLICY-DAY-V1`  
4. Telemetry noninterference  
5. Bounded controls (≤3 configs each; evidence-gated)  
6. Compare base/controlled/student → ≤1 recommendation  
7. Overnight plan with conditional parent; hard stop  

## Forbidden

Overnight execution; auto-upload; portal mutation; rented compute; Phase 10;
recommending Tactical/Hybrid; silent PASS on policy exceptions; packaging jaxlib
without size/latency feasibility.
