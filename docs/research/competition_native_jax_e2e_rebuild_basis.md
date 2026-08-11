# End-to-end JAX rebuild — methodological / provenance basis

## Official MIT simulator (hot-path allowed when functional)

| Field | Value |
|-------|-------|
| Repository | https://github.com/strensio/generals-bots |
| Path | `third_party/generals-bots` |
| Commit | `9e3b9d13cca51caa1bb07db48bb85c9e90ce0462` |
| Licence | MIT (`third_party/generals-bots/LICENSE`) |

### Files inspected / reused in compiled hot path

| File | Role |
|------|------|
| `generals/core/game.py` | `GameState`, `@jax.jit step`, `get_observation`, `create_initial_state` |
| `generals/core/action.py` | `compute_valid_move_mask` |
| `generals/core/observation.py` | Observation PyTree |
| `generals/core/env.py` | competition mode presets (`pad_to=21`, truncation 1200, DT 800) |
| `generals/core/grid.py` | `generate_grid` for rectangular boards |
| `generals/modifiers/build_castles.py` | `apply_build_actions`, `build_cost_grid`, `strip_neutral_castles` |
| `generals/modifiers/deathtouch.py` | `step(..., turn=800)` |
| `competition/matchup.py` | `make_transition` composition reference (build then DT/base) |

Attribution: Copyright (c) 2024 Matej Straka — MIT. QuantSilico wraps these primitives; it does not claim authorship of the transition kernels.

### QuantSilico-owned (not copied from external research)

- 3970-action codec packing / legal-mask layout
- observation memory PyTree and channel packing for the depth-4 transformer
- fused `vmap` + `lax.scan` self-play collect
- JAX PPO / GAE / EMA / Optax learner
- curriculum / train entrypoints under `competition_native_jax`

## External research (method only)

| Source | Status |
|--------|--------|
| `external_paper_method` | `UNPINNED_PENDING_CLONE` — methodological inspiration only |
| `external_public_code` | `UNPINNED_PENDING_CLONE` — inspection only; **not copied** |

Runtime identifiers remain `competition_native_jax`. No AverageJoe / author-derived IDs in code.

## Architecture classification target

`END_TO_END_COMPETITION_JAX_ROLLOUT`

Forbidden as success: `BATCHED_HOST_ENV_JAX_POLICY`.
