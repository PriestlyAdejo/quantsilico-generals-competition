"""Reward shaping knobs for the REWARD-SHAPING-R1 family (predeclared EV-0044).

Background (EV-0035/38/39/43 accumulation): five independent 2M-transition
PPO lines all land at exact draw parity. The pinned competition engine reward
is PURELY TERMINAL (+1 win / -1 loss / 0 draw, generals/core/env.py), so in a
near-100% draw regime the policy gradient receives almost no signal; WIN
CONVERSION is the binding constraint. This family tests whether a bounded
progress signal on non-terminal ticks converts into wins without harming
learner health or being reward-hacked.

PPO_SEMANTICS: UNCHANGED. Action selection, legal masks, sampling and serving
are untouched; shaping changes ONLY the training reward fed to GAE during
self-play rollouts. Mode "none" is byte-identical to the control path.

Modes (applied ONLY to the trained seat's reward stream):
  none       - identity (control arms).
  kill_delta - r' = r + beta * (opponent army lost - own army lost) on
               strictly-alive ticks. Zero-sum symmetric under seat swap.
  potential  - army-total potential (Ng et al. 1999) at gamma = 1.0:
               Phi(s) = log(own army total + 1) - log(opponent army total + 1).
               WARNING (EV-0046): army totals stay EXACTLY symmetric in
               fresh-window self-play until combat, so this mode is
               signal-vacuous in the 32-tick screening regime. Retained for
               completeness and longer-horizon follow-ups.
  land_potential - land-territory potential at gamma = 1.0:
               Phi(s) = log(own land + 1) - log(opponent land + 1);
               alive ticks get Phi(s') - Phi(s); terminal ticks get -Phi(s)
               (Phi(terminal) := 0). Return-invariant (preserves the optimal
               policy). Expansion asymmetry arises within the first ticks of
               every game, so this mode carries real signal in the screening
               regime (EV-0046 probe: 42% nonzero ticks).

All potential modes: alive ticks get Phi(s') - Phi(s); episode-end ticks
(win/loss AND the 1200-turn hard draw) get -Phi(s) (Phi(terminal) := 0),
return-invariant at gamma = 1.0. Terminal/truncation ticks keep the ENGINE
reward untouched otherwise (post-terminal ownership transfer would
contaminate deltas).
"""

from __future__ import annotations

import jax.numpy as jnp

MODE_NONE = "none"
MODE_KILL_DELTA = "kill_delta"
MODE_POTENTIAL = "potential"
MODE_LAND_POTENTIAL = "land_potential"
VALID_MODES = (MODE_NONE, MODE_KILL_DELTA, MODE_POTENTIAL, MODE_LAND_POTENTIAL)

_ACTIVE_MODE = MODE_NONE
_ACTIVE_BETA = 0.0


def set_active_shaping(mode: str, beta: float) -> None:
    """Trace-time configuration (module constant inside jitted scans)."""
    global _ACTIVE_MODE, _ACTIVE_BETA
    if mode not in VALID_MODES:
        raise ValueError(f"unknown reward shaping mode: {mode!r}")
    if not (beta >= 0.0):
        raise ValueError("reward shaping beta must be >= 0")
    _ACTIVE_MODE = mode
    _ACTIVE_BETA = float(beta)


def active_shaping() -> tuple[str, float]:
    return _ACTIVE_MODE, _ACTIVE_BETA


def _player_totals(state) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batched engine states: ownership is (B, 2, H, W); seat 0 = trained seat."""
    own = jnp.sum(state.armies * state.ownership[:, 0], axis=(-2, -1))
    opp = jnp.sum(state.armies * state.ownership[:, 1], axis=(-2, -1))
    return own.astype(jnp.float32), opp.astype(jnp.float32)


def _potential(state) -> jnp.ndarray:
    own, opp = _player_totals(state)
    return jnp.log(own + 1.0) - jnp.log(opp + 1.0)


def _land_potential(state) -> jnp.ndarray:
    """Log-ratio of owned land counts; asymmetric from the first captures."""
    own = jnp.sum(state.ownership[:, 0], axis=(-2, -1)).astype(jnp.float32)
    opp = jnp.sum(state.ownership[:, 1], axis=(-2, -1)).astype(jnp.float32)
    return jnp.log(own + 1.0) - jnp.log(opp + 1.0)


def shape_step_rewards(
    states,
    next_states,
    terminal_rewards: jnp.ndarray,
    terminated: jnp.ndarray,
    truncated: jnp.ndarray,
    mode: str,
    beta: float,
) -> jnp.ndarray:
    """Shaped reward for the trained seat over one engine step.

    Identity for mode "none". Shaped increments apply ONLY where the episode
    is strictly alive (not terminated, not truncated); terminal ticks return
    the engine reward unchanged.
    """
    if mode == MODE_NONE or beta == 0.0:
        return terminal_rewards
    alive = (~(terminated.astype(bool) | truncated.astype(bool))).astype(jnp.float32)
    if mode == MODE_KILL_DELTA:
        own0, opp0 = _player_totals(states)
        own1, opp1 = _player_totals(next_states)
        increment = beta * ((opp0 - opp1) - (own0 - own1))
        return terminal_rewards + alive * increment
    if mode == MODE_POTENTIAL or mode == MODE_LAND_POTENTIAL:
        # alive ticks: Phi(s') - Phi(s); terminal ticks: -Phi(s) (Phi(terminal)
        # := 0). The correction is applied OUTSIDE the alive mask by design.
        phi = _land_potential if mode == MODE_LAND_POTENTIAL else _potential
        increment = beta * (phi(next_states) - phi(states))
        # Correction at ANY true episode end (win/loss AND the 1200-turn hard
        # draw): both are genuine terminations, so the shaped return must
        # telescope to the engine return in both cases (Ng et al. 1999).
        episode_end = (terminated.astype(bool) | truncated.astype(bool)).astype(jnp.float32)
        terminal_correction = -beta * phi(states) * episode_end
        return terminal_rewards + alive * increment + terminal_correction
    raise ValueError(f"unknown reward shaping mode: {mode!r}")
