"""NumPy PPO / GAE / HL-Gauss helpers for competition-native training."""

from __future__ import annotations

import numpy as np

from generals_bot.competition_native_jax.constants import (
    EMA_TAU,
    HL_GAUSS_BINS,
    HL_GAUSS_MAX,
    HL_GAUSS_MIN,
    HL_GAUSS_SIGMA,
)
from generals_bot.competition_native_jax.legal_mask import masked_log_softmax


def ema_update(ema: dict[str, np.ndarray], current: dict[str, np.ndarray], tau: float = EMA_TAU) -> dict[str, np.ndarray]:
    out = {}
    for k, v in current.items():
        if k not in ema:
            out[k] = v.copy()
        else:
            out[k] = tau * ema[k] + (1.0 - tau) * v
    return out


def gae_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float = 1.0,
    lam: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """rewards/values/dones length T; values length T+1 bootstrap."""
    t = len(rewards)
    adv = np.zeros(t, dtype=np.float64)
    last = 0.0
    for i in reversed(range(t)):
        next_nonterminal = 1.0 - float(dones[i])
        delta = rewards[i] + gamma * values[i + 1] * next_nonterminal - values[i]
        last = delta + gamma * lam * next_nonterminal * last
        adv[i] = last
    returns = adv + values[:-1]
    return adv.astype(np.float32), returns.astype(np.float32)


def hl_gauss_target(return_scalar: float) -> np.ndarray:
    centers = np.linspace(HL_GAUSS_MIN, HL_GAUSS_MAX, HL_GAUSS_BINS, dtype=np.float64)
    # approximate bin width
    z = np.exp(-0.5 * ((centers - return_scalar) / HL_GAUSS_SIGMA) ** 2)
    z = z / z.sum()
    return z.astype(np.float32)


def policy_ratio(logp_new: float, logp_old: float) -> float:
    return float(np.exp(logp_new - logp_old))


def assert_zero_update_ratio(
    logits: np.ndarray,
    mask: np.ndarray,
    action_index: int,
    *,
    atol: float = 1e-5,
) -> float:
    """With identical logits for old/new, ρ must be ≈ 1."""
    logp = masked_log_softmax(logits, mask)
    logp_a = float(logp[action_index])
    rho = policy_ratio(logp_a, logp_a)
    if abs(rho - 1.0) > atol:
        raise AssertionError(f"zero-update ratio {rho} != 1")
    return rho


def clipped_ppo_loss(
    logp_new: np.ndarray,
    logp_old: np.ndarray,
    advantages: np.ndarray,
    *,
    clip: float = 0.2,
) -> float:
    ratio = np.exp(logp_new - logp_old)
    unclipped = ratio * advantages
    clipped = np.clip(ratio, 1.0 - clip, 1.0 + clip) * advantages
    return float(-np.mean(np.minimum(unclipped, clipped)))
