"""Unit tests for the TOPADV top-advantage-fraction mask (Stage-4A knob).

Asserts the default-preserving identity and the masking contract: the
strongest |advantage| transitions keep their exact values, weaker ones are
zeroed, ties at the threshold are included.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from train.competition_native_jax.top_advantage_jax import top_advantage_mask


def test_full_fraction_is_exact_identity():
    adv = jnp.array([0.3, -1.2, 0.0, 0.7, -0.5, 2.0, -0.1, 0.9])
    out = top_advantage_mask(adv, 1.0)
    assert jnp.array_equal(out, adv)


def test_half_fraction_keeps_strongest_magnitudes():
    adv = jnp.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
    out = top_advantage_mask(adv, 0.5)
    # weakest four zeroed; strongest four preserved EXACTLY (same float32 values)
    assert jnp.array_equal(out[:4], jnp.zeros(4))
    assert jnp.array_equal(out[4:], adv[4:])


def test_small_fraction_keeps_at_least_one():
    adv = jnp.array([0.05, -0.02, 3.0, 0.01, -0.9])
    out = top_advantage_mask(adv, 0.01)  # ceil(0.01*5)=1
    assert float(out[2]) == 3.0
    assert jnp.array_equal(jnp.delete(out, 2), jnp.zeros(4))


def test_ties_at_threshold_are_all_included():
    adv = jnp.array([1.0, -1.0, 1.0, 0.2])
    out = top_advantage_mask(adv, 0.5)  # k=2 but three ties at |1.0|
    assert jnp.array_equal(out[:3], adv[:3])
    assert float(out[3]) == 0.0


def test_kept_values_are_exact_not_rescaled():
    adv = jnp.array([2.5, -1.5, 0.01, 0.02])
    out = top_advantage_mask(adv, 0.5)
    assert float(out[0]) == 2.5
    assert float(out[1]) == -1.5


def test_invalid_fraction_rejected():
    adv = jnp.array([1.0, 2.0])
    with pytest.raises(ValueError):
        top_advantage_mask(adv, 0.0)
    with pytest.raises(ValueError):
        top_advantage_mask(adv, 1.5)
    with pytest.raises(ValueError):
        top_advantage_mask(adv, -0.2)
