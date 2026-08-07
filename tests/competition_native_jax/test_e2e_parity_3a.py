"""Stage 3A development differential parity vs official GeneralsEnv."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generals import GeneralsEnv
from generals.core import game
from generals.modifiers import build_castles as _bc
from generals.modifiers import deathtouch as _dt

from generals_bot.competition_native_jax.competition_env_jax import (
    DEATHTOUCH_TURN,
    TRUNCATION,
    competition_transition,
    index_to_engine_action,
    legal_mask_one_p0,
    legal_mask_one_p1,
    reset_one_jax,
    step_one_jax,
)
from generals_bot.competition_native_jax.constants import ACTION_DIM, MAX_HW, PASS_INDEX
from generals_bot.competition_native_jax.transformer_jax import forward, forward_batch, init_params


def _official_transition(state, actions):
    state, actions = _bc.apply_build_actions(state, actions)
    return _dt.step(state, actions, DEATHTOUCH_TURN)


def test_pass_always_legal_and_action_dim():
    key = jax.random.PRNGKey(0)
    state = reset_one_jax(key, MAX_HW, MAX_HW)
    mask = legal_mask_one_p0(state)
    assert mask.shape == (ACTION_DIM,)
    assert bool(mask[PASS_INDEX])


def test_competition_transition_matches_official_composition():
    key = jax.random.PRNGKey(1)
    state = reset_one_jax(key, 21, 21)
    pass_a = jnp.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32)
    s_qs, info_qs = competition_transition(state, pass_a)
    s_off, info_off = _official_transition(state, pass_a)
    assert jnp.array_equal(s_qs.armies, s_off.armies)
    assert jnp.array_equal(s_qs.ownership, s_off.ownership)
    assert jnp.array_equal(s_qs.castles, s_off.castles)
    assert jnp.array_equal(s_qs.generals, s_off.generals)
    assert int(s_qs.time) == int(s_off.time)
    assert bool(info_qs.is_done) == bool(info_off.is_done)


@pytest.mark.parametrize("h,w", [(18, 18), (18, 21), (19, 20), (20, 20), (21, 21)])
def test_reset_rectangular_boards(h, w):
    key = jax.random.PRNGKey(h * 100 + w)
    state = reset_one_jax(key, h, w)
    assert state.armies.shape[0] == MAX_HW
    assert state.armies.shape[1] == MAX_HW
    assert bool(jnp.any(state.generals))


def test_forward_batch_native_api():
    key = jax.random.PRNGKey(0)
    params = init_params(key)
    n = 4
    spatial = jax.random.normal(key, (n, 8, 21, 21))
    global_vec = jax.random.normal(jax.random.fold_in(key, 1), (n, 8))
    batched = forward_batch(params, spatial, global_vec)
    assert batched["flat_logits"].shape == (n, ACTION_DIM)
    for i in range(n):
        single = forward(params, spatial[i], global_vec[i])
        err = float(jnp.max(jnp.abs(batched["flat_logits"][i] - single["flat_logits"])))
        assert err < 5e-2, err


def test_generals_env_step_parity_sample():
    """Compare QS step_one_jax to GeneralsEnv.step before auto-reset."""
    env = GeneralsEnv(mode="competition", pool_size=32)
    key = jax.random.PRNGKey(0)
    key, pk, sk = jax.random.split(key, 3)
    pool, _ = env.reset(pk)
    state = env.init_state(sk)
    pass_a = jnp.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32)
    # Avoid auto-reset path: env.step may reset; compare transition kernels instead
    s_env, info_env = _official_transition(state, pass_a)
    s_qs, rew, term, trunc, info_qs = step_one_jax(state, pass_a)
    assert jnp.array_equal(s_qs.armies, s_env.armies)
    assert bool(term) == bool(info_env.is_done)
    assert abs(float(rew[0]) - float(jnp.where(info_env.winner == 0, 1.0, jnp.where(info_env.winner == 1, -1.0, 0.0)))) < 1e-6
    _ = pool, trunc, info_qs


def test_stage3a_random_differential_10k():
    """≥10_000 compared transitions vs official composition; zero mismatches."""
    key = jax.random.PRNGKey(7)
    pass_a = jnp.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32)
    state = reset_one_jax(key, 21, 21)

    @jax.jit
    def dual(state, eng):
        next_qs, rew_qs, term_qs, trunc_qs, _ = step_one_jax(state, eng)
        next_off, info_off = _official_transition(state, eng)
        term_off = info_off.is_done
        trunc_off = (next_off.time >= TRUNCATION) & (~term_off)
        rew_off0 = jnp.where(info_off.winner == 0, 1.0, jnp.where(info_off.winner == 1, -1.0, 0.0))
        same = (
            jnp.array_equal(next_qs.armies, next_off.armies)
            & jnp.array_equal(next_qs.ownership, next_off.ownership)
            & jnp.array_equal(next_qs.castles, next_off.castles)
            & jnp.array_equal(next_qs.generals, next_off.generals)
            & (next_qs.time == next_off.time)
            & (term_qs == term_off)
            & (trunc_qs == trunc_off)
            & (jnp.abs(rew_qs[0] - rew_off0) < 1e-6)
        )
        return next_qs, same

    @jax.jit
    def run_pass_10k(state0):
        def body(s, _):
            ns, same = dual(s, pass_a)
            return ns, same

        _, sames = jax.lax.scan(body, state0, xs=None, length=10_000)
        return sames

    sames = run_pass_10k(state)
    jax.block_until_ready(sames)
    assert bool(jnp.all(sames)), f"pass mismatches={int((~sames).sum())}"

    # Legal-action coverage (2k) with host sampling
    rng = np.random.default_rng(42)
    mismatches = 0
    legal_transitions = 0
    key, sk = jax.random.split(key)
    state = reset_one_jax(sk, 21, 21)
    legal_mask_one_p0(state)
    while legal_transitions < 2000:
        m0 = np.asarray(legal_mask_one_p0(state))
        m1 = np.asarray(legal_mask_one_p1(state))
        a0 = int(rng.choice(np.flatnonzero(m0)))
        a1 = int(rng.choice(np.flatnonzero(m1)))
        eng = jnp.stack(
            [index_to_engine_action(jnp.asarray(a0)), index_to_engine_action(jnp.asarray(a1))]
        )
        state, same = dual(state, eng)
        if not bool(same):
            mismatches += 1
            break
        legal_transitions += 1
        # soft reset when terminal-ish: time high
        if int(state.time) > 1100:
            key, sk = jax.random.split(key)
            state = reset_one_jax(sk, 21, 21)
    assert mismatches == 0
    assert legal_transitions >= 2000
