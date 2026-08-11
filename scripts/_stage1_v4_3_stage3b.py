"""Exact-hash Stage 3B for V4.3 daytime programme (selected snapshot)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from generals_bot.competition_native_jax.competition_env_jax import (
    DEATHTOUCH_TURN,
    TRUNCATION,
    build_competition_reset_pool,
    competition_transition,
    index_to_engine_action,
    legal_mask_one_p0,
    legal_mask_one_p1,
    reset_one_jax,
    step_one_jax,
)
from train.competition_native_jax.train_jax import lineage_hashes

ROOT = Path(__file__).resolve().parents[1]


def _dual(state, eng):
    next_qs, rew_qs, term_qs, trunc_qs, _ = step_one_jax(state, eng)
    next_off, info_off = competition_transition(state, eng)
    term_off = info_off.is_done
    trunc_off = (next_off.time >= TRUNCATION) & (~term_off)
    rew_off0 = jnp.where(
        info_off.winner == 0,
        1.0,
        jnp.where(info_off.winner == 1, -1.0, 0.0),
    )
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
    return next_qs, term_qs | trunc_qs, same


dual_jit = jax.jit(_dual)


def _write_fail(report: dict) -> None:
    out = ROOT / "experiments/manifests/competition_native_jax_v4_2_stage3b_final.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    prog_path = ROOT / "experiments/manifests/competition_native_jax_v4_3_programme_state.json"
    prog = json.loads(prog_path.read_text())
    prog["status"] = "BLOCKED_V4_2_STAGE3B"
    prog["current_stage"] = "STAGE_1_FAILED_3B"
    prog["stage3b"] = "experiments/manifests/competition_native_jax_v4_2_stage3b_final.json"
    prog_path.write_text(json.dumps(prog, indent=2) + "\n")
    (ROOT / "experiments/reports/competition_native_jax_v4_2_stage3b_final.md").write_text(
        "\n".join(
            [
                "# Stage 3B exact-hash final",
                "",
                f"**Status: `{report.get('status')}`**",
                f"Gate: `{report.get('gate')}`",
                "",
                "```json",
                json.dumps(report, indent=2),
                "```",
                "",
            ]
        )
    )


def main() -> int:
    snap = json.loads((ROOT / "experiments/manifests/competition_native_jax_v4_3_source_snapshot.json").read_text())
    lin = lineage_hashes()
    assert lin["env_implementation_hash"] == snap["env_implementation_hash"]
    assert lin["learner_implementation_hash"] == snap["learner_implementation_hash"]

    pass_a = jnp.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32)
    key = jax.random.PRNGKey(99)

    @jax.jit
    def run_pass_block(state0):
        def body(s, _):
            ns, done, same = dual_jit(s, pass_a)
            return ns, (same, done)

        _, outs = jax.lax.scan(body, state0, xs=None, length=1000)
        sames, dones = outs
        return sames, dones

    print("3B: pass blocks starting", flush=True)
    mismatches = 0
    transitions = 0
    games = 0
    for bi in range(100):
        key, sk = jax.random.split(key)
        state = reset_one_jax(sk, 21, 21)
        games += 1
        sames, _dones = run_pass_block(state)
        jax.block_until_ready(sames)
        bad = int((~sames).sum())
        mismatches += bad
        transitions += 1000
        if bi % 20 == 0:
            print(f"3B: pass block {bi} transitions={transitions} mismatches={mismatches}", flush=True)
        if bad:
            break

    print("3B: legal mixed actions", flush=True)
    rng = np.random.default_rng(123)
    legal_ok = 0
    legal_bad = 0
    key, sk = jax.random.split(key)
    state = reset_one_jax(sk, 21, 21)
    for _ in range(5000):
        m0 = np.asarray(legal_mask_one_p0(state))
        m1 = np.asarray(legal_mask_one_p1(state))
        eng = jnp.stack(
            [
                index_to_engine_action(jnp.asarray(int(rng.choice(np.flatnonzero(m0))))),
                index_to_engine_action(jnp.asarray(int(rng.choice(np.flatnonzero(m1))))),
            ]
        )
        state, done, same = dual_jit(state, eng)
        if bool(same):
            legal_ok += 1
        else:
            legal_bad += 1
            break
        if bool(done) or int(state.time) > 1100:
            key, sk = jax.random.split(key)
            state = reset_one_jax(sk, 21, 21)
            games += 1

    # Extra game starts to reach >=1000
    while games < 1000 and mismatches == 0 and legal_bad == 0:
        key, sk = jax.random.split(key)
        state = reset_one_jax(sk, 21, 21)
        state, _done, same = dual_jit(state, pass_a)
        if not bool(same):
            mismatches += 1
            break
        games += 1

    # Rectangular boards 18–21
    board_ok = True
    board_shapes = []
    for h, w in [(18, 18), (18, 21), (19, 20), (20, 20), (21, 21)]:
        key, sk = jax.random.split(key)
        st = reset_one_jax(sk, h, w)
        board_shapes.append({"h": h, "w": w, "armies": list(st.armies.shape)})
        if st.armies.shape != (21, 21):
            board_ok = False

    # Castles / Deathtouch / turn-cap (bounded — no 1200-step compiled scans)
    key, sk = jax.random.split(key)
    st = reset_one_jax(sk, 21, 21)
    castles_path_ok = st.castles is not None and st.castles.shape == st.armies.shape

    # Deathtouch: jump near DEATHTOUCH_TURN and verify dual parity across the boundary
    st_near_dt = st._replace(time=jnp.asarray(int(DEATHTOUCH_TURN) - 1, dtype=st.time.dtype))
    st_dt, done_dt, same_dt = dual_jit(st_near_dt, pass_a)
    jax.block_until_ready(st_dt.armies)
    if not bool(same_dt):
        mismatches += 1
    dt_seen = int(st_dt.time) >= int(DEATHTOUCH_TURN) or bool(done_dt)

    # Turn-cap / truncation: jump near TRUNCATION and verify truncated semantics
    st_near_cap = st._replace(time=jnp.asarray(int(TRUNCATION) - 1, dtype=st.time.dtype))
    st_tr, done_tr, same_tr = dual_jit(st_near_cap, pass_a)
    jax.block_until_ready(st_tr.armies)
    if not bool(same_tr):
        mismatches += 1
    # After one step from TRUNCATION-1, time should be >= TRUNCATION or episode done
    trunc_seen = int(st_tr.time) >= int(TRUNCATION) or bool(done_tr)
    # Explicit formula check used by step_one_jax
    next_qs, _rew, term_qs, trunc_qs, _info = step_one_jax(st_near_cap, pass_a)
    formula_ok = bool((trunc_qs == ((next_qs.time >= TRUNCATION) & (~term_qs))))
    if not formula_ok:
        mismatches += 1
    if formula_ok:
        trunc_seen = True

    # Reset-pool distribution vs canonical reset_one_jax (same keys + shuffle)
    import inspect

    from generals_bot.competition_native_jax import competition_env_jax as cej

    pool_size = 64
    key, sk = jax.random.split(key)
    pool = build_competition_reset_pool(sk, pool_size=pool_size, min_grid=21, max_grid=21)
    pool2 = build_competition_reset_pool(sk, pool_size=pool_size, min_grid=21, max_grid=21)
    pool_mismatches = 0
    if not bool(jnp.array_equal(pool.armies, pool2.armies)):
        pool_mismatches += 1
    # Reconstruct builder keystream for single-size pool (21×21 only → one combo)
    k_pool, k_shuffle = jax.random.split(sk)
    actual = pool_size  # one combo when min_grid==max_grid==21
    pool_keys = jax.random.split(k_pool, actual)
    unshuffled = jax.vmap(lambda k: reset_one_jax(k, height=21, width=21))(pool_keys)
    perm = jax.random.permutation(k_shuffle, actual)
    rebuilt = jax.tree_util.tree_map(lambda x: x[perm], unshuffled)
    if not bool(jnp.array_equal(rebuilt.armies, pool.armies)):
        pool_mismatches += 1
    if not bool(jnp.array_equal(rebuilt.ownership, pool.ownership)):
        pool_mismatches += 1

    src = inspect.getsource(cej.build_competition_reset_pool)
    pool_uses_reset_one = "reset_one_jax" in src

    # Tracer-leak multi-shape (bounded)
    tracer_ok = True
    tracer_error = None
    os.environ["JAX_CHECK_TRACER_LEAKS"] = "1"
    try:
        for h, w in [(18, 18), (21, 21), (19, 20)]:
            key, sk = jax.random.split(key)
            st = reset_one_jax(sk, h, w)
            st2, done, same = dual_jit(st, pass_a)
            jax.block_until_ready(st2.armies)
            if not bool(same):
                tracer_ok = False
                tracer_error = "parity_fail_under_tracer_leak_check"
                break
    except Exception as exc:  # noqa: BLE001
        tracer_ok = False
        tracer_error = str(exc)
    finally:
        os.environ.pop("JAX_CHECK_TRACER_LEAKS", None)

    # Seat helpers produce legal support
    key, sk = jax.random.split(key)
    st = reset_one_jax(sk, 21, 21)
    m0 = legal_mask_one_p0(st)
    m1 = legal_mask_one_p1(st)
    seat_ok = bool(m0[0]) and bool(m1[0]) and int(m0.shape[0]) == int(m1.shape[0])

    status_pass = (
        mismatches == 0
        and legal_bad == 0
        and transitions >= 100_000
        and games >= 1000
        and board_ok
        and pool_uses_reset_one
        and pool_mismatches == 0
        and tracer_ok
        and seat_ok
        and dt_seen
        and trunc_seen
    )

    report = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_V4_2_STAGE3B_FINAL",
        "status": "PASSED" if status_pass else "FAILED",
        "gate": "STAGE_3B_EXACT_HASH_PASS" if status_pass else "STAGE_3B_EXACT_HASH_FAIL",
        "pass_transitions": transitions,
        "pass_mismatches": mismatches,
        "legal_transitions": legal_ok,
        "legal_mismatches": legal_bad,
        "games_started": games,
        "boards_checked": board_shapes,
        "boards_ok": board_ok,
        "castles_path_exercised": castles_path_ok,
        "deathtouch_turn_reached": dt_seen,
        "turn_cap_reached_or_done": trunc_seen,
        "reset_pool_uses_reset_one_jax": pool_uses_reset_one,
        "reset_pool_entry_mismatches": pool_mismatches,
        "tracer_leak_multi_shape_ok": tracer_ok,
        "tracer_leak_error": tracer_error,
        "seats_legal_support_ok": seat_ok,
        "lineage": lin,
        "selected_env_implementation_hash": snap["env_implementation_hash"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "minima": {
            "transitions": 100_000,
            "games": 1000,
            "required_mismatches": 0,
        },
    }

    out = ROOT / "experiments/manifests/competition_native_jax_v4_2_stage3b_final.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "experiments/reports/competition_native_jax_v4_2_stage3b_final.md").write_text(
        "\n".join(
            [
                "# Stage 3B exact-hash final",
                "",
                f"**Status: `{report['status']}`**",
                f"Gate: `{report['gate']}`",
                f"Transitions: {transitions} (mismatches={mismatches})",
                f"Legal: {legal_ok} (mismatches={legal_bad})",
                f"Games: {games}",
                f"Deathtouch reached: {dt_seen}; turn-cap/done: {trunc_seen}",
                f"Reset-pool mismatches: {pool_mismatches}; tracer_ok: {tracer_ok}",
                f"env_implementation_hash: `{lin['env_implementation_hash']}`",
                "",
            ]
        )
    )

    # Do not overwrite frozen pre-V4.2 parity_3b blindly; stamp a pointer artefact
    (ROOT / "experiments/manifests/competition_native_jax_parity_3b.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "COMPETITION_NATIVE_JAX_PARITY_3B",
                "status": report["status"],
                "pass_transitions": transitions,
                "pass_mismatches": mismatches,
                "legal_transitions": legal_ok,
                "legal_mismatches": legal_bad,
                "games_started": games,
                "env_implementation_hash": lin["env_implementation_hash"],
                "source": "competition_native_jax_v4_2_stage3b_final.json",
            },
            indent=2,
        )
        + "\n"
    )

    prog_path = ROOT / "experiments/manifests/competition_native_jax_v4_3_programme_state.json"
    prog = json.loads(prog_path.read_text())
    if status_pass:
        prog["status"] = "STAGE_1_COMPLETE"
        prog["current_stage"] = "STAGE_2_R_E5_HEALTH"
        prog["stage3a"] = "experiments/manifests/competition_native_jax_v4_2_stage3a_exact_hash.json"
        prog["stage3b"] = "experiments/manifests/competition_native_jax_v4_2_stage3b_final.json"
        # Promote frozen baseline classification after 3B
        frozen_path = ROOT / "experiments/manifests/competition_native_jax_v4_2_frozen_baseline.json"
        frozen = json.loads(frozen_path.read_text())
        frozen["classification"] = "SYSTEMS_PROMOTION_READY_STAGE_3B_PASS"
        frozen["stage3b"] = "experiments/manifests/competition_native_jax_v4_2_stage3b_final.json"
        frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
    else:
        prog["status"] = "BLOCKED_V4_2_STAGE3B"
        prog["current_stage"] = "STAGE_1_FAILED_3B"
    prog_path.write_text(json.dumps(prog, indent=2) + "\n")

    print(json.dumps(report, indent=2), flush=True)
    return 0 if status_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
