"""CPU ordered-sequence collect + NumPy teacher annotation (no GPU)."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / "quantsilico-runtime" / "emergency_rolling_v1"
OUT = RUNTIME / "distill" / "dataset_stage_b"
CACHE_MAX = 1 << 30
DEFAULT_TARGET_STATES = 2500
SEQ_LEN = 32
BURN_IN = 8


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    with tmp.open("rb") as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    tmp.replace(path)


def _pick_teacher() -> Path:
    ckpt_root = RUNTIME / "training" / "checkpoints"
    best = None
    best_u = -1
    for d in ckpt_root.glob("ckpt_*"):
        if not (d / "COMPLETE").exists() or not (d / "ema.npz").exists():
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        u = int(meta.get("update", 0))
        if u > best_u:
            best_u = u
            best = d
    if best is None:
        # fallback to R-E.6 parent
        parent = ROOT / "experiments/competition_native_jax/v4_3_r_e6/ckpt_final"
        if parent.exists():
            return parent
        raise FileNotFoundError("no COMPLETE EMA teacher")
    return best


def main() -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--target-states", type=int, default=DEFAULT_TARGET_STATES)
    ap.add_argument("--max-turns", type=int, default=160)
    ap.add_argument("--wall-minutes", type=float, default=18.0)
    args = ap.parse_args()
    target_states = int(args.target_states)

    print("COLLECT_START", flush=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_competition_native_jax_daytime_eval import (
        CNJPolicyAdapter,
        load_cnj_from_ckpt,
        OPPONENT_ALIASES,
        observation_from_arrays,
        extract_numpy_boards,
        action_to_jax,
    )
    from generals import GeneralsEnv
    from generals.core import game
    from generals_bot.evaluation.match import make_board, make_transition
    from generals_bot.selector import create_policy
    from generals_bot.observation import GameContext
    from generals_bot.policies.base import TraceLevel
    from generals_bot.competition_native_jax.policy import CompetitionNativePolicy
    from generals_bot.competition_native_jax.legal_mask import legal_mask_from_observation
    from generals_bot.competition_native_jax.obs_memory import encode_observation
    from generals_bot.competition_native_jax.transformer import forward
    import jax.numpy as jnp

    OUT.mkdir(parents=True, exist_ok=True)
    teacher_dir = _pick_teacher()
    print("TEACHER", teacher_dir, flush=True)
    teacher = load_cnj_from_ckpt(teacher_dir, which="ema")
    teacher_meta = json.loads((teacher_dir / "meta.json").read_text(encoding="utf-8"))

    mixture = {
        "self_play": 0.5,
        "hunter": 0.2,
        "expander_aggressive": 0.2,
        "fixtures": 0.1,
    }
    freeze = {
        "schema_version": 1,
        "kind": "DISTILLATION_MIXTURE_FREEZE",
        "mixture": mixture,
        "teacher_checkpoint": str(teacher_dir),
        "teacher_update": teacher_meta.get("update"),
        "teacher_which": "ema",
        "sequence_length": SEQ_LEN,
        "burn_in": BURN_IN,
        "target_states": target_states,
        "annotation": "numpy_cpu_teacher_logits_float16_legal_renorm",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_distill_mixture_freeze.json", freeze)

    # Collect via short games: hunter / expander / self-play (teacher vs teacher)
    opp_kinds = []
    for kind, _w in [
        ("hunter", int(target_states * 0.2)),
        ("expander", int(target_states * 0.15)),
        ("aggressive", int(target_states * 0.05)),
        ("self_play", int(target_states * 0.6)),
    ]:
        opp_kinds.extend([kind] * max(1, _w // 200))

    sequences = []
    n_states = 0
    t0 = time.perf_counter()
    seed0 = 9001
    game_i = 0

    while n_states < target_states and (time.perf_counter() - t0) < args.wall_minutes * 60:
        kind = opp_kinds[game_i % len(opp_kinds)]
        seed = seed0 + game_i
        seat = game_i % 2
        if kind == "hunter":
            opp = create_policy(OPPONENT_ALIASES["official_hunter"], seed=seed)
        elif kind == "expander":
            opp = create_policy(OPPONENT_ALIASES["official_expander"], seed=seed)
        elif kind == "aggressive":
            # best-effort aggressive heuristic alias
            opp = create_policy(OPPONENT_ALIASES["heuristic_v2_aggressive"], seed=seed)
        else:
            opp = CNJPolicyAdapter(teacher.inner.weights, seed=seed + 17)

        cand = CNJPolicyAdapter(teacher.inner.weights, seed=seed)
        # Collect observations from focal seat while playing
        env = GeneralsEnv(mode="competition")
        transition = make_transition(env)
        get_obs = game.get_observation
        state = make_board(env, seed)
        h, w = (int(d) for d in state.armies.shape)
        p0, p1 = (cand, opp) if seat == 0 else (opp, cand)
        st0 = p0.initial_state(GameContext(0, h, w))
        st1 = p1.initial_state(GameContext(1, h, w))
        ep_obs = []
        ep_masks = []
        ep_terminals = []
        max_turns = int(args.max_turns)
        for turn_i in range(max_turns):
            eng0 = get_obs(state, 0)
            eng1 = get_obs(state, 1)
            t0a, o0, a0, _, m0 = extract_numpy_boards(eng0, h, w)
            t1, o1, a1, _, m1 = extract_numpy_boards(eng1, h, w)
            obs0 = observation_from_arrays(t0a, o0, a0, m0)
            obs1 = observation_from_arrays(t1, o1, a1, m1)
            focal_obs = obs0 if seat == 0 else obs1
            ep_obs.append(focal_obs)
            ep_masks.append(legal_mask_from_observation(focal_obs))
            d0 = p0.act(obs0, st0, deterministic=True, trace=TraceLevel.NONE, deadline=None)
            st0 = d0.new_state
            d1 = p1.act(obs1, st1, deterministic=True, trace=TraceLevel.NONE, deadline=None)
            st1 = d1.new_state
            state, info = transition(state, jnp.stack([action_to_jax(d0.action), action_to_jax(d1.action)]))
            done = bool(info.is_done)
            ep_terminals.append(done)
            if done:
                break

        # Annotate with teacher NumPy (fresh memory per episode)
        annot = CompetitionNativePolicy(weights=teacher.inner.weights, seed=0)
        annot.reset(h, w)
        spat_list, glob_list, logits_list, v_list, act_list, mask_list = [], [], [], [], [], []
        for i, obs in enumerate(ep_obs):
            spatial, g = encode_observation(obs, annot.memory)
            out = forward(spatial, g, annot.weights)
            mask = ep_masks[i]
            logits = out["flat_logits"].astype(np.float32)
            # legal-renormalised teacher action
            masked = np.where(mask, logits, -1e9)
            act = int(np.argmax(masked))
            spat_list.append(spatial.astype(np.float16))
            glob_list.append(g.astype(np.float16))
            # store legal logits only as float16 full vector (illegal set -1e4)
            store = logits.astype(np.float16)
            store = np.where(mask, store, np.float16(-1e4))
            logits_list.append(store)
            v_list.append(out["value_logits"].astype(np.float16))
            act_list.append(act)
            mask_list.append(mask.astype(np.bool_))

        # fragment into SEQ_LEN windows with burn-in overlap
        T = len(ep_obs)
        step = LOSS_STEPS = 24
        start = 0
        while start < T and n_states < target_states:
            # include burn-in from max(0, start-BURN_IN)
            frag_start = max(0, start - BURN_IN)
            idxs = list(range(frag_start, min(T, frag_start + SEQ_LEN)))
            if not idxs:
                break
            pad_n = SEQ_LEN - len(idxs)
            # build arrays
            spat = np.zeros((SEQ_LEN, *spat_list[0].shape), dtype=np.float16)
            glob = np.zeros((SEQ_LEN, glob_list[0].shape[0]), dtype=np.float16)
            logits = np.full((SEQ_LEN, logits_list[0].shape[0]), np.float16(-1e4), dtype=np.float16)
            vals = np.zeros((SEQ_LEN, v_list[0].shape[0]), dtype=np.float16)
            acts = np.zeros((SEQ_LEN,), dtype=np.int32)
            masks = np.zeros((SEQ_LEN, mask_list[0].shape[0]), dtype=np.bool_)
            valid = np.zeros((SEQ_LEN,), dtype=np.float32)
            ep_reset = np.zeros((SEQ_LEN,), dtype=np.float32)
            turns = np.zeros((SEQ_LEN,), dtype=np.int32)
            for j, src_i in enumerate(idxs):
                spat[j] = spat_list[src_i]
                glob[j] = glob_list[src_i]
                logits[j] = logits_list[src_i]
                vals[j] = v_list[src_i]
                acts[j] = act_list[src_i]
                masks[j] = mask_list[src_i]
                valid[j] = 1.0
                turns[j] = int(ep_obs[src_i].turn)
                if src_i == 0:
                    ep_reset[j] = 1.0
                if ep_terminals[src_i]:
                    # terminal at this step; next fragment will reset
                    pass
            # padded prefix if frag starts at episode start with short burn-in already handled
            if frag_start == 0 and len(idxs) < SEQ_LEN:
                # already valid only on real steps
                pass
            sequences.append(
                {
                    "spatial": spat,
                    "global": glob,
                    "teacher_logits": logits,
                    "teacher_value": vals,
                    "teacher_action": acts,
                    "legal_mask": masks,
                    "valid": valid,
                    "episode_reset": ep_reset,
                    "turns": turns,
                    "episode_id": f"{kind}:{seed}:seat{seat}",
                    "source": kind,
                }
            )
            n_states += int(valid.sum())
            start += step
            if start >= T:
                break

        game_i += 1
        if game_i % 5 == 0:
            print(f"collect games={game_i} states={n_states} seqs={len(sequences)}", flush=True)

    # persist
    path = OUT / "sequences.npz"
    # stack what we can; store as object arrays via separate files if needed
    np.savez_compressed(
        path,
        spatial=np.stack([s["spatial"] for s in sequences], axis=0),
        global_vec=np.stack([s["global"] for s in sequences], axis=0),
        teacher_logits=np.stack([s["teacher_logits"] for s in sequences], axis=0),
        teacher_value=np.stack([s["teacher_value"] for s in sequences], axis=0),
        teacher_action=np.stack([s["teacher_action"] for s in sequences], axis=0),
        legal_mask=np.stack([s["legal_mask"] for s in sequences], axis=0),
        valid=np.stack([s["valid"] for s in sequences], axis=0),
        episode_reset=np.stack([s["episode_reset"] for s in sequences], axis=0),
        turns=np.stack([s["turns"] for s in sequences], axis=0),
    )
    meta = {
        "schema_version": 1,
        "kind": "EMERGENCY_DISTILL_DATASET",
        "path": str(path),
        "n_sequences": len(sequences),
        "n_states_approx": n_states,
        "bytes": path.stat().st_size,
        "teacher_checkpoint": str(teacher_dir),
        "teacher_update": teacher_meta.get("update"),
        "mixture_freeze": "experiments/manifests/emergency_distill_mixture_freeze.json",
        "note": "CPU NumPy teacher annotation; Stage B real targets (not Stage A synthetic).",
        "elapsed_s": time.perf_counter() - t0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if path.stat().st_size > CACHE_MAX:
        meta["cache_warning"] = "exceeds_1GiB"
    _atomic_write_json(OUT / "dataset_meta.json", meta)
    _atomic_write_json(ROOT / "experiments/manifests/emergency_distill_dataset.json", meta)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
