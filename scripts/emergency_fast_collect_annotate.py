"""Fast CPU distill dataset: NumPy EMA teacher + short games (no JAX weight load)."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / "quantsilico-runtime" / "emergency_rolling_v1"
OUT = RUNTIME / "distill" / "dataset_stage_b"
SEQ_LEN = 32
BURN_IN = 8
CACHE_MAX = 1 << 30


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
    best, best_u = None, -1
    for d in ckpt_root.glob("ckpt_*"):
        if (d / "COMPLETE").exists() and (d / "ema.npz").exists() and not d.name.endswith(".tmp"):
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            u = int(meta.get("update", 0))
            if u > best_u:
                best_u, best = u, d
    if best is None:
        raise FileNotFoundError("no teacher")
    return best


def main() -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--target-states", type=int, default=2000)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--wall-minutes", type=float, default=14.0)
    args = ap.parse_args()

    print("FAST_COLLECT_START", flush=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from emergency_load_ema_numpy import load_ema_numpy
    from run_competition_native_jax_daytime_eval import (
        CNJPolicyAdapter,
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

    teacher_dir = _pick_teacher()
    print("TEACHER", teacher_dir, flush=True)
    weights = load_ema_numpy(teacher_dir)
    print("WEIGHTS_LOADED", weights.patch_proj.shape, "layers", len(weights.attn_w), flush=True)
    teacher_meta = json.loads((teacher_dir / "meta.json").read_text(encoding="utf-8"))
    teacher_pol = CompetitionNativePolicy(weights=weights, seed=0)

    freeze = {
        "schema_version": 1,
        "kind": "DISTILLATION_MIXTURE_FREEZE",
        "mixture": {"self_play": 0.6, "hunter": 0.2, "expander_aggressive": 0.15, "fixtures": 0.05},
        "teacher_checkpoint": str(teacher_dir),
        "teacher_update": teacher_meta.get("update"),
        "teacher_which": "ema",
        "sequence_length": SEQ_LEN,
        "burn_in": BURN_IN,
        "target_states": args.target_states,
        "annotation": "numpy_cpu_teacher_logits_float16_legal_renorm",
        "loader": "emergency_load_ema_numpy",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_distill_mixture_freeze.json", freeze)

    OUT.mkdir(parents=True, exist_ok=True)
    sequences = []
    n_states = 0
    t0 = time.perf_counter()
    game_i = 0
    kinds = ["self_play", "self_play", "self_play", "hunter", "expander", "aggressive"]

    while n_states < args.target_states and (time.perf_counter() - t0) < args.wall_minutes * 60:
        kind = kinds[game_i % len(kinds)]
        seed = 9100 + game_i
        seat = game_i % 2
        cand = CNJPolicyAdapter(weights, seed=seed)
        if kind == "hunter":
            opp = create_policy(OPPONENT_ALIASES["official_hunter"], seed=seed)
        elif kind == "expander":
            opp = create_policy(OPPONENT_ALIASES["official_expander"], seed=seed)
        elif kind == "aggressive":
            opp = create_policy(OPPONENT_ALIASES["heuristic_v2_aggressive"], seed=seed)
        else:
            opp = CNJPolicyAdapter(weights, seed=seed + 99)

        env = GeneralsEnv(mode="competition")
        transition = make_transition(env)
        get_obs = game.get_observation
        state = make_board(env, seed)
        h, w = (int(d) for d in state.armies.shape)
        p0, p1 = (cand, opp) if seat == 0 else (opp, cand)
        st0 = p0.initial_state(GameContext(0, h, w))
        st1 = p1.initial_state(GameContext(1, h, w))
        ep_obs, ep_masks, ep_terminals = [], [], []
        for turn_i in range(args.max_turns):
            eng0 = get_obs(state, 0)
            eng1 = get_obs(state, 1)
            t0a, o0, a0, _, m0 = extract_numpy_boards(eng0, h, w)
            t1, o1, a1, _, m1 = extract_numpy_boards(eng1, h, w)
            obs0 = observation_from_arrays(t0a, o0, a0, m0)
            obs1 = observation_from_arrays(t1, o1, a1, m1)
            focal = obs0 if seat == 0 else obs1
            ep_obs.append(focal)
            ep_masks.append(legal_mask_from_observation(focal))
            d0 = p0.act(obs0, st0, deterministic=True, trace=TraceLevel.NONE, deadline=None)
            st0 = d0.new_state
            d1 = p1.act(obs1, st1, deterministic=True, trace=TraceLevel.NONE, deadline=None)
            st1 = d1.new_state
            state, info = transition(state, jnp.stack([action_to_jax(d0.action), action_to_jax(d1.action)]))
            done = bool(info.is_done)
            ep_terminals.append(done)
            if done:
                break

        annot = CompetitionNativePolicy(weights=weights, seed=0)
        annot.reset(h, w)
        spat_list, glob_list, logits_list, v_list, act_list, mask_list = [], [], [], [], [], []
        for i, obs in enumerate(ep_obs):
            spatial, g = encode_observation(obs, annot.memory)
            out = forward(spatial, g, annot.weights)
            mask = ep_masks[i]
            logits = out["flat_logits"].astype(np.float32)
            act = int(np.argmax(np.where(mask, logits, -1e9)))
            spat_list.append(spatial.astype(np.float16))
            glob_list.append(g.astype(np.float16))
            store = np.where(mask, logits, -1e4).astype(np.float16)
            logits_list.append(store)
            v_list.append(out["value_logits"].astype(np.float16))
            act_list.append(act)
            mask_list.append(mask.astype(np.bool_))

        T = len(ep_obs)
        start = 0
        step = 24
        while start < T and n_states < args.target_states:
            frag_start = max(0, start - BURN_IN)
            idxs = list(range(frag_start, min(T, frag_start + SEQ_LEN)))
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
                }
            )
            n_states += int(valid.sum())
            start += step

        game_i += 1
        if game_i % 2 == 0:
            print(f"collect games={game_i} states={n_states} seqs={len(sequences)}", flush=True)

    path = OUT / "sequences.npz"
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
        "note": "CPU NumPy teacher annotation via emergency_load_ema_numpy; Stage B real targets.",
        "elapsed_s": time.perf_counter() - t0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "cache_ok": path.stat().st_size <= CACHE_MAX,
    }
    _atomic_write_json(OUT / "dataset_meta.json", meta)
    _atomic_write_json(ROOT / "experiments/manifests/emergency_distill_dataset.json", meta)
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
