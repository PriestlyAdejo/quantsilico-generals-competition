"""Ultra-fast Stage-B dataset: self-play only, incremental save, NumPy teacher."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / "quantsilico-runtime" / "emergency_rolling_v1"
OUT = RUNTIME / "distill" / "dataset_stage_b"
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


def _save(sequences, meta_extra: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
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
        "n_states_approx": int(sum(s["valid"].sum() for s in sequences)),
        "bytes": path.stat().st_size,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **meta_extra,
    }
    _atomic_write_json(OUT / "dataset_meta.json", meta)
    _atomic_write_json(ROOT / "experiments/manifests/emergency_distill_dataset.json", meta)
    return path


def main() -> int:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from emergency_load_ema_numpy import load_ema_numpy
    from run_competition_native_jax_daytime_eval import (
        CNJPolicyAdapter,
        observation_from_arrays,
        extract_numpy_boards,
        action_to_jax,
    )
    from generals import GeneralsEnv
    from generals.core import game
    from generals_bot.evaluation.match import make_board, make_transition
    from generals_bot.observation import GameContext
    from generals_bot.policies.base import TraceLevel
    from generals_bot.competition_native_jax.policy import CompetitionNativePolicy
    from generals_bot.competition_native_jax.legal_mask import legal_mask_from_observation
    from generals_bot.competition_native_jax.obs_memory import encode_observation
    from generals_bot.competition_native_jax.transformer import forward
    import jax.numpy as jnp

    print("ULTRA_COLLECT_START", flush=True)
    ckpt_root = RUNTIME / "training" / "checkpoints"
    teacher_dir = max(
        [d for d in ckpt_root.glob("ckpt_*") if (d / "COMPLETE").exists() and (d / "ema.npz").exists()],
        key=lambda d: int(json.loads((d / "meta.json").read_text())["update"]),
    )
    weights = load_ema_numpy(teacher_dir)
    teacher_meta = json.loads((teacher_dir / "meta.json").read_text(encoding="utf-8"))
    print("TEACHER", teacher_dir, "update", teacher_meta.get("update"), flush=True)

    freeze = {
        "schema_version": 1,
        "kind": "DISTILLATION_MIXTURE_FREEZE",
        "mixture": {"self_play": 1.0, "note": "ultra_fast_deadline_path; proportions follow availability"},
        "teacher_checkpoint": str(teacher_dir),
        "teacher_update": teacher_meta.get("update"),
        "teacher_which": "ema",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_distill_mixture_freeze.json", freeze)

    sequences = []
    n_states = 0
    target = 1800
    t0 = time.perf_counter()
    game_i = 0
    max_turns = 60
    while n_states < target and (time.perf_counter() - t0) < 10 * 60:
        seed = 9200 + game_i
        seat = game_i % 2
        cand = CNJPolicyAdapter(weights, seed=seed)
        opp = CNJPolicyAdapter(weights, seed=seed + 7)
        env = GeneralsEnv(mode="competition")
        transition = make_transition(env)
        get_obs = game.get_observation
        state = make_board(env, seed)
        h, w = (int(d) for d in state.armies.shape)
        p0, p1 = (cand, opp) if seat == 0 else (opp, cand)
        st0 = p0.initial_state(GameContext(0, h, w))
        st1 = p1.initial_state(GameContext(1, h, w))
        ep_obs, ep_masks = [], []
        for _ in range(max_turns):
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
            if bool(info.is_done):
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
            logits_list.append(np.where(mask, logits, -1e4).astype(np.float16))
            v_list.append(out["value_logits"].astype(np.float16))
            act_list.append(act)
            mask_list.append(mask.astype(np.bool_))

        T = len(ep_obs)
        start = 0
        while start < T and n_states < target:
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
            start += 24

        game_i += 1
        print(f"games={game_i} states={n_states} seqs={len(sequences)}", flush=True)
        if game_i % 3 == 0:
            _save(
                sequences,
                {
                    "teacher_checkpoint": str(teacher_dir),
                    "teacher_update": teacher_meta.get("update"),
                    "partial": True,
                    "elapsed_s": time.perf_counter() - t0,
                },
            )

    path = _save(
        sequences,
        {
            "teacher_checkpoint": str(teacher_dir),
            "teacher_update": teacher_meta.get("update"),
            "partial": False,
            "elapsed_s": time.perf_counter() - t0,
            "note": "ultra_fast self-play NumPy-annotated Stage B",
        },
    )
    print("DONE", path, "states", n_states, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
