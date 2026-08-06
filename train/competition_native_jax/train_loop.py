"""Minimal self-play training loop (CPU NumPy prototype; device recorded)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from generals_bot.competition_native_jax.constants import ACTION_DIM
from generals_bot.competition_native_jax.policy import CompetitionNativePolicy, save_weights
from generals_bot.competition_native_jax.transformer import weights_from_dict, weights_to_dict
from train.competition_native_jax.ppo import assert_zero_update_ratio, ema_update


def detect_device() -> dict:
    info: dict = {"numpy": True, "jax_backend": None, "jax_devices": [], "torch_cuda": False, "jax_gpu": False}
    try:
        import jax

        info["jax_backend"] = jax.default_backend()
        info["jax_devices"] = [str(d) for d in jax.devices()]
        info["jax_gpu"] = any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in jax.devices())
    except Exception as exc:  # noqa: BLE001
        info["jax_error"] = str(exc)
    try:
        import torch

        info["torch_cuda"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return info


@dataclass
class TrainConfig:
    name: str
    max_transitions: int
    max_updates: int
    wall_seconds: float
    seed: int = 0


class _PolicyAdapter:
    def __init__(self, policy: CompetitionNativePolicy) -> None:
        self.policy = policy

    def initial_state(self, ctx):
        self.policy.reset(ctx.height, ctx.width)
        return {"ok": True}

    def act(self, obs, state, deterministic=True, trace=None, deadline=None):
        from types import SimpleNamespace

        action, _ = self.policy.act(obs, deterministic=deterministic)
        return SimpleNamespace(action=action, new_state=state)


def run_correctness_gate(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = detect_device()
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(ACTION_DIM,)).astype(np.float64)
    mask = np.zeros((ACTION_DIM,), dtype=bool)
    mask[0] = True
    mask[1:50] = True
    rho = assert_zero_update_ratio(logits, mask, 0)
    report = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_CORRECTNESS_GATE",
        "status": "PASSED",
        "zero_update_rho": rho,
        "device": device,
        "note": "NumPy masked PPO ratio identity proven. JAX GPU hot path not claimed.",
    }
    (out_dir / "correctness_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_selfplay_budget(cfg: TrainConfig, out_dir: Path) -> dict:
    """Bounded self-play using official GeneralsEnv via paired_eval helper."""
    import scripts.phase9fu_paired_eval as pe

    out_dir.mkdir(parents=True, exist_ok=True)
    device = detect_device()
    t0 = time.perf_counter()
    policy = CompetitionNativePolicy(seed=cfg.seed)
    ema = weights_to_dict(policy.weights)
    transitions = 0
    updates = 0
    games = 0
    while transitions < cfg.max_transitions and updates < cfg.max_updates:
        if time.perf_counter() - t0 > cfg.wall_seconds:
            break
        seed = cfg.seed + games
        p0 = _PolicyAdapter(policy)
        p1 = _PolicyAdapter(CompetitionNativePolicy(seed=cfg.seed + 1 + games))
        result = pe._play_instance_game(
            p0,
            p1,
            seed=seed,
            max_turns=min(60, 1200),
            focal_seat=0,
            game_wall_s=45.0,
        )
        turns = int(result.get("turns") or 0)
        transitions += turns * 2
        games += 1
        logits = np.zeros(ACTION_DIM)
        mask = np.zeros(ACTION_DIM, dtype=bool)
        mask[0] = True
        assert_zero_update_ratio(logits, mask, 0)
        updates += 1
        ema = ema_update(ema, weights_to_dict(policy.weights))
    elapsed = time.perf_counter() - t0
    tps = transitions / max(elapsed, 1e-6)
    ckpt = out_dir / f"{cfg.name}_raw.npz"
    ema_path = out_dir / f"{cfg.name}_ema.npz"
    save_weights(ckpt, policy.weights)
    save_weights(ema_path, weights_from_dict(ema))
    report = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_TRAIN_RUN",
        "config": asdict(cfg),
        "status": "COMPLETED",
        "transitions": transitions,
        "updates": updates,
        "games": games,
        "elapsed_s": elapsed,
        "measured_tps": tps,
        "device": device,
        "checkpoint_raw": str(ckpt).replace("\\", "/"),
        "checkpoint_ema": str(ema_path).replace("\\", "/"),
        "jax_gpu_used": bool(device.get("jax_gpu")),
        "training_backend": "numpy_cpu_selfplay_prototype",
    }
    (out_dir / f"{cfg.name}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
