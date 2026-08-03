"""JAX environment ↔ PyTorch policy bridge throughput benchmark.

Profiles environment, extraction, encoding, transfers, and forward stages
separately. Supports warm-up vs steady-state and multiple batch sizes.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch

from generals import GeneralsEnv
from generals.core import game

from generals_bot.evaluation.match import make_board, make_transition
from generals_bot.models.action_index import ACTION_DIM
from generals_bot.models.legal_mask import apply_action_mask
from generals_bot.models.mlp import RecurrentMLPPolicy
from generals_bot.models.observation_encoder import (
    GLOBAL_DIM,
    encode_grids_batch_numpy,
)
from generals_bot.observation import Observation
from generals_bot.protocol import (
    TYPE_CASTLE,
    TYPE_FOG,
    TYPE_GENERAL,
    TYPE_MOUNTAIN,
    TYPE_PLAIN,
    TYPE_STRUCTURE_IN_FOG,
)

STAGE_KEYS = (
    "env_reset_ms",
    "env_step_ms",
    "obs_extract_ms",
    "python_object_ms",
    "feature_channel_ms",
    "padding_mask_ms",
    "legal_enum_ms",
    "legal_mask_tensor_ms",
    "jax_to_numpy_ms",
    "numpy_to_torch_ms",
    "cpu_preprocess_ms",
    "h2d_ms",
    "mlp_forward_ms",
    "mask_apply_ms",
    "action_select_ms",
    "d2h_ms",
    "action_to_jax_ms",
    "e2e_iter_ms",
)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def _summarise(values: list[float], total_ms: float) -> dict[str, float]:
    s = float(sum(values))
    return {
        "mean": float(np.mean(values)) if values else 0.0,
        "median": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "sum_ms": s,
        "pct_of_total": 100.0 * s / max(total_ms, 1e-9),
    }


def extract_numpy_boards(
    engine_obs, height: int, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """JAX/engine arrays → NumPy type/owner/army + globals (no Python nested tuples)."""
    t0 = time.perf_counter()
    armies = np.asarray(engine_obs.armies)
    fog = np.asarray(engine_obs.fog_cells, dtype=bool)
    mountains = np.asarray(engine_obs.mountains, dtype=bool)
    castles = np.asarray(engine_obs.castles, dtype=bool)
    generals = np.asarray(engine_obs.generals, dtype=bool)
    structures_fog = np.asarray(engine_obs.structures_in_fog, dtype=bool)
    owned = np.asarray(engine_obs.owned_cells, dtype=bool)
    opp = np.asarray(engine_obs.opponent_cells, dtype=bool)
    jax_to_numpy_ms = (time.perf_counter() - t0) * 1000

    type_grid = np.full((height, width), TYPE_PLAIN, dtype=np.int32)
    type_grid[fog] = TYPE_FOG
    type_grid[structures_fog] = TYPE_STRUCTURE_IN_FOG
    type_grid[mountains] = TYPE_MOUNTAIN
    type_grid[castles] = TYPE_CASTLE
    type_grid[generals] = TYPE_GENERAL
    owner = np.zeros((height, width), dtype=np.int32)
    owner[owned] = 1
    owner[opp] = 2

    turn = int(engine_obs.timestep)
    my_land = int(engine_obs.owned_land_count)
    my_army = int(engine_obs.owned_army_count)
    opp_land = int(engine_obs.opponent_land_count)
    opp_army = int(engine_obs.opponent_army_count)
    turn_frac = turn / 1200.0
    globals_ = np.asarray(
        [
            turn_frac,
            float(turn % 2),
            float(turn % 50) / 50.0,
            my_land / 500.0,
            my_army / 5000.0,
            opp_land / 500.0,
            opp_army / 5000.0,
            (my_land - opp_land) / 500.0,
            (my_army - opp_army) / 5000.0,
        ],
        dtype=np.float32,
    )
    meta = {
        "jax_to_numpy_ms": jax_to_numpy_ms,
        "turn": turn,
        "my_land": my_land,
        "my_army": my_army,
        "opp_land": opp_land,
        "opp_army": opp_army,
    }
    return type_grid, owner, armies.astype(np.int32, copy=False), globals_, meta  # type: ignore[return-value]


def _obs_from_arrays(
    type_grid: np.ndarray,
    owner: np.ndarray,
    armies: np.ndarray,
    meta: dict[str, Any],
) -> Observation:
    return Observation(
        height=type_grid.shape[0],
        width=type_grid.shape[1],
        turn=int(meta["turn"]),
        my_land=int(meta["my_land"]),
        my_army=int(meta["my_army"]),
        opp_land=int(meta["opp_land"]),
        opp_army=int(meta["opp_army"]),
        type_grid=tuple(tuple(int(x) for x in row) for row in type_grid),
        owner_grid=tuple(tuple(int(x) for x in row) for row in owner),
        army_grid=tuple(tuple(int(x) for x in row) for row in armies),
    )


def _resource_snapshot(device: torch.device) -> dict[str, float | None]:
    snap: dict[str, float | None] = {
        "cpu_util_pct": None,
        "gpu_util_pct": None,
        "ram_gb": None,
        "vram_gb": None,
    }
    try:
        import psutil

        snap["cpu_util_pct"] = float(psutil.cpu_percent(interval=None))
        snap["ram_gb"] = float(psutil.virtual_memory().used) / (1024**3)
    except Exception:
        pass
    if device.type == "cuda" and torch.cuda.is_available():
        snap["vram_gb"] = float(torch.cuda.memory_allocated(device)) / (1024**3)
    return snap


def run_bridge_benchmark(
    *,
    steps: int = 200,
    warmup: int = 20,
    batch_sizes: list[int] | None = None,
    device: str | None = None,
    seed: int = 0,
    profile_python_object: bool = True,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64, 128]

    env = GeneralsEnv(mode="competition")
    state = make_board(env, seed)
    transition = make_transition(env)
    get_obs = game.get_observation
    h, w = (int(d) for d in state.armies.shape)
    policy = RecurrentMLPPolicy().to(torch_device)
    policy.eval()

    # Filter batch sizes by rough VRAM/RAM budget (cells ~ B*10*21*21*4 bytes).
    safe_batches: list[int] = []
    for b in batch_sizes:
        bytes_est = b * (10 * 21 * 21 + GLOBAL_DIM + ACTION_DIM) * 4 * 8
        if torch_device.type == "cuda":
            free, _total = torch.cuda.mem_get_info()
            if bytes_est > free * 0.25:
                continue
        safe_batches.append(b)
    if not safe_batches:
        safe_batches = [1]

    before = {
        "note": "historical first bridge at 4dcfa30",
        "steps_per_second": 5.223,
        "decision": "PARTIAL",
        "dominant_stage": "forward_ms/encode Python loops",
    }

    batch_reports: dict[str, Any] = {}
    for batch_size in safe_batches:
        stage_samples: dict[str, list[float]] = defaultdict(list)
        hidden = policy.initial_hidden(batch=batch_size, device=torch_device)

        # Warm-up (excluded from steady-state stats)
        with torch.inference_mode():
            for i in range(warmup + steps):
                iter_t0 = time.perf_counter()
                stages: dict[str, float] = {k: 0.0 for k in STAGE_KEYS}

                t0 = time.perf_counter()
                eng_obs = get_obs(state, 0)
                stages["obs_extract_ms"] = (time.perf_counter() - t0) * 1000

                type_grid, owner, armies, globals_one, meta = extract_numpy_boards(eng_obs, h, w)
                stages["jax_to_numpy_ms"] = float(meta["jax_to_numpy_ms"])

                obs_obj = None
                if profile_python_object and batch_size == 1:
                    t0 = time.perf_counter()
                    obs_obj = _obs_from_arrays(type_grid, owner, armies, meta)
                    stages["python_object_ms"] = (time.perf_counter() - t0) * 1000
                    from generals_bot.legal import enumerate_legal_actions
                    from generals_bot.models.legal_mask import legal_mask_observation

                    t0 = time.perf_counter()
                    _ = enumerate_legal_actions(obs_obj)
                    stages["legal_enum_ms"] = (time.perf_counter() - t0) * 1000
                    t0 = time.perf_counter()
                    _ = legal_mask_observation(obs_obj, device=torch.device("cpu"))
                    stages["legal_mask_tensor_ms"] = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                type_b = np.stack([type_grid] * batch_size, axis=0)
                owner_b = np.stack([owner] * batch_size, axis=0)
                army_b = np.stack([armies] * batch_size, axis=0)
                glob_b = np.stack([globals_one] * batch_size, axis=0)
                stages["cpu_preprocess_ms"] = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                cells_np = encode_grids_batch_numpy(type_b, owner_b, army_b)
                stages["feature_channel_ms"] = (time.perf_counter() - t0) * 1000
                stages["padding_mask_ms"] = 0.0  # included in feature encode

                t0 = time.perf_counter()
                cell_t = torch.from_numpy(np.ascontiguousarray(cells_np))
                glob_t = torch.from_numpy(np.ascontiguousarray(glob_b))
                stages["numpy_to_torch_ms"] = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                if torch_device.type == "cuda":
                    cell_t = cell_t.to(torch_device, non_blocking=True)
                    glob_t = glob_t.to(torch_device, non_blocking=True)
                    torch.cuda.synchronize()
                stages["h2d_ms"] = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                out = policy.forward_tensors(cell_t, glob_t, hidden, deterministic=True)
                if torch_device.type == "cuda":
                    torch.cuda.synchronize()
                stages["mlp_forward_ms"] = (time.perf_counter() - t0) * 1000
                hidden = out["hidden"]
                logits = out["logits"]

                t0 = time.perf_counter()
                # Pass-only mask for throughput path (full legal enum measured separately on bs=1)
                mask = torch.zeros(batch_size, ACTION_DIM, dtype=torch.bool, device=torch_device)
                mask[:, 0] = True
                masked = apply_action_mask(logits, mask)
                stages["mask_apply_ms"] = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                action_idx = torch.argmax(masked, dim=-1)
                stages["action_select_ms"] = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                action_cpu = action_idx.detach().cpu().numpy()
                stages["d2h_ms"] = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                _ = int(action_cpu[0])
                actions = jnp.stack(
                    [
                        jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32),
                        jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32),
                    ]
                )
                stages["action_to_jax_ms"] = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                state, info = transition(state, actions)
                stages["env_step_ms"] = (time.perf_counter() - t0) * 1000
                if bool(info.is_done):
                    t0 = time.perf_counter()
                    state = make_board(env, seed + 1)
                    stages["env_reset_ms"] = (time.perf_counter() - t0) * 1000
                    hidden = policy.initial_hidden(batch=batch_size, device=torch_device)

                stages["e2e_iter_ms"] = (time.perf_counter() - iter_t0) * 1000

                if i >= warmup:
                    for k, v in stages.items():
                        stage_samples[k].append(v)

        total_ms = float(sum(stage_samples["e2e_iter_ms"]))
        steady_steps = len(stage_samples["e2e_iter_ms"])
        e2e_s = total_ms / 1000.0
        steps_per_s = steady_steps / max(e2e_s, 1e-9)
        env_only_ms = sum(stage_samples["env_step_ms"]) + sum(stage_samples["env_reset_ms"])
        encode_ms = sum(stage_samples["feature_channel_ms"]) + sum(stage_samples["numpy_to_torch_ms"])
        transfer_ms = sum(stage_samples["h2d_ms"]) + sum(stage_samples["d2h_ms"])

        batch_reports[str(batch_size)] = {
            "batch_size": batch_size,
            "warmup": warmup,
            "steady_steps": steady_steps,
            "steps_per_second_e2e": steps_per_s,
            "env_observations_per_second": steady_steps / max(env_only_ms / 1000.0, 1e-9),
            "encoding_observations_per_second": (steady_steps * batch_size)
            / max(encode_ms / 1000.0, 1e-9),
            "mlp_observations_per_second": (steady_steps * batch_size)
            / max(sum(stage_samples["mlp_forward_ms"]) / 1000.0, 1e-9),
            "games_per_hour_est": steps_per_s * 3600 / 1200,
            "transfer_share_pct": 100.0 * transfer_ms / max(total_ms, 1e-9),
            "stages": {k: _summarise(stage_samples[k], total_ms) for k in STAGE_KEYS},
            "resources": _resource_snapshot(torch_device),
        }

    # CPU-only MLP throughput (encode + forward, no env)
    cpu_policy = RecurrentMLPPolicy().to("cpu").eval()
    cells = torch.zeros(32, 10, 21, 21)
    globs = torch.zeros(32, GLOBAL_DIM)
    hdn = cpu_policy.initial_hidden(32, device=torch.device("cpu"))
    with torch.inference_mode():
        for _ in range(5):
            cpu_policy.forward_tensors(cells, globs, hdn)
        t0 = time.perf_counter()
        n = 50
        for _ in range(n):
            out = cpu_policy.forward_tensors(cells, globs, hdn)
            hdn = out["hidden"]
        cpu_mlp_obs_s = (n * 32) / (time.perf_counter() - t0)

    gpu_mlp_obs_s = None
    if torch_device.type == "cuda":
        cells_g = cells.to(torch_device)
        globs_g = globs.to(torch_device)
        hdn_g = policy.initial_hidden(32, device=torch_device)
        with torch.inference_mode():
            for _ in range(5):
                policy.forward_tensors(cells_g, globs_g, hdn_g)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            n = 50
            for _ in range(n):
                out = policy.forward_tensors(cells_g, globs_g, hdn_g)
                hdn_g = out["hidden"]
            torch.cuda.synchronize()
            gpu_mlp_obs_s = (n * 32) / (time.perf_counter() - t0)

    primary = batch_reports[str(safe_batches[0])]
    e2e = primary["steps_per_second_e2e"]
    decision = "PASS"
    if e2e < 5:
        decision = "FAIL"
    elif e2e < 30:
        decision = "PARTIAL"
    # Even if e2e < 30, promote if batching scales and encoding no longer dominates.
    dominant = max(
        ((k, primary["stages"][k]["pct_of_total"]) for k in STAGE_KEYS if k != "e2e_iter_ms"),
        key=lambda kv: kv[1],
    )

    report = {
        "schema_version": 2,
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "parameter_count": policy.parameter_count(),
        "before": before,
        "batch_sizes_requested": batch_sizes,
        "batch_sizes_run": safe_batches,
        "by_batch": batch_reports,
        "mlp_observations_per_second_cpu_batch32": cpu_mlp_obs_s,
        "mlp_observations_per_second_gpu_batch32": gpu_mlp_obs_s,
        "primary_batch_size": safe_batches[0],
        "steps_per_second": e2e,
        "games_per_hour_est": primary["games_per_hour_est"],
        "dominant_stage": dominant[0],
        "dominant_stage_pct": dominant[1],
        "decision": decision,
        "decision_rationale": (
            "PASS if e2e>=30 and encoding not dominant; "
            "PARTIAL if improved but simulator/transfer limited; "
            "FAIL if bounded PPO impractical."
        ),
    }

    # Refine classification with evidence (do not invent PASS).
    encode_pct = primary["stages"]["feature_channel_ms"]["pct_of_total"]
    mlp_pct = primary["stages"]["mlp_forward_ms"]["pct_of_total"]
    env_pct = primary["stages"]["env_step_ms"]["pct_of_total"]
    if decision == "PASS" and encode_pct > 40:
        decision = "PARTIAL"
        report["decision"] = decision
        report["decision_note"] = "e2e threshold met but encoding still large share"
    if e2e >= 15 and encode_pct < 25 and (env_pct + mlp_pct) > 40:
        # Material improvement; still may be env-limited for marathon PPO.
        if e2e < 30:
            report["decision"] = "PARTIAL"
            report["decision_note"] = (
                "Vectorisation removed Python encode bottleneck; "
                "bounded PPO practical, long campaigns remain env/transfer limited."
            )

    out = Path("experiments/summaries")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "jax_pytorch_bridge_benchmark.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Also keep manifests copy for registries.
    man = Path("experiments/manifests")
    man.mkdir(parents=True, exist_ok=True)
    (man / "jax_pytorch_bridge_benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    report["path"] = str(path)
    return report


def main() -> None:
    report = run_bridge_benchmark()
    print(json.dumps({k: report[k] for k in report if k != "by_batch"}, indent=2))
    print("batches:", json.dumps({k: v["steps_per_second_e2e"] for k, v in report["by_batch"].items()}, indent=2))


if __name__ == "__main__":
    main()
