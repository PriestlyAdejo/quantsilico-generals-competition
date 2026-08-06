"""EARLY_STUDENT_FEASIBILITY_RULE: shape/runtime gate with random/smoke weights only.

This is NOT trained distillation. Distillation is forbidden until a teacher checkpoint
is selected.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from generals_bot.competition_native_jax.constants import EMB_DIM, N_HEADS, N_LAYERS
from generals_bot.competition_native_jax.obs_memory import N_GLOBAL, N_SPATIAL
from generals_bot.competition_native_jax.policy import CompetitionNativePolicy
from generals_bot.competition_native_jax.transformer import TransformerWeights, init_weights
from generals_bot.observation import Observation


def _synthetic_obs(h: int = 18, w: int = 18) -> Observation:
    z = tuple(tuple(0 for _ in range(w)) for _ in range(h))
    owners = [list(row) for row in z]
    armies = [list(row) for row in z]
    types = [list(row) for row in z]
    owners[5][5] = 1
    armies[5][5] = 20
    types[5][5] = 4
    return Observation(
        h,
        w,
        10,
        1,
        20,
        0,
        0,
        tuple(tuple(r) for r in types),
        tuple(tuple(r) for r in owners),
        tuple(tuple(r) for r in armies),
    )


def _init_shape(seed: int, emb: int, layers: int, heads: int) -> TransformerWeights:
    """Init a smaller student shape by reusing Xavier init then resizing heads."""
    # init_weights uses module constants; build manually for alternate shapes.
    rng = np.random.default_rng(seed)

    def xavier(shape: tuple[int, ...]) -> np.ndarray:
        fan_in = shape[0] if len(shape) >= 2 else shape[-1]
        fan_out = shape[-1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=shape).astype(np.float32)

    from generals_bot.competition_native_jax.constants import (
        HL_GAUSS_BINS,
        NUM_PATCHES,
        PATCH,
    )
    from generals_bot.competition_native_jax.patchify import PATCH as _  # noqa: F401

    w = TransformerWeights(
        patch_proj=xavier((N_SPATIAL * PATCH * PATCH, emb)),
        cls=xavier((emb,)),
        pos=xavier((NUM_PATCHES + 1, emb)),
        global_proj=xavier((N_GLOBAL, emb)),
        attn_w=[],
        attn_out=[],
        ff_w1=[],
        ff_w2=[],
        move_head=xavier((emb, PATCH * PATCH * 8)),
        build_head=xavier((emb, PATCH * PATCH)),
        pass_head=xavier((emb,)),
        value_head=xavier((emb, HL_GAUSS_BINS)),
    )
    for _i in range(layers):
        w.attn_w.append(xavier((emb, 3 * emb)))
        w.attn_out.append(xavier((emb, emb)))
        w.ff_w1.append(xavier((emb, 4 * emb)))
        w.ff_w2.append(xavier((4 * emb, emb)))
    # heads is informational for this NumPy path (emb must be divisible)
    if emb % heads != 0:
        raise ValueError(f"emb {emb} not divisible by heads {heads}")
    return w


def _measure_policy(weights: TransformerWeights, n: int = 12) -> dict:
    policy = CompetitionNativePolicy(weights=weights, seed=0)
    obs = _synthetic_obs()
    policy.reset(18, 18)
    t0 = time.perf_counter()
    policy.act(obs)
    cold = time.perf_counter() - t0
    lats: list[float] = []
    for _ in range(n):
        t1 = time.perf_counter()
        policy.act(obs)
        lats.append(time.perf_counter() - t1)
    lats_s = sorted(lats)
    return {
        "cold_start_s": cold,
        "p50_s": lats_s[len(lats_s) // 2],
        "p95_s": lats_s[int(len(lats_s) * 0.95)],
        "p99_s": lats_s[-1],
    }


def main() -> None:
    limits = json.loads(
        Path("experiments/manifests/competition_native_jax_deployment_limits.json").read_text(encoding="utf-8-sig")
    )
    max_p99 = float(limits["limits"]["max_p99_for_promotion_s"])
    shapes = [
        {"name": "teacher_locked", "emb": EMB_DIM, "layers": N_LAYERS, "heads": N_HEADS},
        {"name": "student_emb96_d2_h4", "emb": 96, "layers": 2, "heads": 4},
        {"name": "student_emb64_d2_h4", "emb": 64, "layers": 2, "heads": 4},
        {"name": "student_emb48_d1_h4", "emb": 48, "layers": 1, "heads": 4},
    ]
    results = []
    credible = None
    for spec in shapes:
        try:
            if spec["name"] == "teacher_locked":
                w = init_weights(0)
            else:
                w = _init_shape(0, spec["emb"], spec["layers"], spec["heads"])
            # Monkeypatch forward expects N_LAYERS from weights list lengths — OK.
            # CompetitionNativePolicy uses forward() which iterates len(attn_w).
            m = _measure_policy(w)
            ok = m["p99_s"] <= max_p99
            row = {**spec, **m, "credible_cpu_package_path": ok, "status": "OK" if ok else "TOO_SLOW"}
            if ok and credible is None and spec["name"] != "teacher_locked":
                credible = row
            results.append(row)
        except Exception as exc:  # noqa: BLE001
            results.append({**spec, "status": "ERROR", "error": str(exc), "credible_cpu_package_path": False})

    classification = (
        "STUDENT_SHAPE_FEASIBLE"
        if credible is not None
        else "NO_CREDIBLE_STUDENT_SHAPE"
    )
    if any(r.get("name") == "teacher_locked" and r.get("credible_cpu_package_path") for r in results):
        classification = "DIRECTLY_DEPLOYABLE_TEACHER_SHAPE"
    gate = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_EARLY_STUDENT_FEASIBILITY",
        "status": classification,
        "note": "Random/smoke weights only. Not distillation. Teacher training not yet selected.",
        "limits_ref": "experiments/manifests/competition_native_jax_deployment_limits.json",
        "max_p99_for_promotion_s": max_p99,
        "shapes": results,
        "selected_student_shape": credible,
        "distillation_authorized": False,
    }
    out = Path("experiments/manifests/competition_native_jax_early_student_feasibility.json")
    out.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(classification, json.dumps(credible or {}, indent=2), flush=True)


if __name__ == "__main__":
    main()
