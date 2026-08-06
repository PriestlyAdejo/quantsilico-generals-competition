"""Early deployment feasibility measurement."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from generals_bot.competition_native_jax.policy import CompetitionNativePolicy, load_weights
from generals_bot.observation import Observation


def main() -> None:
    ckpt = Path("experiments/competition_native_jax/smoke/smoke_raw.npz")
    w = load_weights(ckpt)
    policy = CompetitionNativePolicy(weights=w, seed=0)
    policy.reset(18, 18)
    z = tuple(tuple(0 for _ in range(18)) for _ in range(18))
    owners = [list(row) for row in z]
    armies = [list(row) for row in z]
    types = [list(row) for row in z]
    owners[5][5] = 1
    armies[5][5] = 20
    types[5][5] = 4
    obs = Observation(
        18,
        18,
        10,
        1,
        20,
        0,
        0,
        tuple(tuple(r) for r in types),
        tuple(tuple(r) for r in owners),
        tuple(tuple(r) for r in armies),
    )
    t0 = time.perf_counter()
    policy2 = CompetitionNativePolicy(weights=load_weights(ckpt), seed=0)
    policy2.reset(18, 18)
    policy2.act(obs)
    cold = time.perf_counter() - t0
    lat = []
    for _ in range(12):
        t1 = time.perf_counter()
        policy.act(obs)
        lat.append(time.perf_counter() - t1)
    lat_s = sorted(lat)
    p50 = lat_s[len(lat_s) // 2]
    p95 = lat_s[int(len(lat_s) * 0.95)]
    p99 = lat_s[-1]
    size = ckpt.stat().st_size
    zip_est = size + 200_000
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    limits = json.loads(
        Path("experiments/manifests/competition_native_jax_deployment_limits.json").read_text(encoding="utf-8-sig")
    )
    feas = "DEPLOYMENT_ARCHITECTURE_FEASIBLE"
    if p99 > limits["limits"]["ordinary_action_deadline_s"]:
        feas = "DEPLOYMENT_REQUIRES_DISTILLATION"
    if size > limits["limits"]["compressed_package_bytes"]:
        feas = "DEPLOYMENT_RUNTIME_BLOCKED"
    gate = {
        "schema_version": 1,
        "kind": "COMPETITION_NATIVE_JAX_EARLY_DEPLOYMENT_GATE",
        "status": feas,
        "checkpoint": str(ckpt).replace("\\", "/"),
        "checkpoint_sha256": sha,
        "serialized_param_bytes": size,
        "zip_estimate_bytes": zip_est,
        "cold_load_and_first_action_s": cold,
        "steady_p50_s": p50,
        "steady_p95_s": p95,
        "steady_p99_s": p99,
        "backend": "numpy_transformer",
        "jaxlib_packaged": False,
        "parity_note": "deployment uses same NumPy transformer as smoke training prototype",
        "limits_ref": "experiments/manifests/competition_native_jax_deployment_limits.json",
    }
    out = Path("experiments/manifests/competition_native_jax_early_deployment_gate.json")
    out.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(feas, "p99", p99, "cold", cold, flush=True)


if __name__ == "__main__":
    main()
