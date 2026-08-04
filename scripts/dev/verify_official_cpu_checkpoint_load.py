"""Official-.venv style CPU load + one legal action from a safetensors checkpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from generals_bot.models.checkpoint import apply_state_dict, load_checkpoint_payload
from generals_bot.models.factory import build_model
from generals_bot.models.legal_mask import apply_action_mask, legal_mask_observation
from generals_bot.models.observation_encoder import encode_globals, encode_observation
from generals_bot.observation import Observation

REPO = Path(__file__).resolve().parents[2]


def _blank_obs() -> Observation:
    h = w = 8
    return Observation(
        height=h,
        width=w,
        turn=12,
        my_land=4,
        my_army=12,
        opp_land=3,
        opp_army=9,
        type_grid=tuple(tuple(4 if r == 0 and c == 0 else 1 for c in range(w)) for r in range(h)),
        owner_grid=tuple(tuple(1 if r < 2 and c < 2 else 0 for c in range(w)) for r in range(h)),
        army_grid=tuple(
            tuple(8 if r == 0 and c == 0 else (2 if r < 2 and c < 2 else 0) for c in range(w))
            for r in range(h)
        ),
    )


def verify_checkpoint(config_path: Path) -> dict:
    payload = load_checkpoint_payload(config_path)
    arch = payload["architecture"]
    model = build_model(arch, payload.get("config"))
    apply_state_dict(model, config_path, map_location="cpu")
    model.eval()
    obs = _blank_obs()
    device = torch.device("cpu")
    h = model.initial_hidden(batch=1, device=device)
    cells = encode_observation(obs, device=device)
    if cells.ndim == 3:
        cells = cells.unsqueeze(0)
    if not hasattr(model, "stem") and not hasattr(model, "input_proj"):
        cells = cells.reshape(1, -1)
    glob = encode_globals(obs, device=device).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        kwargs = {}
        if hasattr(model, "initial_cell_memory"):
            kwargs["cell_memory"] = model.initial_cell_memory(1, device=device)
        out = model.forward_tensors(cells, glob, h, deterministic=True, **kwargs)
        mask = legal_mask_observation(obs, device=device).unsqueeze(0)
        masked = apply_action_mask(out["logits"], mask)
        action = int(torch.argmax(masked, dim=-1).item())
        legal = bool(mask[0, action].item())
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "checkpoint": str(config_path).replace("\\", "/"),
        "architecture": arch,
        "parameter_count": payload.get("parameter_count"),
        "action_index": action,
        "legal": legal,
        "latency_ms": latency_ms,
        "device": "cpu",
        "ok": legal and bool(torch.isfinite(out["logits"]).all()),
    }


def main() -> None:
    candidates = [
        REPO / "experiments" / "checkpoints" / "ppo" / "recurrent_mlp_v1" / "model.json",
        REPO / "experiments" / "checkpoints" / "ppo" / "recurrent_cnn_v2" / "model.json",
        REPO / "experiments" / "checkpoints" / "ppo" / "recurrent_graph_belief_v2" / "model.json",
        REPO
        / "experiments"
        / "checkpoints"
        / "readiness"
        / "recurrent_mlp_v1"
        / "readiness.json",
    ]
    results = []
    for path in candidates:
        if path.is_file():
            results.append(verify_checkpoint(path))
    report = {
        "schema_version": 1,
        "kind": "OFFICIAL_VENV_CPU_LOAD",
        "results": results,
        "all_ok": bool(results) and all(r["ok"] for r in results),
        "note": "Uses project .venv on CPU map_location; approximates official runtime load path.",
    }
    out = REPO / "experiments" / "manifests" / "official_venv_cpu_load.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
