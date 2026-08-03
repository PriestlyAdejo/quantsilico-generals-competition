"""Offline explainability hooks (Captum when available)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from generals_bot.models.factory import build_model
from generals_bot.models.observation_encoder import encode_globals, encode_observation
from generals_bot.observation import Observation


def integrated_gradients_smoke(observation: Observation, architecture: str = "recurrent_mlp_v1") -> dict:
    model = build_model(architecture).eval()
    cells = encode_observation(observation).reshape(1, -1)
    glob = encode_globals(observation).unsqueeze(0)
    hidden = model.initial_hidden()

    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        return {
            "status": "SKIPPED_NO_CAPTUM",
            "architecture": architecture,
            "note": "Captum not installed in this environment",
        }

    def forward_fn(x: torch.Tensor) -> torch.Tensor:
        # x may be expanded by Captum along batch; broadcast globals/hidden.
        b = x.shape[0]
        g = glob.expand(b, -1)
        h = hidden.expand(b, -1)
        out = model.forward_tensors(x, g, h, deterministic=True)
        return out["value"]

    ig = IntegratedGradients(forward_fn)
    attributions = ig.attribute(cells, n_steps=8)
    path = Path("experiments/manifests/explain_smoke.json")
    report = {
        "status": "OK",
        "architecture": architecture,
        "attr_abs_mean": float(attributions.abs().mean().item()),
        "attr_shape": list(attributions.shape),
        "fidelity_note": "smoke attribution only; not a full fidelity study",
    }
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
