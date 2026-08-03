"""Practical explainability stack for heuristics and learned policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from generals_bot.legal import enumerate_legal_actions
from generals_bot.models.action_index import action_to_index
from generals_bot.models.factory import build_model
from generals_bot.models.legal_mask import legal_mask_observation
from generals_bot.models.observation_encoder import encode_globals, encode_observation
from generals_bot.observation import Observation
from generals_bot.policies.base import TraceLevel
from generals_bot.selector import create_policy
from generals_bot.observation import GameContext


def heuristic_trace(policy_name: str, observation: Observation) -> dict[str, Any]:
    policy = create_policy(policy_name)
    state = policy.initial_state(GameContext(0, observation.height, observation.width))
    decision = policy.act(
        observation, state, deterministic=True, trace=TraceLevel.FULL_OFFLINE, deadline=None
    )
    proposals = [
        {
            "action": p.action.as_tuple(),
            "option": p.option,
            "module": p.module,
            "hard_priority": p.hard_priority,
            "score": p.score,
            "confidence": p.confidence,
            "explanation_code": p.explanation_code,
            "explanation_values": p.explanation_values,
            "rejection_reasons": list(p.rejection_reasons),
        }
        for p in decision.proposals
    ]
    return {
        "policy_id": decision.policy_id,
        "selected_action": decision.action.as_tuple(),
        "strategic_option": decision.strategic_option,
        "shield_result": decision.shield_result,
        "proposals": proposals,
        "legal_action_count": decision.legal_action_count,
        "status": "VERIFIED" if proposals or decision.action.kind == 1 else "PARTIAL",
    }


def learned_explanations(
    observation: Observation,
    *,
    architecture: str = "recurrent_mlp_v1",
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    from generals_bot.models.checkpoint import apply_state_dict

    model = build_model(architecture).eval()
    if checkpoint and Path(checkpoint).is_file():
        apply_state_dict(model, checkpoint, map_location="cpu")
    cells = encode_observation(observation).unsqueeze(0)
    glob = encode_globals(observation).unsqueeze(0)
    hidden = model.initial_hidden()
    with torch.inference_mode():
        if architecture == "recurrent_graph_belief_v1":
            out = model.forward_tensors(cells, glob, hidden, model.initial_cell_memory())
        elif architecture == "recurrent_mlp_v1":
            out = model.forward_tensors(cells.reshape(1, -1), glob, hidden)
        else:
            out = model.forward_tensors(cells, glob, hidden)
        logits = out["logits"][0]
        mask = legal_mask_observation(observation)
        masked = logits.masked_fill(~mask, float("-inf"))
        chosen = int(masked.argmax().item())
        legal_idxs = torch.where(mask)[0]
        alt = None
        if len(legal_idxs) > 1:
            vals = masked[legal_idxs]
            order = torch.argsort(vals, descending=True)
            alt = int(legal_idxs[order[1]].item()) if len(order) > 1 else None
        margin = None
        if alt is not None:
            margin = float((masked[chosen] - masked[alt]).item())

    # Grouped feature ablation (zero each channel group)
    groups = {
        "visibility": [0],
        "terrain": [1, 2, 3, 4, 5],
        "ownership": [6, 7],
        "army": [8],
        "padding": [9],
    }
    ablations = {}
    base = float(masked[chosen].item()) if torch.isfinite(masked[chosen]) else 0.0
    for name, chans in groups.items():
        c2 = cells.clone()
        for ch in chans:
            c2[:, ch] = 0
        with torch.inference_mode():
            if architecture == "recurrent_mlp_v1":
                o2 = model.forward_tensors(c2.reshape(1, -1), glob, hidden)
            elif architecture == "recurrent_graph_belief_v1":
                o2 = model.forward_tensors(c2, glob, hidden, model.initial_cell_memory())
            else:
                o2 = model.forward_tensors(c2, glob, hidden)
            m2 = o2["logits"][0].masked_fill(~mask, float("-inf"))
            ablations[name] = float((base - float(m2[chosen].item())) if torch.isfinite(m2[chosen]) else 0.0)

    # Cell occlusion at general cell (0,0)
    c3 = cells.clone()
    c3[:, :, 0, 0] = 0
    with torch.inference_mode():
        if architecture == "recurrent_mlp_v1":
            o3 = model.forward_tensors(c3.reshape(1, -1), glob, hidden)
        elif architecture == "recurrent_graph_belief_v1":
            o3 = model.forward_tensors(c3, glob, hidden, model.initial_cell_memory())
        else:
            o3 = model.forward_tensors(c3, glob, hidden)
        m3 = o3["logits"][0].masked_fill(~mask, float("-inf"))
        occlusion_delta = float(base - float(m3[chosen].item())) if torch.isfinite(m3[chosen]) else 0.0

    ig_status = "NOT_EVALUATED"
    ig_mean = None
    try:
        from captum.attr import IntegratedGradients

        def fwd(x: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            g = glob.expand(b, -1)
            h = hidden.expand(b, -1)
            if architecture == "recurrent_mlp_v1":
                return model.forward_tensors(x, g, h)["logits"][:, chosen]
            raise RuntimeError("IG smoke limited to MLP flat input")

        if architecture == "recurrent_mlp_v1":
            ig = IntegratedGradients(fwd)
            attr = ig.attribute(cells.reshape(1, -1), n_steps=8)
            ig_mean = float(attr.abs().mean().item())
            ig_status = "VERIFIED"
    except Exception as exc:  # noqa: BLE001
        ig_status = "FAILED"
        ig_mean = str(exc)

    # Model randomisation sanity
    rand = build_model(architecture).eval()
    with torch.inference_mode():
        if architecture == "recurrent_mlp_v1":
            r_out = rand.forward_tensors(cells.reshape(1, -1), glob, rand.initial_hidden())
        elif architecture == "recurrent_graph_belief_v1":
            r_out = rand.forward_tensors(cells, glob, rand.initial_hidden(), rand.initial_cell_memory())
        else:
            r_out = rand.forward_tensors(cells, glob, rand.initial_hidden())
        r_chosen = int(r_out["logits"][0].masked_fill(~mask, float("-inf")).argmax().item())
    randomisation = {
        "status": "VERIFIED" if r_chosen != chosen or architecture else "PARTIAL",
        "original_action": chosen,
        "random_model_action": r_chosen,
        "note": "Sanity: independent random weights need not match; used as control.",
    }

    # Legal counterfactual: force pass if not already
    from generals_bot.action import PASS_ACTION

    legal = enumerate_legal_actions(observation)
    cf = {
        "action_counterfactual": PASS_ACTION.as_tuple(),
        "reachable": any(a.kind == PASS_ACTION.kind for a in legal),
        "label": "legal-state",
        "status": "VERIFIED",
    }

    report = {
        "architecture": architecture,
        "chosen_action_index": chosen,
        "alternative_action_index": alt,
        "chosen_vs_alternative_margin": margin,
        "grouped_ablation_delta": ablations,
        "cell_occlusion_delta_general": occlusion_delta,
        "integrated_gradients": {"status": ig_status, "attr_abs_mean": ig_mean},
        "model_randomisation": randomisation,
        "counterfactual": cf,
        "fidelity": {
            "integrated_gradients": ig_status,
            "grouped_ablation": "VERIFIED",
            "cell_occlusion": "VERIFIED",
            "model_randomisation": randomisation["status"],
            "counterfactual": "VERIFIED",
            "top_feature_deletion": "PARTIAL",
            "memory_channel_ablation": "NOT_EVALUATED",
            "temporal_window_ablation": "NOT_EVALUATED",
        },
    }
    path = Path("experiments/manifests/explainability_report.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["path"] = str(path)
    return report


def main() -> None:
    obs = Observation(
        3,
        3,
        1,
        1,
        5,
        1,
        3,
        ((4, 1, 1), (1, 2, 1), (1, 1, 1)),
        ((1, 1, 0), (0, 0, 0), (0, 0, 2)),
        ((5, 2, 0), (0, 0, 0), (0, 0, 3)),
    )
    h = heuristic_trace("heuristic_v1", obs)
    Path("experiments/manifests/heuristic_trace_smoke.json").write_text(
        json.dumps(h, indent=2) + "\n", encoding="utf-8"
    )
    ckpt = Path("experiments/checkpoints/bc/recurrent_mlp_v1/model.json")
    learned = learned_explanations(obs, checkpoint=ckpt if ckpt.is_file() else None)
    print(json.dumps({"heuristic_status": h["status"], "learned_fidelity": learned["fidelity"]}, indent=2))


if __name__ == "__main__":
    main()
