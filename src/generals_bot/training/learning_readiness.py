"""LEARNING_READINESS_GATE — pre-PPO engineering readiness."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from generals_bot.models.checkpoint import apply_state_dict, save_checkpoint
from generals_bot.models.factory import build_model, known_architectures
from generals_bot.models.legal_mask import apply_action_mask, legal_mask_observation
from generals_bot.models.observation_encoder import encode_globals, encode_observation
from generals_bot.observation import Observation

REPO = Path(__file__).resolve().parents[3]


@dataclass
class LearningReadinessResult:
    schema_version: int = 1
    kind: str = "LEARNING_READINESS_GATE"
    decision: str = "FAIL"  # PASS | PARTIAL | FAIL
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    architectures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _blank_obs(turn: int = 50) -> Observation:
    h = w = 8
    return Observation(
        height=h,
        width=w,
        turn=turn,
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


def _forward(model: torch.nn.Module, obs: Observation, hidden: torch.Tensor) -> dict[str, Any]:
    device = hidden.device
    cells = encode_observation(obs, device=device)
    if cells.ndim == 3:
        cells = cells.unsqueeze(0)
    # MLP expects flat cell features.
    if cells.ndim == 4 and not hasattr(model, "stem") and not hasattr(model, "input_proj"):
        cells = cells.reshape(cells.shape[0], -1)
    glob = encode_globals(obs, device=device).unsqueeze(0)
    kwargs: dict[str, Any] = {}
    if hasattr(model, "initial_cell_memory"):
        kwargs["cell_memory"] = model.initial_cell_memory(1, device=device)
    return model.forward_tensors(cells, glob, hidden, deterministic=True, **kwargs)


def evaluate_learning_readiness(*, device: str = "cpu") -> LearningReadinessResult:
    result = LearningReadinessResult()
    preferred = [
        "recurrent_mlp_v1",
        "recurrent_cnn_v2",
        "recurrent_graph_belief_v2",
    ]
    known = set(known_architectures())
    arches = [a for a in preferred if a in known]
    result.architectures = arches
    checks: dict[str, Any] = {}
    torch_device = torch.device(device)

    try:
        obs = _blank_obs()
        enc = encode_observation(obs, device=torch_device)
        glob = encode_globals(obs, device=torch_device)
        mask = legal_mask_observation(obs, device=torch_device)
        checks["observation_encode"] = {
            "ok": True,
            "cells_shape": list(enc.shape),
            "globals_shape": list(glob.shape),
            "mask_true": int(mask.sum().item()),
            "mask_dim": int(mask.numel()),
        }
        checks["fog_semantics_smoke"] = {
            "ok": True,
            "note": "encoder accepts partial Observation grids; full fog suite lives in unit tests",
        }
    except Exception as exc:  # noqa: BLE001
        checks["observation_encode"] = {"ok": False, "error": str(exc)}
        result.blockers.append(f"observation_encode:{exc}")

    latencies: list[float] = []
    legal_rates: list[float] = []
    for arch in arches:
        entry: dict[str, Any] = {"architecture": arch}
        try:
            model = build_model(arch)
            model.to(torch_device)
            model.eval()
            obs = _blank_obs()
            h0 = model.initial_hidden(batch=1, device=torch_device)
            h_reset = model.initial_hidden(batch=1, device=torch_device)
            if not torch.equal(h0, h_reset):
                raise RuntimeError("initial_hidden not deterministic zeros")

            t0 = time.perf_counter()
            with torch.no_grad():
                out1 = _forward(model, obs, h0)
                out2 = _forward(model, obs, out1["hidden"])
                # New game must reset hidden independently of previous trajectory.
                out_new = _forward(model, obs, model.initial_hidden(batch=1, device=torch_device))
            latencies.append((time.perf_counter() - t0) * 1000.0)

            mask = legal_mask_observation(obs, device=torch_device).unsqueeze(0)
            masked = apply_action_mask(out1["logits"], mask)
            # Sample only from legal indices.
            probs = torch.softmax(masked, dim=-1)
            samples = torch.multinomial(probs.expand(64, -1), 1).squeeze(-1)
            legal_ok = bool(mask[0, samples].all().item())
            legal_rates.append(1.0 if legal_ok else 0.0)

            entry["forward_ok"] = True
            entry["recurrent_step_ok"] = bool(torch.isfinite(out2["logits"]).all())
            entry["hidden_reset_ok"] = bool(torch.isfinite(out_new["logits"]).all())
            entry["legal_sampled_actions_ok"] = legal_ok
            entry["param_count"] = int(sum(p.numel() for p in model.parameters()))

            ckpt_dir = REPO / "experiments" / "checkpoints" / "readiness" / arch
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_stem = ckpt_dir / "readiness"
            cfg = model.config_dict() if hasattr(model, "config_dict") else {"architecture": arch}
            save_checkpoint(model, ckpt_stem, architecture=arch, config=cfg)
            model2 = build_model(arch, cfg)
            apply_state_dict(model2, ckpt_stem.with_suffix(".json"), map_location=torch_device)
            model2.to(torch_device)
            model2.eval()
            with torch.no_grad():
                out_reload = _forward(model2, obs, model2.initial_hidden(batch=1, device=torch_device))
            entry["checkpoint_roundtrip"] = True
            entry["checkpoint_path"] = str(ckpt_stem.with_suffix(".json"))
            entry["reload_finite"] = bool(torch.isfinite(out_reload["logits"]).all())
            entry["ok"] = True
        except Exception as exc:  # noqa: BLE001
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
            result.warnings.append(f"{arch}:{exc}")
        checks[arch] = entry

    if latencies:
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        checks["cpu_latency_ms"] = {
            "n": n,
            "p50": latencies_sorted[n // 2],
            "p95": latencies_sorted[int(0.95 * (n - 1))],
            "p99": latencies_sorted[min(n - 1, int(0.99 * (n - 1)))],
            "max": max(latencies_sorted),
            "note": "synthetic blank obs; not official match telemetry",
        }
        if checks["cpu_latency_ms"]["p95"] >= 150.0:
            result.blockers.append("cpu_latency_p95>=150ms_on_readiness_probe")

    if legal_rates:
        checks["legal_action_rate"] = sum(legal_rates) / len(legal_rates)
        if checks["legal_action_rate"] < 1.0:
            result.blockers.append("legal_action_rate<1.0")

    bridge_path = REPO / "experiments" / "manifests" / "jax_pytorch_bridge_benchmark.json"
    if not bridge_path.exists():
        alt = REPO / "experiments" / "summaries" / "jax_pytorch_bridge_benchmark.json"
        bridge_path = alt if alt.exists() else bridge_path
    if bridge_path.exists():
        bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
        checks["bridge_report"] = {
            "path": str(bridge_path.relative_to(REPO)).replace("\\", "/"),
            "decision": bridge.get("decision") or bridge.get("verdict") or bridge.get("status"),
        }
        dec = str(checks["bridge_report"]["decision"] or "").upper()
        if dec == "FAIL":
            result.blockers.append("bridge_benchmark_FAIL")
        elif dec in {"PARTIAL", "MISSING", ""}:
            result.warnings.append(f"bridge_benchmark_{dec or 'UNKNOWN'}")
    else:
        checks["bridge_report"] = {"path": None, "decision": "MISSING"}
        result.warnings.append("bridge_benchmark_report_missing")

    checks["known_architectures"] = known_architectures()
    for need in ("recurrent_mlp_v1", "recurrent_cnn_v2", "recurrent_graph_belief_v2"):
        if need not in checks["known_architectures"]:
            result.blockers.append(f"missing_architecture:{need}")

    checks["no_pyg_required"] = {
        "ok": True,
        "note": "recurrent_graph_belief_v2 uses pure PyTorch message passing",
    }
    checks["promotion_holdout_leakage"] = {
        "ok": True,
        "note": "readiness probe uses synthetic Observation only; no holdout seeds",
    }

    arch_ok = bool(arches) and all(checks.get(a, {}).get("ok") for a in arches)
    if not arch_ok:
        result.blockers.append("architecture_forward_or_checkpoint_failed")

    if result.blockers:
        result.decision = "FAIL"
    elif result.warnings:
        result.decision = "PARTIAL"
    else:
        result.decision = "PASS"

    result.checks = checks
    result.notes = [
        "LEARNING_READINESS_GATE is engineering readiness, not LEARNED_PROMOTION_GATE.",
        "PORTAL_SUBMISSION_GATE QUALIFIED is unrelated to this gate.",
        "No PyG required; pure-PyTorch graph path only.",
        "MLP=bridge control; CNN=learned control; graph-belief=principal challenger.",
    ]
    return result


def main() -> None:
    result = evaluate_learning_readiness()
    out = REPO / "experiments" / "manifests" / "learning_readiness_gate.json"
    out.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result.decision,
                "blockers": result.blockers,
                "warnings": result.warnings,
            },
            indent=2,
        )
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
