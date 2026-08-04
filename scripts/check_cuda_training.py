"""CUDA training smoke + GPU execution + PPO update-effectiveness gates (Phase 9E)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from generals_bot.models.checkpoint import apply_state_dict
from generals_bot.models.factory import build_model
from generals_bot.models.model_forward import adapt_forward_output
from generals_bot.training.device_policy import (
    assert_module_on_cuda,
    cuda_runtime_snapshot,
    resolve_training_device,
)
from generals_bot.training.ppo import run_bounded_ppo

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "experiments" / "manifests"

CNN_CKPT = REPO / "experiments/checkpoints/initial/initial_cnn_cnn_bc_init_seed11/chunk_0/model.json"
GRAPH_CKPT = REPO / "experiments/checkpoints/initial/initial_graph_graph_bc_init_seed7/chunk_0/model.json"

ARCHS = {
    "cnn": ("recurrent_cnn_v2", CNN_CKPT),
    "graph": ("recurrent_graph_belief_v2", GRAPH_CKPT),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _param_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        total += float(p.detach().float().norm().item() ** 2)
    return total**0.5


def _forward_result(model: nn.Module, cells: torch.Tensor, glob: torch.Tensor, hidden: torch.Tensor):
    # Graph policies take positional cell_memory; CNN does not.
    if hasattr(model, "initial_cell_memory"):
        cell_mem = model.initial_cell_memory(cells.shape[0], device=cells.device)
        raw = model.forward_tensors(cells, glob, hidden, cell_mem)
    else:
        raw = model.forward_tensors(cells, glob, hidden)
    return adapt_forward_output(raw)


def _forward_logits(model: nn.Module, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    cells = torch.zeros(1, 10, 21, 21, device=device)
    glob = torch.zeros(1, 9, device=device)
    hidden = model.initial_hidden(1, device=torch.device(device))
    with torch.no_grad():
        fwd = _forward_result(model, cells, glob, hidden)
    return fwd.logits.detach(), fwd.value.detach()


def smoke_architecture(arch_key: str, device: str) -> dict[str, Any]:
    architecture, ckpt = ARCHS[arch_key]
    if not ckpt.is_file():
        return {"architecture": architecture, "gate": "FAIL", "error": f"missing checkpoint {ckpt}"}
    model = build_model(architecture).to(device)
    apply_state_dict(model, ckpt, map_location=device)
    assert_module_on_cuda(model, expected=device)
    model.train()
    cells = torch.randn(4, 10, 21, 21, device=device)
    glob = torch.randn(4, 9, device=device)
    hidden = model.initial_hidden(4, device=torch.device(device))
    fwd = _forward_result(model, cells, glob, hidden)
    loss = fwd.logits.float().pow(2).mean() + fwd.value.float().pow(2).mean()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_modules: dict[str, float] = {}
    params_with_grad = 0
    params_total = 0
    for name, p in model.named_parameters():
        params_total += 1
        if p.grad is not None and float(p.grad.abs().sum()) > 0:
            params_with_grad += 1
            root = name.split(".", 1)[0]
            grad_modules[root] = grad_modules.get(root, 0.0) + float(p.grad.detach().norm().item())
    opt.step()
    torch.cuda.synchronize()
    delta = 0.0
    for n, p in model.named_parameters():
        delta += float((p.detach() - before[n]).norm().item() ** 2)
    delta = delta**0.5
    mem = int(torch.cuda.memory_allocated())
    gate = "PASS" if params_with_grad > 0 and delta > 0 and mem > 0 and torch.isfinite(loss) else "FAIL"
    return {
        "architecture": architecture,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": _sha256(ckpt),
        "checkpoint_sha16": _sha256(ckpt)[:16],
        "device": device,
        "loss": float(loss.detach().cpu()),
        "params_total": params_total,
        "params_with_grad": params_with_grad,
        "pct_params_with_grad": params_with_grad / max(params_total, 1),
        "grad_norm_by_module": grad_modules,
        "parameter_delta_norm": delta,
        "cuda_memory_allocated_bytes": mem,
        "gate": gate,
    }


def ppo_update_probe(arch_key: str, device: str) -> dict[str, Any]:
    architecture, ckpt = ARCHS[arch_key]
    model0 = build_model(architecture).to(device)
    apply_state_dict(model0, ckpt, map_location=device)
    logits0, value0 = _forward_logits(model0, device)
    norm0 = _param_norm(model0)

    out_dir = REPO / "experiments" / "checkpoints" / "phase9e_gpu_smoke" / arch_key
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = run_bounded_ppo(
        architecture=architecture,
        rollout_steps=32,
        updates=2,
        lr=3e-4,
        seed=9001,
        device=device,
        init_checkpoint=ckpt,
        out_dir=out_dir,
    )
    elapsed = time.perf_counter() - t0

    # Prefer trainer-reported checkpoint
    written_path = Path(result.get("checkpoint") or "")
    written = [written_path] if written_path.is_file() else sorted(out_dir.glob("**/model.json"))
    model1 = build_model(architecture).to(device)
    if written:
        apply_state_dict(model1, written[-1], map_location=device)
    else:
        apply_state_dict(model1, ckpt, map_location=device)
    logits1, value1 = _forward_logits(model1, device)
    norm1 = _param_norm(model1)
    action_delta = float((logits1 - logits0).norm().item())
    value_delta = float((value1 - value0).norm().item())
    param_delta = abs(norm1 - norm0)

    history = result.get("history") or []
    finite_losses = True
    nonzero_adv = False
    for h in history:
        for key in ("policy_loss", "value_loss", "entropy"):
            if key in h and not torch.isfinite(torch.tensor(float(h[key]))):
                finite_losses = False
        if abs(float(h.get("advantage_std", 0.0))) > 1e-8 or abs(float(h.get("advantage_mean", 0.0))) > 1e-8:
            nonzero_adv = True

    gate = "PASS"
    reasons: list[str] = []
    if not str(result.get("device", device)).startswith("cuda"):
        gate = "FAIL"
        reasons.append("ppo reported non-cuda device")
    if not history:
        gate = "FAIL"
        reasons.append("empty PPO history")
    if not finite_losses:
        gate = "FAIL"
        reasons.append("non-finite losses")
    if not nonzero_adv and action_delta <= 0 and param_delta <= 0:
        gate = "FAIL"
        reasons.append("no useful advantages and no measurable parameter/policy change")
    if action_delta <= 0 and value_delta <= 0 and param_delta <= 0 and not history:
        gate = "FAIL"
        reasons.append("no measurable policy/value/parameter change")

    return {
        "architecture": architecture,
        "device": result.get("device", device),
        "elapsed_s": elapsed,
        "ppo_result_keys": sorted(result.keys()),
        "history": history,
        "env_steps": int(result.get("rollout_steps", 0)) * int(result.get("updates", 0)),
        "action_distribution_delta": action_delta,
        "value_prediction_delta": value_delta,
        "parameter_norm_delta": param_delta,
        "nonzero_advantages_observed": nonzero_adv,
        "checkpoint_written": [str(p) for p in written],
        "gate": gate,
        "reasons": reasons,
        "raw_summary": {k: result[k] for k in result if k not in {"history", "telemetry", "reward_config"}},
    }


def cpu_gpu_benchmark(arch_key: str, device: str, batches: tuple[int, ...] = (1, 4, 16, 32)) -> dict[str, Any]:
    architecture, ckpt = ARCHS[arch_key]
    rows = []
    for batch in batches:
        for dev in ("cpu", device):
            model = build_model(architecture).to(dev)
            apply_state_dict(model, ckpt, map_location=dev)
            model.eval()
            cells = torch.randn(batch, 10, 21, 21, device=dev)
            glob = torch.randn(batch, 9, device=dev)
            hidden = model.initial_hidden(batch, device=torch.device(dev))
            # warmup
            with torch.no_grad():
                _ = _forward_result(model, cells, glob, hidden)
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            iters = 20
            with torch.no_grad():
                for _ in range(iters):
                    _ = _forward_result(model, cells, glob, hidden)
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            rows.append(
                {
                    "architecture": architecture,
                    "device": dev,
                    "batch": batch,
                    "iters": iters,
                    "seconds": elapsed,
                    "samples_per_second": (batch * iters) / max(elapsed, 1e-9),
                }
            )
    return {"rows": rows}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "ppo_probe", "benchmark", "all"], default="all")
    args = p.parse_args()

    device = resolve_training_device("auto", context="check_cuda_training")
    created = datetime.now(timezone.utc).isoformat()
    snap = cuda_runtime_snapshot()

    smoke_reports = {}
    ppo_reports = {}
    bench = {}

    if args.mode in {"smoke", "all"}:
        for key in ARCHS:
            smoke_reports[key] = smoke_architecture(key, device)

    if args.mode in {"ppo_probe", "all"}:
        for key in ARCHS:
            ppo_reports[key] = ppo_update_probe(key, device)

    if args.mode in {"benchmark", "all"}:
        for key in ARCHS:
            bench[key] = cpu_gpu_benchmark(key, device)

    smoke_gate = "PASS" if smoke_reports and all(v.get("gate") == "PASS" for v in smoke_reports.values()) else (
        "SKIP" if not smoke_reports else "FAIL"
    )
    exec_gate = smoke_gate
    ppo_gate = "PASS" if ppo_reports and all(v.get("gate") == "PASS" for v in ppo_reports.values()) else (
        "SKIP" if not ppo_reports else "FAIL"
    )

    report = {
        "schema_version": 1,
        "kind": "CUDA_TRAINING_GATES",
        "created_at": created,
        "device": device,
        "torch": snap,
        "smoke": smoke_reports,
        "ppo_update_probe": ppo_reports,
        "benchmark": bench,
        "CUDA_TRAINING_SMOKE_GATE": smoke_gate,
        "GPU_EXECUTION_GATE": exec_gate,
        "PPO_UPDATE_EFFECTIVENESS_GATE": ppo_gate,
    }
    out = MANIFESTS / "cuda_training_gates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Human benchmark summary
    if bench:
        lines = ["# GPU training benchmark (Phase 9E)", "", f"device: `{device}`", ""]
        lines.append("| arch | device | batch | samples/s |")
        lines.append("| --- | --- | ---: | ---: |")
        for key, payload in bench.items():
            for row in payload["rows"]:
                lines.append(
                    f"| {key} | {row['device']} | {row['batch']} | {row['samples_per_second']:.1f} |"
                )
        (REPO / "experiments/reports/gpu_training_benchmark.md").parent.mkdir(parents=True, exist_ok=True)
        (REPO / "experiments/reports/gpu_training_benchmark.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    print(
        json.dumps(
            {
                "CUDA_TRAINING_SMOKE_GATE": smoke_gate,
                "GPU_EXECUTION_GATE": exec_gate,
                "PPO_UPDATE_EFFECTIVENESS_GATE": ppo_gate,
                "out": str(out),
            },
            indent=2,
        )
    )
    if "FAIL" in {smoke_gate, exec_gate, ppo_gate}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
