"""Behaviour cloning trainer for recurrent MLP / CNN / graph policies."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from generals_bot.models.checkpoint import save_checkpoint
from generals_bot.models.cnn import RecurrentCNNPolicy
from generals_bot.models.factory import build_model
from generals_bot.models.factorised_action import FactorisedLossWeights, component_metrics, factorised_bc_loss
from generals_bot.models.graph import RecurrentGraphBeliefPolicy
from generals_bot.models.legal_mask import apply_action_mask
from generals_bot.models.mlp import RecurrentMLPPolicy
from generals_bot.training.collect_bc import collect_trajectories, load_dataset, save_dataset

DEFAULT_POLICIES = [
    "heuristic_v0",
    "heuristic_v1",
    "heuristic_aggressive",
    "heuristic_defensive",
    "heuristic_castle",
    "heuristic_deathtouch",
]


def _read_seeds(path: Path, limit: int | None = None) -> list[int]:
    seeds = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        seeds = seeds[:limit]
    return seeds


def _batches(n: int, batch_size: int):
    idx = torch.randperm(n)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def train_bc(
    *,
    architecture: str,
    train_path: Path,
    val_path: Path | None = None,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    model = build_model(architecture).to(torch_device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train = load_dataset(train_path)
    for k, v in train.items():
        train[k] = v.to(torch_device)
    val = load_dataset(val_path) if val_path and val_path.is_file() else None
    if val is not None:
        for k, v in val.items():
            val[k] = v.to(torch_device)

    history: list[dict[str, float]] = []
    n = train["cells"].shape[0]
    t0 = time.perf_counter()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        seen = 0
        for batch_idx in _batches(n, batch_size):
            cells = train["cells"][batch_idx]
            globs = train["globals"][batch_idx]
            targets = train["action_index"][batch_idx]
            options = train["option_index"][batch_idx]
            masks = train["legal_mask"][batch_idx]
            b = cells.shape[0]
            if isinstance(model, RecurrentGraphBeliefPolicy):
                hidden = model.initial_hidden(b, device=torch_device)
                cell_mem = model.initial_cell_memory(b, device=torch_device)
                out = model.forward_tensors(
                    cells,
                    globs,
                    hidden,
                    cell_mem,
                    deterministic=True,
                    teacher_option=options,
                )
            else:
                hidden = model.initial_hidden(b, device=torch_device)
                flat = cells if not isinstance(model, RecurrentMLPPolicy) else cells.reshape(b, -1)
                if isinstance(model, RecurrentCNNPolicy):
                    flat = cells
                out = model.forward_tensors(
                    flat, globs, hidden, deterministic=True, teacher_option=options
                )
            # Joint action + option imitation with legal masking.
            masked_logits = apply_action_mask(out["logits"], masks)
            use_factorised = bool(getattr(model, "factorised", False)) and "source_logits" in out
            if use_factorised:
                # Keep mask on exact head by writing -inf into out copy for CE
                out_loss = dict(out)
                out_loss["logits"] = masked_logits
                loss, loss_stats = factorised_bc_loss(
                    out_loss,
                    targets,
                    options,
                    out.get("mixture_logits"),
                    FactorisedLossWeights(),
                )
            else:
                loss = F.cross_entropy(masked_logits, targets) + 0.5 * F.cross_entropy(
                    out["mixture_logits"], options
                )
                loss_stats = {}
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * b
            pred = masked_logits.argmax(dim=-1)
            correct += int((pred == targets).sum().item())
            seen += b
        row = {
            "epoch": float(epoch),
            "train_loss": total_loss / max(seen, 1),
            "train_action_acc": correct / max(seen, 1),
            **{f"train_{k}": v for k, v in loss_stats.items()},
        }
        if val is not None:
            row.update(_eval_split(model, val, torch_device))
        history.append(row)

    elapsed = time.perf_counter() - t0
    out_dir = out_dir or Path("experiments/checkpoints/bc") / architecture
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model"
    save_checkpoint(
        model,
        ckpt,
        architecture=architecture,
        config=model.config_dict(),  # type: ignore[attr-defined]
    )
    report = {
        "architecture": architecture,
        "device": device,
        "epochs": epochs,
        "train_n": n,
        "val_n": int(val["cells"].shape[0]) if val is not None else 0,
        "history": history,
        "elapsed_s": elapsed,
        "checkpoint": str(ckpt.with_suffix(".json")),
        "parameter_count": model.parameter_count(),  # type: ignore[attr-defined]
        "final_train_action_acc": history[-1]["train_action_acc"] if history else 0.0,
        "final_val_action_acc": history[-1].get("val_action_acc") if history else None,
    }
    (out_dir / "bc_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


@torch.inference_mode()
def _eval_split(model: nn.Module, data: dict[str, torch.Tensor], device: torch.device) -> dict[str, float]:
    model.eval()
    n = data["cells"].shape[0]
    correct = 0
    option_correct = 0
    comp_sums: dict[str, float] = {}
    comp_counts: dict[str, int] = {}
    for start in range(0, n, 64):
        sl = slice(start, start + 64)
        cells = data["cells"][sl]
        globs = data["globals"][sl]
        targets = data["action_index"][sl]
        options = data["option_index"][sl]
        masks = data["legal_mask"][sl]
        b = cells.shape[0]
        if isinstance(model, RecurrentGraphBeliefPolicy):
            hidden = model.initial_hidden(b, device=device)
            cell_mem = model.initial_cell_memory(b, device=device)
            out = model.forward_tensors(
                cells, globs, hidden, cell_mem, deterministic=True, teacher_option=options
            )
        else:
            hidden = model.initial_hidden(b, device=device)
            flat = cells.reshape(b, -1) if isinstance(model, RecurrentMLPPolicy) else cells
            out = model.forward_tensors(
                flat, globs, hidden, deterministic=True, teacher_option=options
            )
        masked = apply_action_mask(out["logits"], masks)
        pred = masked.argmax(dim=-1)
        correct += int((pred == targets).sum().item())
        option_correct += int((out["option_index"] == options).sum().item())
        comps = component_metrics(out["logits"], targets, masks)
        for k, v in comps.items():
            if k not in comp_sums:
                comp_sums[k] = 0.0
            if v == v:  # not NaN
                comp_sums[k] += float(v) * b
                comp_counts[k] = comp_counts.get(k, 0) + b
    model.train()
    out_metrics = {
        "val_action_acc": correct / max(n, 1),
        "val_option_acc": option_correct / max(n, 1),
    }
    for k, total in comp_sums.items():
        out_metrics[f"val_{k}"] = total / max(comp_counts.get(k, 1), 1)
    return out_metrics


def run_bc_pipeline(
    *,
    tiny: bool = False,
    train_seed_limit: int = 8,
    val_seed_limit: int = 4,
    max_turns: int = 40,
    epochs: int = 8,
    architectures: list[str] | None = None,
) -> dict[str, Any]:
    train_seeds = _read_seeds(Path("experiments/seeds/train.txt"), 2 if tiny else train_seed_limit)
    val_seeds = _read_seeds(Path("experiments/seeds/validation.txt"), 1 if tiny else val_seed_limit)
    # Never touch promotion_holdout.txt
    policies = DEFAULT_POLICIES[:2] if tiny else DEFAULT_POLICIES
    train_samples = collect_trajectories(
        policies=policies, seeds=train_seeds, max_turns=20 if tiny else max_turns
    )
    val_samples = collect_trajectories(
        policies=policies, seeds=val_seeds, max_turns=20 if tiny else max_turns
    )
    data_root = Path("experiments/datasets/bc")
    train_path = save_dataset(train_samples, data_root / ("tiny_train.npz" if tiny else "smoke_train.npz"))
    val_path = save_dataset(val_samples, data_root / ("tiny_val.npz" if tiny else "smoke_val.npz"))

    architectures = architectures or [
        "recurrent_mlp_v1",
        "recurrent_cnn_v2",
        "recurrent_graph_belief_v2",
    ]
    reports = {}
    for arch in architectures:
        reports[arch] = train_bc(
            architecture=arch,
            train_path=train_path,
            val_path=val_path,
            epochs=epochs if epochs > 0 else (80 if tiny else 8),
            batch_size=8 if tiny else 32,
            lr=5e-3 if tiny else 1e-3,
            out_dir=Path("experiments/checkpoints/bc") / ("tiny_" + arch if tiny else arch),
        )
    summary = {
        "tiny": tiny,
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "policies": policies,
        "architectures": architectures,
        "reports": reports,
    }
    out = Path("experiments/manifests/bc_smoke.json" if not tiny else "experiments/manifests/bc_tiny.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["path"] = str(out)
    return summary


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--epochs", type=int, default=0, help="0 = default (80 tiny / 8 smoke)")
    p.add_argument("--train-seeds", type=int, default=8)
    p.add_argument(
        "--architectures",
        default="recurrent_mlp_v1,recurrent_cnn_v2,recurrent_graph_belief_v2",
        help="Comma-separated architecture ids",
    )
    args = p.parse_args()
    arches = [a.strip() for a in args.architectures.split(",") if a.strip()]
    summary = run_bc_pipeline(
        tiny=args.tiny,
        epochs=args.epochs,
        train_seed_limit=args.train_seeds,
        architectures=arches,
    )
    print(json.dumps({k: summary[k] for k in summary if k != "reports"}, indent=2))
    for arch, rep in summary["reports"].items():
        print(arch, "train_acc", rep["final_train_action_acc"], "val_acc", rep["final_val_action_acc"])


if __name__ == "__main__":
    main()
