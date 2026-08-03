"""Factorised spatial action head for CNN/graph policies (schema v2).

Produces official flattened ACTION_DIM logits from shared spatial parameters:
  PASS | BUILD(source) | MOVE(source, direction, split)

The MLP control may continue using a flat Linear actor; label that limitation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from generals_bot.action import KIND_BUILD, KIND_MOVE, KIND_PASS, Action
from generals_bot.models.action_index import (
    ACTION_DIM,
    BUILD_OFFSET,
    MAX_HW,
    MOVE_COUNT,
    MOVE_OFFSET,
    NUM_DIRECTIONS,
    NUM_SPLITS,
    PASS_INDEX,
    action_to_index,
    build_index,
    index_to_action,
    move_index,
)


@dataclass
class FactorisedLossWeights:
    exact: float = 1.0
    source: float = 0.5
    action_type: float = 0.3
    direction: float = 0.3
    split: float = 0.2
    option: float = 0.5
    schema_version: int = 2


def decompose_action_index(index: int) -> dict[str, int]:
    """Map official flat index → factorised labels."""
    action = index_to_action(index)
    source = action.row * MAX_HW + action.col
    if action.kind == KIND_PASS:
        return {
            "action_type": 0,  # PASS
            "source": 0,
            "direction": 0,
            "split": 0,
            "is_build": 0,
            "is_move": 0,
            "is_pass": 1,
        }
    if action.kind == KIND_BUILD:
        return {
            "action_type": 2,  # BUILD
            "source": source,
            "direction": 0,
            "split": 0,
            "is_build": 1,
            "is_move": 0,
            "is_pass": 0,
        }
    return {
        "action_type": 1,  # MOVE
        "source": source,
        "direction": int(action.direction),
        "split": int(action.split),
        "is_build": 0,
        "is_move": 1,
        "is_pass": 0,
    }


def compose_action_index(
    *,
    action_type: int,
    source: int,
    direction: int = 0,
    split: int = 0,
) -> int:
    row, col = divmod(int(source), MAX_HW)
    if action_type == 0:
        return PASS_INDEX
    if action_type == 2:
        return build_index(row, col)
    return move_index(row, col, int(direction), int(split))


def roundtrip_ok(index: int) -> bool:
    parts = decompose_action_index(index)
    rebuilt = compose_action_index(
        action_type=parts["action_type"],
        source=parts["source"],
        direction=parts["direction"],
        split=parts["split"],
    )
    return rebuilt == index and index_to_action(rebuilt) == index_to_action(index)


class FactorisedActionHead(nn.Module):
    """Shared spatial → factorised → flattened ACTION_DIM logits."""

    def __init__(self, channels: int, context_dim: int) -> None:
        super().__init__()
        self.channels = channels
        # Fuse global/option context into spatial map via 1x1 bias.
        self.context_proj = nn.Linear(context_dim, channels)
        self.source_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.build_head = nn.Conv2d(channels, 1, kernel_size=1)
        # Direction logits: 4 maps from shared features (spatial reuse).
        self.direction_head = nn.Conv2d(channels, NUM_DIRECTIONS, kernel_size=1)
        # Split conditioned on direction via small MLP on concatenated feats.
        self.split_head = nn.Sequential(
            nn.Conv2d(channels + NUM_DIRECTIONS, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, NUM_DIRECTIONS * NUM_SPLITS, kernel_size=1),
        )
        self.pass_head = nn.Linear(context_dim, 1)

    def forward(self, spatial: Tensor, context: Tensor) -> dict[str, Tensor]:
        """
        spatial: (B, C, H, W) with H=W=MAX_HW
        context: (B, context_dim) — recurrent + option embedding
        """
        b, c, h, w = spatial.shape
        assert h == MAX_HW and w == MAX_HW
        ctx_map = self.context_proj(context).unsqueeze(-1).unsqueeze(-1)
        feats = spatial + ctx_map
        source_logits = self.source_head(feats).flatten(1)  # (B, H*W)
        build_logits = self.build_head(feats).flatten(1)  # (B, H*W)
        dir_maps = self.direction_head(feats)  # (B, 4, H, W)
        direction_logits = dir_maps.flatten(2).transpose(1, 2).reshape(b, -1)  # unused flat
        # Per-cell direction scores (B, H*W, 4)
        dir_per_cell = dir_maps.flatten(2).transpose(1, 2)
        split_in = torch.cat([feats, dir_maps], dim=1)
        split_maps = self.split_head(split_in)  # (B, 4*2, H, W)
        split_per = split_maps.flatten(2).transpose(1, 2).reshape(b, h * w, NUM_DIRECTIONS, NUM_SPLITS)
        pass_logit = self.pass_head(context).squeeze(-1)  # (B,)

        flat = self._assemble_flat(
            pass_logit=pass_logit,
            build_logits=build_logits,
            source_logits=source_logits,
            dir_per_cell=dir_per_cell,
            split_per=split_per,
        )
        return {
            "logits": flat,
            "source_logits": source_logits,
            "build_logits": build_logits,
            "direction_maps": dir_maps,
            "dir_per_cell": dir_per_cell,
            "split_per": split_per,
            "pass_logit": pass_logit,
            "action_type_logits": self._action_type_logits(pass_logit, source_logits, build_logits),
        }

    def _action_type_logits(
        self, pass_logit: Tensor, source_logits: Tensor, build_logits: Tensor
    ) -> Tensor:
        # Soft type scores for auxiliary loss: PASS, MOVE, BUILD
        move_score = source_logits.max(dim=-1).values
        build_score = build_logits.max(dim=-1).values
        return torch.stack([pass_logit, move_score, build_score], dim=-1)

    def _assemble_flat(
        self,
        *,
        pass_logit: Tensor,
        build_logits: Tensor,
        source_logits: Tensor,
        dir_per_cell: Tensor,
        split_per: Tensor,
    ) -> Tensor:
        """Combine factorised scores into ACTION_DIM using shared parameters."""
        b = pass_logit.shape[0]
        device = pass_logit.device
        flat = pass_logit.new_zeros(b, ACTION_DIM)
        flat[:, PASS_INDEX] = pass_logit
        flat[:, BUILD_OFFSET:MOVE_OFFSET] = build_logits
        # MOVE: source + direction + split (broadcast shared params)
        # move_index layout: for cell, for dir, for split
        # score = source[cell] + dir[cell,dir] + split[cell,dir,split]
        src = source_logits.view(b, MAX_HW * MAX_HW, 1, 1)
        direc = dir_per_cell.view(b, MAX_HW * MAX_HW, NUM_DIRECTIONS, 1)
        spl = split_per
        move_scores = (src + direc + spl).reshape(b, MOVE_COUNT)
        flat[:, MOVE_OFFSET:] = move_scores
        return flat


def factorised_bc_loss(
    out: dict[str, Tensor],
    targets: Tensor,
    options: Tensor | None,
    mixture_logits: Tensor | None,
    weights: FactorisedLossWeights | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Exact CE + factorised auxiliary CEs."""
    weights = weights or FactorisedLossWeights()
    logits = out["logits"]
    loss_exact = F.cross_entropy(logits, targets)
    # Build factorised targets
    parts = [decompose_action_index(int(t.item())) for t in targets.detach().cpu()]
    device = logits.device
    source_t = torch.tensor([p["source"] for p in parts], device=device, dtype=torch.long)
    type_t = torch.tensor([p["action_type"] for p in parts], device=device, dtype=torch.long)
    dir_t = torch.tensor([p["direction"] for p in parts], device=device, dtype=torch.long)
    split_t = torch.tensor([p["split"] for p in parts], device=device, dtype=torch.long)
    is_move = torch.tensor([p["is_move"] for p in parts], device=device, dtype=torch.bool)

    loss_source = F.cross_entropy(out["source_logits"], source_t)
    loss_type = F.cross_entropy(out["action_type_logits"], type_t)
    # Direction/split only on MOVE actions
    if bool(is_move.any()):
        dir_logits = out["dir_per_cell"][is_move, source_t[is_move], :]
        loss_dir = F.cross_entropy(dir_logits, dir_t[is_move])
        split_logits = out["split_per"][is_move, source_t[is_move], dir_t[is_move], :]
        loss_split = F.cross_entropy(split_logits, split_t[is_move])
    else:
        loss_dir = logits.new_zeros(())
        loss_split = logits.new_zeros(())

    loss_option = logits.new_zeros(())
    if options is not None and mixture_logits is not None:
        loss_option = F.cross_entropy(mixture_logits, options)

    total = (
        weights.exact * loss_exact
        + weights.source * loss_source
        + weights.action_type * loss_type
        + weights.direction * loss_dir
        + weights.split * loss_split
        + weights.option * loss_option
    )
    stats = {
        "loss_exact": float(loss_exact.item()),
        "loss_source": float(loss_source.item()),
        "loss_type": float(loss_type.item()),
        "loss_dir": float(loss_dir.item()) if loss_dir.ndim == 0 else float(loss_dir),
        "loss_split": float(loss_split.item()) if loss_split.ndim == 0 else float(loss_split),
        "loss_option": float(loss_option.item()) if loss_option.ndim == 0 else float(loss_option),
        "loss_total": float(total.item()),
    }
    return total, stats


@torch.inference_mode()
def component_metrics(logits: Tensor, targets: Tensor, legal_mask: Tensor | None = None) -> dict[str, float]:
    """Exact + factorised component accuracies."""
    if legal_mask is not None:
        masked = logits.masked_fill(~legal_mask, float("-inf"))
    else:
        masked = logits
    pred = masked.argmax(dim=-1)
    exact = float((pred == targets).float().mean().item())
    # top-k
    k3 = min(3, masked.shape[-1])
    k5 = min(5, masked.shape[-1])
    top3 = masked.topk(k3, dim=-1).indices
    top5 = masked.topk(k5, dim=-1).indices
    top3_acc = float((top3 == targets.unsqueeze(-1)).any(dim=-1).float().mean().item())
    top5_acc = float((top5 == targets.unsqueeze(-1)).any(dim=-1).float().mean().item())
    # mean rank of target
    order = masked.argsort(dim=-1, descending=True)
    ranks = (order == targets.unsqueeze(-1)).float().argmax(dim=-1).float()
    mean_rank = float(ranks.mean().item())

    parts_t = [decompose_action_index(int(t.item())) for t in targets.cpu()]
    parts_p = [decompose_action_index(int(t.item())) for t in pred.cpu()]
    source_acc = sum(a["source"] == b["source"] for a, b in zip(parts_t, parts_p)) / max(len(parts_t), 1)
    type_acc = sum(a["action_type"] == b["action_type"] for a, b in zip(parts_t, parts_p)) / max(len(parts_t), 1)
    move_pairs = [(a, b) for a, b in zip(parts_t, parts_p) if a["is_move"]]
    if move_pairs:
        dir_acc = sum(a["direction"] == b["direction"] for a, b in move_pairs) / len(move_pairs)
        split_acc = sum(a["split"] == b["split"] for a, b in move_pairs) / len(move_pairs)
    else:
        dir_acc = split_acc = float("nan")
    build_t = [a["is_build"] for a in parts_t]
    build_p = [a["is_build"] for a in parts_p]
    tp = sum(t and p for t, p in zip(build_t, build_p))
    fp = sum((not t) and p for t, p in zip(build_t, build_p))
    fn = sum(t and (not p) for t, p in zip(build_t, build_p))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    legal_rate = 1.0
    if legal_mask is not None:
        legal_rate = float(legal_mask.gather(1, pred.unsqueeze(1)).float().mean().item())
    return {
        "exact_action_acc": exact,
        "top3_exact_acc": top3_acc,
        "top5_exact_acc": top5_acc,
        "mean_selected_rank": mean_rank,
        "source_acc": float(source_acc),
        "action_type_acc": float(type_acc),
        "direction_acc": float(dir_acc),
        "split_acc": float(split_acc),
        "build_precision": float(prec),
        "build_recall": float(rec),
        "legal_action_rate": legal_rate,
    }
