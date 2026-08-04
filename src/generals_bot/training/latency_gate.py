"""Competition-size CPU latency gate for CNN and pure-Torch graph policies."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from generals_bot.models.factory import build_model
from generals_bot.models.legal_mask import apply_action_mask, legal_mask_observation
from generals_bot.models.observation_encoder import encode_globals, encode_observation
from generals_bot.observation import Observation

REPO = Path(__file__).resolve().parents[3]
BOARD_SIZES = [(18, 18), (18, 21), (21, 18), (21, 21)]
ARCHITECTURES = ("recurrent_cnn_v2", "recurrent_graph_belief_v2")


@dataclass
class LatencyStats:
    p50: float
    p95: float
    p99: float
    maximum: float
    first_ms: float
    n: int


@dataclass
class GateResult:
    schema_version: int = 1
    kind: str = "COMPETITION_SIZE_LATENCY_GATE"
    classification: dict[str, str] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _obs(h: int, w: int, *, dense: bool = False, turn: int = 400) -> Observation:
    type_grid = []
    owner_grid = []
    army_grid = []
    for r in range(h):
        tr, ow, ar = [], [], []
        for c in range(w):
            if r == 0 and c == 0:
                tr.append(4)
                ow.append(1)
                ar.append(40 if dense else 8)
            elif dense and (r + c) % 5 == 0:
                tr.append(2)
                ow.append(1 if (r + c) % 10 == 0 else 2)
                ar.append(12)
            elif (r + c) % 7 == 0:
                tr.append(0)  # mountain
                ow.append(0)
                ar.append(0)
            else:
                tr.append(1)
                ow.append(1 if r < h // 3 and c < w // 3 else (2 if dense and r > 2 * h // 3 else 0))
                ar.append(3 if ow[-1] else 0)
        type_grid.append(tuple(tr))
        owner_grid.append(tuple(ow))
        army_grid.append(tuple(ar))
    return Observation(
        height=h,
        width=w,
        turn=turn,
        my_land=sum(1 for row in owner_grid for v in row if v == 1),
        my_army=sum(army_grid[r][c] for r in range(h) for c in range(w) if owner_grid[r][c] == 1),
        opp_land=sum(1 for row in owner_grid for v in row if v == 2),
        opp_army=sum(army_grid[r][c] for r in range(h) for c in range(w) if owner_grid[r][c] == 2),
        type_grid=tuple(type_grid),
        owner_grid=tuple(owner_grid),
        army_grid=tuple(army_grid),
    )


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _forward(model: torch.nn.Module, obs: Observation, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
    device = hidden.device
    cells = encode_observation(obs, device=device)
    if cells.ndim == 3:
        cells = cells.unsqueeze(0)
    if not hasattr(model, "stem") and not hasattr(model, "input_proj"):
        cells = cells.reshape(cells.shape[0], -1)
    glob = encode_globals(obs, device=device).unsqueeze(0)
    kwargs: dict[str, Any] = {}
    if hasattr(model, "initial_cell_memory"):
        kwargs["cell_memory"] = model.initial_cell_memory(1, device=device)
    out = model.forward_tensors(cells, glob, hidden, deterministic=True, **kwargs)
    mask = legal_mask_observation(obs, device=device).unsqueeze(0)
    out = dict(out)
    out["logits"] = apply_action_mask(out["logits"], mask)
    return out


def _bench_one(architecture: str, h: int, w: int, *, dense: bool, repeats: int = 40, warmup: int = 8) -> dict[str, Any]:
    # Soft one-core preference (best-effort on Windows).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    model = build_model(architecture).eval()
    device = torch.device("cpu")
    model.to(device)
    obs = _obs(h, w, dense=dense)
    hidden = model.initial_hidden(batch=1, device=device)

    # Cold first inference
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = _forward(model, obs, hidden)
    first_ms = (time.perf_counter() - t0) * 1000.0

    with torch.inference_mode():
        for _ in range(warmup):
            out = _forward(model, obs, hidden)
            hidden = out["hidden"]

    samples: list[float] = []
    hidden = model.initial_hidden(batch=1, device=device)
    with torch.inference_mode():
        for _ in range(repeats):
            t1 = time.perf_counter()
            out = _forward(model, obs, hidden)
            hidden = out["hidden"]
            samples.append((time.perf_counter() - t1) * 1000.0)
    samples_sorted = sorted(samples)
    stats = LatencyStats(
        p50=_percentile(samples_sorted, 0.50),
        p95=_percentile(samples_sorted, 0.95),
        p99=_percentile(samples_sorted, 0.99),
        maximum=max(samples_sorted),
        first_ms=first_ms,
        n=len(samples_sorted),
    )
    return {
        "board": f"{h}x{w}",
        "dense": dense,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "stats_ms": asdict(stats),
    }


def classify(stats_list: list[dict[str, Any]]) -> str:
    worst_p99 = max(s["stats_ms"]["p99"] for s in stats_list)
    worst_max = max(s["stats_ms"]["maximum"] for s in stats_list)
    if worst_p99 <= 120.0 and worst_max < 150.0:
        return "PASS"
    if worst_p99 <= 140.0 and worst_max < 150.0:
        return "PARTIAL"
    return "FAIL"


def run_latency_gate(*, repeats: int = 40) -> GateResult:
    result = GateResult(
        notes=[
            "Internal safety gate (p99<=120 PASS; <=140 PARTIAL; else FAIL).",
            "Blank 8x8 ~139ms graph smoke is NOT this gate.",
            "One-thread best-effort; dashboard should be stopped during measurement.",
        ]
    )
    for arch in ARCHITECTURES:
        arch_rows: list[dict[str, Any]] = []
        for h, w in BOARD_SIZES:
            for dense in (False, True):
                arch_rows.append(_bench_one(arch, h, w, dense=dense, repeats=repeats))
        result.results[arch] = arch_rows
        result.classification[arch] = classify(arch_rows)
    return result


def main() -> None:
    gate = run_latency_gate()
    out = REPO / "experiments" / "manifests" / "competition_size_latency_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gate.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"classification": gate.classification, "path": str(out)}, indent=2))


if __name__ == "__main__":
    main()
