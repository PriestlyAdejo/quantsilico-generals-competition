"""Evaluation metrics helpers."""

from __future__ import annotations

from statistics import median


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(latencies_ms, 50) if latencies_ms else None,
        "p95": percentile(latencies_ms, 95) if latencies_ms else None,
        "p99": percentile(latencies_ms, 99) if latencies_ms else None,
        "median": median(latencies_ms) if latencies_ms else None,
        "count": float(len(latencies_ms)),
    }
