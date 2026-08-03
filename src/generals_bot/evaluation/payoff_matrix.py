"""Payoff matrix helpers."""

from __future__ import annotations

from typing import Any


def empty_payoff(labels: list[str]) -> dict[str, Any]:
    n = len(labels)
    return {
        "schema_version": 1,
        "labels": labels,
        "matrix": [[0.0 for _ in range(n)] for _ in range(n)],
        "counts": [[0 for _ in range(n)] for _ in range(n)],
    }


def add_result(
    payoff: dict[str, Any],
    row_label: str,
    col_label: str,
    score_for_row: float,
) -> None:
    labels: list[str] = payoff["labels"]
    i = labels.index(row_label)
    j = labels.index(col_label)
    n = payoff["counts"][i][j]
    old = payoff["matrix"][i][j]
    payoff["matrix"][i][j] = (old * n + score_for_row) / (n + 1)
    payoff["counts"][i][j] = n + 1
