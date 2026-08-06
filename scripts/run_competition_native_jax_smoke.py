"""Bounded smoke runner for competition_native_jax."""
from __future__ import annotations

from pathlib import Path

from train.competition_native_jax.train_loop import TrainConfig, run_selfplay_budget


def main() -> None:
    out = Path("experiments/competition_native_jax/smoke")
    smoke = run_selfplay_budget(
        TrainConfig(name="smoke", max_transitions=120, max_updates=2, wall_seconds=240, seed=1),
        out,
    )
    print("DONE", smoke["status"], smoke["transitions"], smoke["games"], smoke["measured_tps"], flush=True)


if __name__ == "__main__":
    main()
