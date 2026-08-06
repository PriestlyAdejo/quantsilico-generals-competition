"""Stdio runtime for competition-native NumPy policy (deployment path)."""

from __future__ import annotations

import sys
from pathlib import Path

from generals_bot.competition_native_jax.policy import CompetitionNativePolicy, load_weights
from generals_bot.protocol import parse_handshake, parse_observation_frame, serialize_action


def run_stdio(weights_path: Path | None = None) -> None:
    """Minimal stdio loop: handshake, then observation → action lines."""
    weights = load_weights(weights_path) if weights_path else None
    policy = CompetitionNativePolicy(weights=weights, seed=0)
    handshake = sys.stdin.readline()
    if not handshake:
        return
    player_index, height, width = parse_handshake(handshake.strip())
    policy.reset(height, width)
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if line in ("", "end"):
            break
        obs = parse_observation_frame(line)
        action, _ = policy.act(obs, deterministic=True)
        sys.stdout.write(serialize_action(action) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_stdio(path)
