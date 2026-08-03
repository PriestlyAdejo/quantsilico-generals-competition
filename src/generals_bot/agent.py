"""Stdio competition agent loop."""

from __future__ import annotations

import sys
import time
from typing import TextIO

from generals_bot.observation import GameContext
from generals_bot.policies.base import Policy, TraceLevel
from generals_bot.policies.pass_policy import PassPolicy
from generals_bot.protocol import (
    parse_handshake,
    parse_observation_frame,
    serialize_action,
)
from generals_bot.rules import ACTION_DEADLINE_S, FIRST_ACTION_DEADLINE_S


def run_agent(
    policy: Policy,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    deterministic: bool = True,
    trace: TraceLevel = TraceLevel.NONE,
) -> None:
    """Read protocol frames and write one flushed action per observation."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    handshake = stdin.readline()
    if not handshake:
        return
    player_id, height, width = parse_handshake(handshake)
    context = GameContext(player_id=player_id, height=height, width=width)
    state = policy.initial_state(context)
    first = True

    while True:
        scalars = stdin.readline()
        if not scalars:
            return
        grid_lines = [stdin.readline() for _ in range(3 * height)]
        if any(line == "" for line in grid_lines):
            print("incomplete observation frame; exiting", file=stderr)
            return
        observation = parse_observation_frame(
            scalars, grid_lines, height=height, width=width
        )
        deadline = time.perf_counter() + (
            FIRST_ACTION_DEADLINE_S if first else ACTION_DEADLINE_S
        )
        first = False
        try:
            decision = policy.act(
                observation,
                state,
                deterministic=deterministic,
                trace=trace,
                deadline=deadline,
            )
            state = decision.new_state
            line = serialize_action(decision.action)
        except Exception as exc:  # noqa: BLE001 - never crash the match process
            print(f"policy error: {type(exc).__name__}: {exc}", file=stderr)
            from generals_bot.protocol import pass_line

            line = pass_line()

        stdout.write(line + "\n")
        stdout.flush()


def main() -> None:
    run_agent(PassPolicy())


if __name__ == "__main__":
    main()
