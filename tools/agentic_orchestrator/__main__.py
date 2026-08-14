"""Command-line entrypoint for ``python -m tools.agentic_orchestrator``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .orchestrator import (
    AgentInvocationError,
    LiveAgentAdapter,
    Orchestrator,
    OrchestratorError,
    RuntimePaths,
)
from .schemas import State


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _orchestrator(args: argparse.Namespace) -> Orchestrator:
    repo = Path(args.repo).resolve()
    runtime = RuntimePaths(Path(args.runtime_dir).resolve())
    return Orchestrator(repo=repo, runtime=runtime)


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def command_status(args: argparse.Namespace) -> int:
    _print(_orchestrator(args).status())
    return 0


def command_tooling(args: argparse.Namespace) -> int:
    tooling = _orchestrator(args).tooling()
    _print(tooling)
    return 0 if all(item["STATUS"] == "READY" for item in tooling.values()) else 2


def command_dry_run(args: argparse.Namespace) -> int:
    dry_root = Path(args.runtime_dir).resolve() / "dry-run"
    orchestrator = Orchestrator(repo=Path(args.repo).resolve(), runtime=RuntimePaths(dry_root))
    state = orchestrator.status()
    if State(state["STATE"]) != State.IDLE:
        if State(state["STATE"]) in {
            State.ACCEPTED,
            State.BLOCKED,
            State.PAUSED_USAGE,
            State.PAUSED_HUMAN_BOUNDARY,
            State.FAILED,
        }:
            orchestrator.resume()
        else:
            raise OrchestratorError(
                f"dry-run runtime is mid-transition ({state['STATE']}); inspect before retry"
            )
    _print(orchestrator.dry_run())
    return 0


def command_pause(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    orchestrator.pause_human_boundary(args.reason)
    _print(orchestrator.status())
    return 0


def command_resume(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    orchestrator.resume()
    _print(orchestrator.status())
    return 0


def command_run(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    try:
        adapter = LiveAgentAdapter(
            repo=Path(args.repo).resolve(),
            runtime=orchestrator.runtime,
            requested_cursor_identity=args.cursor_model_display,
        )
    except OrchestratorError as exc:
        orchestrator.block(str(exc))
        _print(orchestrator.status())
        return 2

    completed = 0
    limit = 5 if args.until_human_boundary else args.max_tasks
    while completed < limit:
        if State(orchestrator.state["STATE"]) != State.IDLE:
            orchestrator.resume()
        try:
            result = orchestrator.run_once(adapter)
        except AgentInvocationError as exc:
            if exc.result.classification == "USAGE_EXHAUSTED":
                orchestrator.pause_usage(str(exc))
            else:
                orchestrator.block(str(exc))
            _print(orchestrator.status())
            return 2
        except (OrchestratorError, ValueError) as exc:
            try:
                orchestrator.block(str(exc))
            except OrchestratorError:
                pass
            _print(orchestrator.status())
            return 2
        completed += 1
        _print(result)
        state = State(result["STATE"])
        if state == State.PAUSED_HUMAN_BOUNDARY:
            return 3
        if state != State.ACCEPTED:
            return 2
        if args.once:
            break
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(_repo_root()))
    parser.add_argument("--runtime-dir", default=str(_repo_root() / "var" / "agentic"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="show durable supervisor state")
    status.set_defaults(func=command_status)

    tooling = subparsers.add_parser("tooling", help="probe CLI identity/auth/model availability")
    tooling.set_defaults(func=command_tooling)

    dry_run = subparsers.add_parser("dry-run", help="exercise deterministic workflow without edits")
    dry_run.set_defaults(func=command_dry_run)

    run = subparsers.add_parser("run", help="run bounded live tasks")
    run.add_argument(
        "--cursor-model-display",
        default="Grok 4.6",
        help="exact model display identity required from the installed Cursor CLI",
    )
    group = run.add_mutually_exclusive_group()
    group.add_argument("--once", action="store_true", help="run exactly one bounded task")
    group.add_argument("--max-tasks", type=int, default=1, help="maximum bounded tasks; default 1")
    group.add_argument(
        "--until-human-boundary",
        action="store_true",
        help="continue only until the explicit max-tasks ceiling or a human boundary",
    )
    run.set_defaults(func=command_run)

    pause = subparsers.add_parser("pause", help="persist an operator-requested safe pause")
    pause.add_argument("--reason", required=True)
    pause.set_defaults(func=command_pause)

    resume = subparsers.add_parser("resume", help="resume a terminal/pause state to IDLE")
    resume.set_defaults(func=command_resume)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "max_tasks", 1) < 1:
        parser.error("--max-tasks must be at least 1")
    try:
        return int(args.func(args))
    except (OrchestratorError, OSError, ValueError) as exc:
        print(f"orchestrator error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
