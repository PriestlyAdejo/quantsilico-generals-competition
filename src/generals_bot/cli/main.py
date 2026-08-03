"""CLI entrypoint for generals_bot."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _doctor_inline(_: argparse.Namespace) -> int:
    import subprocess

    py = sys.executable
    checks = [
        ["scripts/dev/verify_competition_environment.py"],
        ["scripts/dev/verify_repository.py"],
    ]
    rc = 0
    for args in checks:
        path = REPO_ROOT / args[0]
        if not path.exists():
            print(f"missing {path}")
            rc = 1
            continue
        result = subprocess.run([py, str(path)], cwd=str(REPO_ROOT))
        if result.returncode != 0:
            rc = result.returncode
    return rc


def cmd_match(args: argparse.Namespace) -> int:
    from generals_bot.evaluation.match import run_python_agent_match

    baselines = {
        "pass": REPO_ROOT / "baselines" / "pass_bot" / "main.py",
        "pass_bot": REPO_ROOT / "baselines" / "pass_bot" / "main.py",
        "legal_random": REPO_ROOT / "baselines" / "legal_random" / "main.py",
        "heuristic_v0": REPO_ROOT / "baselines" / "heuristic_v0" / "main.py",
        "heuristic_v1": REPO_ROOT / "baselines" / "heuristic_v1" / "main.py",
        "heuristic_aggressive": REPO_ROOT / "baselines" / "heuristic_aggressive" / "main.py",
        "heuristic_defensive": REPO_ROOT / "baselines" / "heuristic_defensive" / "main.py",
        "heuristic_castle": REPO_ROOT / "baselines" / "heuristic_castle" / "main.py",
        "heuristic_deathtouch": REPO_ROOT / "baselines" / "heuristic_deathtouch" / "main.py",
        "expander": (
            REPO_ROOT
            / "third_party"
            / "generals-bots"
            / "competition"
            / "agents"
            / "expander_python"
            / "main.py"
        ),
    }
    candidate = baselines.get(args.candidate)
    opponent = baselines.get(args.opponent)
    if candidate is None or opponent is None:
        print(f"unknown candidate/opponent. known: {sorted(baselines)}", file=sys.stderr)
        return 2
    result = run_python_agent_match(
        candidate,
        opponent,
        seed=args.seed,
        mode="competition",
        max_turns=args.max_turns,
    )
    payload = {
        "winner": result.winner,
        "turns": result.turns,
        "seed": result.seed,
        "faults0": result.faults0,
        "faults1": result.faults1,
        "elapsed_s": result.elapsed_s,
        "truncated": result.truncated,
        "candidate": args.candidate,
        "opponent": args.opponent,
    }
    if args.record_replay:
        out = (
            REPO_ROOT
            / "replays"
            / "private"
            / f"match_s{args.seed}_{args.candidate}_vs_{args.opponent}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        payload["replay_path"] = str(out)
    print(json.dumps(payload, indent=2))
    return 0 if result.faults0 == 0 and result.faults1 == 0 else 1


def cmd_submission_build(args: argparse.Namespace) -> int:
    from generals_bot.submission.builder import build_heuristic_package

    report = build_heuristic_package(args.candidate)
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.status == "PACKAGED" else 1


def cmd_submission_validate(args: argparse.Namespace) -> int:
    from generals_bot.submission.builder import validate_package

    report = validate_package(Path(args.package))
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.status == "PACKAGED" else 1


def cmd_qualification(args: argparse.Namespace) -> int:
    from generals_bot.evaluation.qualification_runner import run_qualification_suite

    out = Path(args.out) if args.out else (
        REPO_ROOT / "experiments" / "manifests" / f"{args.preset}.json"
    )
    report = run_qualification_suite(
        policies=args.policies,
        preset_name=args.preset,
        out_path=out,
        wall_clock_s=args.wall_clock_s,
    )
    print(json.dumps({k: report[k] for k in report if k != "games"}, indent=2))
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generals_bot")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="verify local environments")
    p_doctor.set_defaults(func=_doctor_inline)

    p_match = sub.add_parser("match", help="run a local competition-mode match")
    p_match.add_argument("--candidate", required=True)
    p_match.add_argument("--opponent", required=True)
    p_match.add_argument("--seed", type=int, default=0)
    p_match.add_argument("--max-turns", type=int, default=None)
    p_match.add_argument("--record-replay", action="store_true")
    p_match.add_argument("--trace", default="none")
    p_match.set_defaults(func=cmd_match)

    p_sub = sub.add_parser("submission", help="build/validate submission packages")
    sub_sub = p_sub.add_subparsers(dest="submission_command", required=True)
    p_build = sub_sub.add_parser("build")
    p_build.add_argument("--candidate", default="heuristic_v0")
    p_build.set_defaults(func=cmd_submission_build)
    p_val = sub_sub.add_parser("validate")
    p_val.add_argument("--package", required=True)
    p_val.set_defaults(func=cmd_submission_validate)

    p_qual = sub.add_parser("qualification", help="run Expander W/D/L qualification suite")
    p_qual.add_argument("--preset", default="qualification_smoke")
    p_qual.add_argument(
        "--policies",
        nargs="+",
        default=["heuristic_v2_qualifier", "heuristic_v1", "heuristic_v0"],
    )
    p_qual.add_argument("--out", default="")
    p_qual.add_argument("--wall-clock-s", type=float, default=None)
    p_qual.set_defaults(func=cmd_qualification)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
