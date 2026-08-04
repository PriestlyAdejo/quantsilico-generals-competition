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
    from generals_bot.selector import list_policies

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
    policy_names = set(list_policies()) | set(baselines) | {"hunter", "official_hunter", "official_expander"}
    candidate_path = baselines.get(args.candidate)
    opponent_path = baselines.get(args.opponent)

    # Prefer subprocess baseline runners when both sides have packaged mains.
    if candidate_path is not None and opponent_path is not None:
        result = run_python_agent_match(
            candidate_path,
            opponent_path,
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
            "runner": "baseline_subprocess",
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
            payload["replay_id"] = out.stem
        print(json.dumps(payload, indent=2))
        return 0 if result.faults0 == 0 and result.faults1 == 0 else 1

    if args.candidate not in policy_names or args.opponent not in policy_names:
        print(
            f"unknown candidate/opponent. known: {sorted(policy_names)}",
            file=sys.stderr,
        )
        return 2

    # In-process selector policies (heuristic_v2f*, official_*, etc.).
    from generals_bot.evaluation.population import _play_inprocess

    max_turns = int(args.max_turns) if args.max_turns is not None else 1200
    rec = _play_inprocess(
        args.candidate,
        args.opponent,
        seed=int(args.seed),
        swap=False,
        max_turns=max_turns,
        bot_commit="",
    )
    winner = rec.winner if rec.winner is not None else -1
    payload = {
        "winner": winner,
        "turns": max_turns if winner is None or winner < 0 else max_turns,
        "seed": int(args.seed),
        "faults0": int(rec.protocol_faults),
        "faults1": 0,
        "elapsed_s": sum(rec.latency_ms) / 1000.0 if rec.latency_ms else 0.0,
        "truncated": winner is None or winner < 0,
        "candidate": args.candidate,
        "opponent": args.opponent,
        "runner": "selector_inprocess",
        "wdl": {"wins": rec.wins, "draws": rec.draws, "losses": rec.losses},
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
        payload["replay_id"] = out.stem
        payload["replay_note"] = "Metadata-only replay; board frames not recorded by in-process runner."
    print(json.dumps(payload, indent=2))
    return 0


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
