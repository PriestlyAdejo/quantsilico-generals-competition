"""Normalize and diff frozen v2f reference against 544215e sources."""

from __future__ import annotations

import difflib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git_show(rev_path: str) -> str:
    return subprocess.check_output(
        ["git", "show", rev_path],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )


def norm_policy(s: str) -> str:
    return (
        s.replace("HeuristicV2FReferencePolicy", "HeuristicV2QualifierPolicy")
        .replace("phase_controller_v2f", "phase_controller")
        .replace("heuristic_v2f_best_reference", "heuristic_v2_qualifier")
        .replace("\r\n", "\n")
    )


def main() -> None:
    a = git_show("544215e:src/generals_bot/policies/heuristic_v2_qualifier.py")
    b = (ROOT / "src/generals_bot/policies/heuristic_v2f_reference.py").read_text(encoding="utf-8")
    na, nb = norm_policy(a).splitlines(), norm_policy(b).splitlines()
    diff = list(difflib.unified_diff(na, nb, fromfile="544215e", tofile="frozen", lineterm="", n=2))
    print("normalized policy diff lines", len(diff))
    print("\n".join(diff[:250] if diff else ["IDENTICAL"]))

    pa = git_show("544215e:src/generals_bot/policies/phase_controller.py")
    pb = (ROOT / "src/generals_bot/policies/phase_controller_v2f.py").read_text(encoding="utf-8")
    pa_n = pa.replace("\r\n", "\n")
    pb_n = (
        pb.replace(" frozen snapshot from commit 544215e (v2f reference)", "")
        .replace("\r\n", "\n")
    )
    # strip docstring-only first-line difference
    da = list(
        difflib.unified_diff(
            pa_n.splitlines(), pb_n.splitlines(), fromfile="pc544", tofile="pcv2f", lineterm="", n=1
        )
    )
    print("==== phase ====")
    print("diff lines", len(da))
    print("\n".join(da[:120] if da else ["IDENTICAL"]))

    # supporting modules changed since 544215e
    out = subprocess.check_output(
        [
            "git",
            "diff",
            "--stat",
            "544215e..HEAD",
            "--",
            "src/generals_bot/map_memory.py",
            "src/generals_bot/legal.py",
            "src/generals_bot/risk/",
            "src/generals_bot/observation.py",
            "src/generals_bot/policies/official_expander.py",
            "src/generals_bot/evaluation/",
            "src/generals_bot/engine/",
            "src/generals_bot/game/",
            "src/generals_bot/action.py",
            "src/generals_bot/protocol.py",
            "src/generals_bot/rules.py",
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    print("==== deps 544215e..HEAD ====")
    print(out or "(none)")


if __name__ == "__main__":
    main()
