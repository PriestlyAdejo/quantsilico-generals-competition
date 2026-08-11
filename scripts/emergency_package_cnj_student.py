"""Package QS-P9G-COMPETITION-EMERGENCY-BOOTSTRAP-V1 student emb96 ZIP (NumPy only)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "QS-P9G-COMPETITION-EMERGENCY-BOOTSTRAP-V1"

CNJ_INCLUDE = [
    "__init__.py",
    "constants.py",
    "action_codec.py",
    "patchify.py",
    "castles.py",
    "deathtouch.py",
    "obs_memory.py",
    "legal_mask.py",
    "transformer.py",
    "policy.py",
    "student_policy_numpy.py",
]

CORE_INCLUDE = [
    "__init__.py",
    "action.py",
    "observation.py",
    "protocol.py",
    "legal.py",
    "castle_cost.py",
    "rules.py",
    "agent.py",
]


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    with tmp.open("rb") as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    tmp.replace(path)


def _write_lf(path: Path, text: str) -> None:
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_student_package(weights_npz: Path, *, out_dir: Path | None = None) -> dict:
    from generals_bot.submission.builder import _zip_package_tree, promote_package_to_submission, validate_package

    weights_npz = Path(weights_npz)
    if not weights_npz.is_file():
        raise FileNotFoundError(weights_npz)

    staging = Path(tempfile.mkdtemp(prefix="cnj_student_", dir=str(ROOT / "submission" / "staging")))
    (ROOT / "submission" / "staging").mkdir(parents=True, exist_ok=True)
    pkg = staging / "pkg"
    pkg.mkdir(parents=True)

    # core generals_bot modules
    gb = pkg / "generals_bot"
    gb.mkdir()
    src_gb = ROOT / "src" / "generals_bot"
    for name in CORE_INCLUDE:
        shutil.copy2(src_gb / name, gb / name)
    # policies.base for agent Protocol types used transitively — copy minimal policies
    pol_dir = gb / "policies"
    pol_dir.mkdir()
    shutil.copy2(src_gb / "policies" / "__init__.py", pol_dir / "__init__.py")
    shutil.copy2(src_gb / "policies" / "base.py", pol_dir / "base.py")
    shutil.copy2(src_gb / "policies" / "pass_policy.py", pol_dir / "pass_policy.py")

    cnj = gb / "competition_native_jax"
    cnj.mkdir()
    src_cnj = src_gb / "competition_native_jax"
    for name in CNJ_INCLUDE:
        shutil.copy2(src_cnj / name, cnj / name)

    shutil.copy2(weights_npz, pkg / "weights.npz")

    _write_lf(
        pkg / "main.py",
        """from pathlib import Path
from generals_bot.agent import run_agent
from generals_bot.observation import GameContext
from generals_bot.policies.base import ActionDecision, PolicyState, TraceLevel
from generals_bot.competition_native_jax.policy import load_weights
from generals_bot.competition_native_jax.student_policy_numpy import StudentCompetitionNativePolicy


class StudentAgentPolicy:
    policy_id = "cnj_student_emb96_d2_h4"

    def __init__(self, weights_path: Path) -> None:
        self.inner = StudentCompetitionNativePolicy(weights=load_weights(weights_path), seed=0)

    def initial_state(self, context: GameContext) -> PolicyState:
        self.inner.reset(context.height, context.width)
        return PolicyState(data={"player_id": context.player_id})

    def act(self, observation, state, *, deterministic, trace, deadline):
        action, _info = self.inner.act(observation, deterministic=deterministic)
        return ActionDecision(action=action, new_state=state, policy_id=self.policy_id, strategic_option="LEARNED")


def main():
    w = Path(__file__).resolve().parent / "weights.npz"
    run_agent(StudentAgentPolicy(w), deterministic=True)


if __name__ == "__main__":
    main()
""",
    )
    _write_lf(pkg / "run.sh", "#!/usr/bin/env bash\nset -euo pipefail\nexec python -u main.py\n")
    (pkg / "run.sh").chmod((pkg / "run.sh").stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    _write_lf(
        pkg / "NOTICE.txt",
        f"QuantSilico Generals emergency learned student package.\n"
        f"candidate_id: {CANDIDATE_ID}\n"
        f"architecture: student_emb96_d2_h4\n"
        f"MANUAL_UPLOAD_ONLY\n",
    )
    _write_lf(
        pkg / "package_manifest.json",
        json.dumps(
            {
                "candidate_id": CANDIDATE_ID,
                "architecture": "student_emb96_d2_h4",
                "weights": "weights.npz",
                "built_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
    )

    zip_path = staging / "package.zip"
    _zip_package_tree(pkg, zip_path)
    structural = validate_package(zip_path)
    promoted = promote_package_to_submission(CANDIDATE_ID, zip_path)
    digest = _sha256(Path(promoted["package_path"]))
    report = {
        "schema_version": 1,
        "kind": "EMERGENCY_LEARNED_PACKAGE",
        "package_id": CANDIDATE_ID,
        "status_class": "EMERGENCY_LEARNED_PROVISIONAL_PACKAGE_EXISTS",
        "technical": "PENDING_QUALIFICATION",
        "competitive": "COMPETITIVELY_UNCONFIRMED",
        "upload": "MANUAL_UPLOAD_ONLY",
        "package_path": promoted["package_path"],
        "sha256": digest,
        "weights_source": str(weights_npz),
        "structural": structural.status,
        "immutable": True,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_learned_package_v1.json", report)
    return report


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    args = p.parse_args()
    report = build_student_package(Path(args.weights))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
