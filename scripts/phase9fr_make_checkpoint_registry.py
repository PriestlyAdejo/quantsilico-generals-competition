from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_ckpt(reg: dict, *, candidate_id: str, key: str, ckpt_json: Path) -> None:
    ckpt_json = ckpt_json.resolve()
    if not ckpt_json.exists():
        raise FileNotFoundError(str(ckpt_json))
    weights_path = ckpt_json.with_suffix(".safetensors")
    reg[key] = {
        "candidate_id": candidate_id,
        "kind": "checkpoint",
        "paths": {
            "config_json": str(ckpt_json),
            "weights_safetensors": str(weights_path) if weights_path.exists() else None,
        },
        "hashes": {
            "config_json_sha256": sha256(ckpt_json),
            "weights_sha256": sha256(weights_path) if weights_path.exists() else None,
        },
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    now = datetime.now(timezone.utc).isoformat()

    control_final = repo / "experiments" / "phase9f_overnight_ppo" / "rl_control" / "final.json"
    curriculum_final = repo / "experiments" / "phase9f_overnight_ppo" / "rl_curriculum" / "final.json"

    registry: dict = {
        "schema_version": 1,
        "created_at": now,
        "kind": "PHASE9FR_CHECKPOINT_REGISTRY",
        "checkpoints": {},
    }

    add_ckpt(
        registry["checkpoints"],
        candidate_id="QS-P9F-CNN-PPO-CONTROL-V1",
        key="QS-P9F-CNN-PPO-CONTROL-V1__final",
        ckpt_json=control_final,
    )
    add_ckpt(
        registry["checkpoints"],
        candidate_id="QS-P9F-CNN-PPO-CURRICULUM-V1",
        key="QS-P9F-CNN-PPO-CURRICULUM-V1__final",
        ckpt_json=curriculum_final,
    )

    out_path = repo / "experiments" / "manifests" / "phase9fr_checkpoint_registry.json"
    out_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    # Rename misleading role (do not change meaning of candidate_id, only the key/label).
    role_path = repo / "experiments" / "manifests" / "phase9f_role_registry.json"
    if role_path.exists():
        role_doc = json.loads(role_path.read_text(encoding="utf-8"))
        roles = role_doc.get("roles", {})
        if "best_learned_research" in roles:
            roles["best_neural_research_checkpoint"] = roles.pop("best_learned_research")
            roles["best_neural_research_checkpoint"]["label"] = "PROVISIONAL_BEST_NEURAL_RESEARCH_ONLY"
            roles["best_neural_research_checkpoint"]["note"] = (
                "Represents the selected research checkpoint after PPO update volume; "
                "does not satisfy strict Linux/CPU/tournament qualification."
            )
        role_doc["roles"] = roles
        role_path.write_text(json.dumps(role_doc, indent=2) + "\n", encoding="utf-8")

    # Simple report.
    rep_lines = []
    rep_lines.append("# Phase 9F-R checkpoint registry")
    rep_lines.append("")
    rep_lines.append(f"Created: {now}")
    rep_lines.append("")
    rep_lines.append("Stored only local (on-disk) checkpoint config + weights with SHA-256 hashes.")
    rep_lines.append("")
    rep_lines.append("Renamed role key `best_learned_research` -> `best_neural_research_checkpoint` (label clarified).")
    rep_path = repo / "experiments" / "reports" / "phase9fr_checkpoint_registry.md"
    rep_path.write_text("\n".join(rep_lines) + "\n", encoding="utf-8")

    print("written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

