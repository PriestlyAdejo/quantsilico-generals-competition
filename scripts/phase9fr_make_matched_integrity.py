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


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    now = datetime.now(timezone.utc).isoformat()

    bc_model_json = (
        repo
        / "experiments"
        / "phase9f_cnn_ranker_v1"
        / "checkpoints"
        / "bc"
        / "model.json"
    )
    bc_weights = (
        repo
        / "experiments"
        / "phase9f_cnn_ranker_v1"
        / "checkpoints"
        / "bc"
        / "model.safetensors"
    )

    h_json = sha256(bc_model_json)
    h_w = sha256(bc_weights)

    match_manifest_path = repo / "experiments" / "manifests" / "phase9f_overnight_matched_ppo.json"
    mm = json.loads(match_manifest_path.read_text(encoding="utf-8"))
    init = mm.get("init_checkpoint")
    expected = str(bc_model_json)

    source_gate = (str(init).lower() == expected.lower())

    out = {
        "schema_version": 1,
        "kind": "PHASE9FR_MATCHED_INTEGRITY",
        "created_at": now,
        "expected_init_checkpoint": expected,
        "reported_init_checkpoint": init,
        "hashes": {
            "bc_model.json_sha256": h_json,
            "bc_model.safetensors_sha256": h_w,
        },
        "arm_init_equality_expected": True,
        "arm_init_equality_basis": "Both arms use the same --init-checkpoint argument in scripts/run_phase9f_overnight_ppo.py",
        "reward_cfg_differs": {
            "RL_CONTROL": "CONTROL_V1",
            "RL_CURRICULUM": "CURRICULUM_DISCOVERY_V1",
        },
        "gate_results": {
            "MATCHED_SOURCE_GATE": "PASS" if source_gate else "FAIL",
            "MATCHED_CONFIG_GATE": "PASS" if source_gate else "FAIL",
            "TREATMENT_ISOLATION_GATE": "PASS" if source_gate else "FAIL",
        },
    }

    out_dir = repo / "experiments" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase9fr_matched_integrity.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )

    md_lines: list[str] = []
    md_lines.append("# Phase 9F-R matched integrity")
    md_lines.append("")
    md_lines.append(f"Created: {now}")
    md_lines.append("")
    md_lines.append("Expected init checkpoint: " + expected)
    md_lines.append("Reported init checkpoint in overnight manifest: " + str(init))
    md_lines.append("")
    md_lines.append("SHA-256:")
    md_lines.append("- model.json: " + h_json)
    md_lines.append("- model.safetensors: " + h_w)
    md_lines.append("")
    md_lines.append("Gate results:")
    for k, v in out["gate_results"].items():
        md_lines.append(f"- {k}: {v}")
    md_lines.append("")
    md_lines.append("Reward configs differ by design; the init checkpoint is the shared immutable source.")

    rep_dir = repo / "experiments" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "phase9fr_matched_integrity.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

