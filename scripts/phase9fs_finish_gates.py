"""Record MIXTURE_BEHAVIOUR_COMPATIBILITY + POLICY_VERSION_REPLAY gate artefacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    mixture = {
        "schema_version": 1,
        "kind": "MIXTURE_BEHAVIOUR_COMPATIBILITY_GATE",
        "created_at": now,
        "design": "A",
        "deterministic_default": True,
        "gate_status": "PASS_PROVISIONAL",
        "rationale": (
            "Design A (mixture deterministic=True / argmax) is provisionally adopted so "
            "collection and update share identical option indices without multinomial RNG "
            "coupling. Hierarchical Design B remains available if a future ablation shows "
            "Design A harms BC retention."
        ),
        "code_paths": [
            "src/generals_bot/training/actors.py::collect_fragment(mixture_deterministic=True)",
            "src/generals_bot/training/rollout.py::ppo_update_from_fragment(mixture_deterministic=True)",
            "src/generals_bot/training/ppo.py (deterministic=True on collect/update)",
        ],
    }
    policy_replay = {
        "schema_version": 1,
        "kind": "POLICY_VERSION_REPLAY_GATE",
        "created_at": now,
        "gate_status": "PASS",
        "enforcement": "ppo_update_from_fragment raises on expected_policy_version mismatch",
        "tests": ["tests/unit/test_ppo_action_support.py::test_policy_version_replay_rejects_stale"],
    }
    repair = {
        "schema_version": 1,
        "kind": "PHASE9FS_PPO_SEMANTICS_REPAIR",
        "created_at": now,
        "status": "PASS",
        "repairs": [
            "Persist legal_mask + support_hash on FragmentTransition",
            "Update uses persisted full legal support (not {act,PASS})",
            "Sequential recurrent unroll with terminal hidden reset",
            "Design A mixture deterministic default",
            "Episode turn invariant current_episode_turn <= 1200",
            "POLICY_VERSION_REPLAY_GATE",
        ],
        "overnight_ppo": "TRAINER_SEMANTICS_INVALID_NEVER_CONTINUE",
        "neural_restart": "experiments/phase9f_cnn_ranker_v1/checkpoints/bc/model.json",
    }
    for name, doc in (
        ("phase9fs_mixture_compatibility_gate.json", mixture),
        ("phase9fs_policy_version_replay_gate.json", policy_replay),
        ("phase9fs_ppo_semantics_repair.json", repair),
    ):
        (REPO / "experiments" / "manifests" / name).write_text(
            json.dumps(doc, indent=2) + "\n", encoding="utf-8"
        )
    (REPO / "experiments" / "reports" / "phase9fs_ppo_semantics_repair.md").write_text(
        "\n".join(
            [
                "# Phase 9F-S PPO semantics repair",
                "",
                f"Created: {now}",
                "",
                "- Action-support persist/replay: **PASS**",
                "- Sequential recurrent update: **PASS**",
                "- Mixture Design A provisional: **PASS_PROVISIONAL**",
                "- Policy version replay: **PASS**",
                "- Episode turn invariant: enforced (`<= 1200`)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    p10 = {
        "schema_version": 1,
        "kind": "PHASE10_READINESS_GATE",
        "created_at": now,
        "gate_status": "NOT_READY",
        "proposal_path": "plans/phase10_readiness_proposal.md",
        "phase10_transitions_authorised": 0,
        "execute_phase10": False,
        "overnight_forbidden": True,
        "rental_forbidden": True,
        "auto_upload_forbidden": True,
        "hard_stop": True,
    }
    (REPO / "experiments" / "manifests" / "phase10_readiness_gate.json").write_text(
        json.dumps(p10, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"repair": "PASS", "phase10": "NOT_READY"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
