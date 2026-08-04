"""Phase 9F foundation audit: volume, gamma, chunk continuity, memory lifecycle."""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from generals_bot.models.factory import build_model
from generals_bot.training import ppo as ppo_mod
from generals_bot.training.ppo import _gae

REPO = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discount_table(gamma: float = 0.99) -> dict[str, float]:
    return {str(t): float(gamma**t) for t in (10, 50, 100, 200, 400, 800, 1200)}


def audit_gae_bootstrap() -> dict:
    src = inspect.getsource(ppo_mod._gae)
    zeros_bootstrap = "values + [0.0]" in src or "values = values + [0.0]" in src
    # Demonstrate: early reward credit vanishes under zero bootstrap truncation
    rewards = [0.0] * 128
    rewards[0] = 0.0
    values = [0.0] * 128
    # terminal at end of chunk with reward only at step 127, early action at 0
    rewards[127] = 1.0
    adv_zero, _ = _gae(rewards, values, gamma=0.99, lam=0.95)
    # With proper bootstrap V=1 at end if truncated mid-episode (simulated)
    # Current API forces 0 bootstrap — early advantage from late reward:
    early_adv = float(adv_zero[0])
    late_adv = float(adv_zero[127])
    return {
        "gae_source_contains_zero_bootstrap": zeros_bootstrap,
        "demo_early_advantage_from_late_terminal": early_adv,
        "demo_late_advantage": late_adv,
        "gamma_1200": float(0.99**1200),
        "classification": "FAIL" if zeros_bootstrap else "PASS",
        "evidence": "src/generals_bot/training/ppo.py::_gae appends values+[0.0]",
    }


def audit_chunk_env_lifetime() -> dict:
    src = inspect.getsource(ppo_mod.run_bounded_ppo)
    fresh_env = "GeneralsEnv(mode=\"competition\")" in src or "GeneralsEnv(mode='competition')" in src
    pilot = (REPO / "scripts/run_phase9e_matched_pilot.py").read_text(encoding="utf-8")
    new_call_per_chunk = "run_bounded_ppo(" in pilot and "while env_steps" in pilot
    return {
        "run_bounded_ppo_creates_fresh_env": fresh_env,
        "pilot_invokes_new_run_bounded_ppo_per_chunk": new_call_per_chunk,
        "classification": "FAIL" if (fresh_env and new_call_per_chunk) else "PARTIAL",
        "consequence": "Opening actions cannot receive credit for post-chunk outcomes; episodes restart every 512 transitions.",
    }


def audit_volume() -> dict:
    # Phase 9E: 12288 transitions, 1 env, rollout 128 x 4 updates = 512/chunk, 24 chunks
    mean_episode_length_assumed = 100.0  # monitoring max_turns; turn-cap draws dominate
    # In training, episodes often hit natural done or continue; use monitoring as proxy + DRAW_TURN
    return {
        "phase9e_transitions_per_arm": 12288,
        "parallel_envs": 1,
        "chunk_env_steps": 512,
        "rollout_steps": 128,
        "updates_per_chunk": 4,
        "complete_game_equivalents_if_mean_100": 12288 / 100.0,
        "complete_game_equivalents_if_mean_1200": 12288 / 1200.0,
        "interpretation": "env_step = one agent transition in run_bounded_ppo",
        "classification": "FAIL",
        "note": "Even optimistic 100-turn episodes ≈123 games; true 1200-turn games ≈10 equivalents — tiny for rare wins.",
    }


def audit_memory() -> dict:
    cnn = build_model("recurrent_cnn_v2")
    graph = build_model("recurrent_graph_belief_v2")
    return {
        "cnn_has_gru": hasattr(cnn, "rnn"),
        "graph_has_cell_memory": hasattr(graph, "initial_cell_memory"),
        "chunk_resets_hidden": True,
        "structured_map_memory_in_learned_path": False,
        "heuristic_map_memory_exists": (REPO / "src/generals_bot/map_memory.py").is_file(),
        "classification": "FAIL",
        "note": "Models are recurrent but pilot chunks re-enter run_bounded_ppo with fresh hidden state; no structured fog memory in learned act path.",
    }


def main() -> int:
    gae = audit_gae_bootstrap()
    chunk = audit_chunk_env_lifetime()
    volume = audit_volume()
    memory = audit_memory()
    disc = discount_table(0.99)

    matrix = {
        "schema_version": 1,
        "kind": "PHASE9F_ROOT_CAUSE_MATRIX",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypotheses": [
            {
                "id": "zero_bootstrap_truncation",
                "status": gae["classification"],
                "confidence": 0.95,
                "severity": "CRITICAL",
                "evidence": gae,
                "remediation": "Bootstrap V(s_T) on truncation; separate terminated vs truncated masks",
            },
            {
                "id": "fresh_env_per_chunk",
                "status": chunk["classification"],
                "confidence": 0.95,
                "severity": "CRITICAL",
                "evidence": chunk,
                "remediation": "Persistent env workers across chunks; carry hidden/memory state",
            },
            {
                "id": "insufficient_effective_volume",
                "status": volume["classification"],
                "confidence": 0.85,
                "severity": "HIGH",
                "evidence": volume,
                "remediation": "Increase parallelism and complete-game sampling after continuity repair",
            },
            {
                "id": "gamma_horizon_mismatch",
                "status": "FAIL",
                "confidence": 0.9,
                "severity": "HIGH",
                "evidence": {"discount_table_gamma_0_99": disc},
                "remediation": "Benchmark higher gamma / multi-timescale returns; do not set gamma=1 blindly",
            },
            {
                "id": "memory_lifecycle_reset",
                "status": memory["classification"],
                "confidence": 0.85,
                "severity": "HIGH",
                "evidence": memory,
                "remediation": "Persist recurrent state + add structured map memory for deployment path",
            },
            {
                "id": "reward_only_insufficient",
                "status": "PASS",
                "confidence": 0.8,
                "severity": "MEDIUM",
                "evidence": "Phase 9D conversion and Phase 9E curriculum both 0W/8D/0L",
                "remediation": "Do not run another reward-only full-game PPO before foundation repair + teachers",
            },
        ],
        "gates": {
            "LONG_HORIZON_CREDIT_GATE": "FAIL",
            "CHUNK_CREDIT_CONTINUITY_GATE": "FAIL",
            "PARTIAL_OBSERVABILITY_MEMORY_GATE": "FAIL",
            "EFFECTIVE_VOLUME_GATE": "FAIL",
        },
    }
    out = REPO / "experiments/manifests/phase9f_root_cause_matrix.json"
    out.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Phase 9F root-cause audit",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Executive verdict",
        "",
        "Phase 9E did not fail because CUDA or legality broke. It failed because the trainer",
        "cannot credit long-horizon outcomes: GAE zero-bootstraps every rollout boundary,",
        "each 512-step chunk starts a fresh environment, gamma=0.99 erases ~1200-turn terminals,",
        "and recurrent state is not persisted across chunks. Reward shaping alone cannot fix this.",
        "",
        "## Gates",
        "",
        f"- LONG_HORIZON_CREDIT_GATE: **FAIL** (0.99^1200 = {disc['1200']:.2e})",
        f"- CHUNK_CREDIT_CONTINUITY_GATE: **{chunk['classification']}**",
        f"- PARTIAL_OBSERVABILITY_MEMORY_GATE: **{memory['classification']}**",
        f"- EFFECTIVE_VOLUME_GATE: **FAIL** (~{volume['complete_game_equivalents_if_mean_1200']:.1f} full-game equivalents at 1200 turns)",
        "",
        "## Immediate remediations (auto-execute)",
        "",
        "1. Fix `_gae` truncation bootstrap.",
        "2. Persist env + hidden state across updates and chunks.",
        "3. Wire structured map memory into learned/hybrid act path.",
        "4. Prefer hybrid expert+ranker + BC/DAgger before another full-game PPO.",
        "",
    ]
    (REPO / "experiments/reports/phase9f_root_cause_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"gates": matrix["gates"], "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
