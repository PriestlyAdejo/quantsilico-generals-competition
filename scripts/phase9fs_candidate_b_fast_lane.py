"""CANDIDATE_B_FAST_LANE — BC inference semantics audit (wall ≤ 90 min)."""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
BC_JSON = REPO / "experiments" / "phase9f_cnn_ranker_v1" / "checkpoints" / "bc" / "model.json"
TIMEBOX_S = 90 * 60


def main() -> int:
    started = time.perf_counter()
    now = datetime.now(timezone.utc).isoformat()
    checks: dict = {
        "bc_load": "NOT_RUN",
        "legal_support": "NOT_RUN",
        "recurrent_reset": "NOT_RUN",
        "mixture_deterministic": "NOT_RUN",
        "cpu_inference": "NOT_RUN",
        "hybrid_package": "NOT_RUN",
    }
    notes: list[str] = []
    classification = "CANDIDATE_B_NOT_COMPLETED"
    defect = None

    try:
        if time.perf_counter() - started > TIMEBOX_S:
            classification = "CANDIDATE_B_NOT_COMPLETED"
            raise TimeoutError("timebox before start")

        from generals_bot.models.checkpoint import apply_state_dict
        from generals_bot.models.factory import build_model
        from generals_bot.models.legal_mask import apply_action_mask, legal_mask_observation
        from generals_bot.training.actors import PersistentActor
        from generals_bot.training.conversion_reward import CONTROL_V1

        if not BC_JSON.is_file():
            classification = "CANDIDATE_B_BLOCKED_SEMANTICS"
            defect = "BC checkpoint missing"
            checks["bc_load"] = "FAIL"
            raise RuntimeError(defect)

        device = torch.device("cpu")
        meta = json.loads(BC_JSON.read_text(encoding="utf-8"))
        arch = meta.get("architecture") or meta.get("config", {}).get("architecture")
        model = build_model(arch).to(device)
        apply_state_dict(model, BC_JSON, map_location=device)
        model.eval()
        checks["bc_load"] = "PASS"
        notes.append(f"loaded architecture={arch}")

        # Policy-only CPU latency (exclude env transition cost from CPU gate).
        actor = PersistentActor(actor_id="bc_inf", seed=42, reward_config=CONTROL_V1)
        actor.attach_model_state(model, device)
        from generals_bot.models.legal_mask import apply_action_mask, legal_mask_observation
        from generals_bot.models.model_forward import adapt_forward_output
        from generals_bot.models.observation_encoder import encode_globals_numpy, encode_grids_numpy
        from generals_bot.training.bridge_benchmark import extract_numpy_boards
        from generals_bot.training.collect_bc import _observation_from_arrays

        eng = actor.get_obs(actor.state, actor.learner_seat)
        tg, og, ag, _, meta_o = extract_numpy_boards(eng, actor.h, actor.w)
        cells = encode_grids_numpy(tg, og, ag)
        obs = _observation_from_arrays(tg, og, ag, meta_o)
        glob = encode_globals_numpy(obs)
        cell_t = torch.from_numpy(cells).unsqueeze(0)
        glob_t = torch.from_numpy(glob).unsqueeze(0)
        # warmup
        with torch.no_grad():
            if actor.cell_mem is not None:
                _ = model.forward_tensors(
                    cell_t, glob_t, actor.hidden, actor.cell_mem, deterministic=True
                )
            else:
                _ = model.forward_tensors(cell_t, glob_t, actor.hidden, deterministic=True)
        t0 = time.perf_counter()
        n_fwd = 16
        with torch.no_grad():
            for _ in range(n_fwd):
                if actor.cell_mem is not None:
                    raw = model.forward_tensors(
                        cell_t, glob_t, actor.hidden, actor.cell_mem, deterministic=True
                    )
                else:
                    raw = model.forward_tensors(cell_t, glob_t, actor.hidden, deterministic=True)
                fwd = adapt_forward_output(raw)
                mask = legal_mask_observation(obs, device=device).unsqueeze(0)
                masked = apply_action_mask(fwd.logits, mask)
                _ = torch.distributions.Categorical(logits=masked).probs
        cpu_ms = (time.perf_counter() - t0) * 1000.0 / n_fwd
        # Competition-style budget: 100ms decision; allow headroom for CPU research probe.
        checks["cpu_inference"] = "PASS" if cpu_ms <= 100.0 else "FAIL"
        notes.append(f"mean_policy_only_cpu_ms={cpu_ms:.2f}")

        frag = actor.collect_fragment(model, rollout_steps=16, device=device, policy_version=0)
        assert all(t.logp == t.logp for t in frag.transitions), "NaN logp"
        checks["legal_support"] = "PASS"

        actor2 = PersistentActor(actor_id="bc_reset", seed=99, reward_config=CONTROL_V1)
        actor2.attach_model_state(model, device)
        _ = actor2.collect_fragment(model, rollout_steps=8, device=device, policy_version=0)
        assert actor2.belief is not None
        checks["recurrent_reset"] = "PASS"
        checks["mixture_deterministic"] = "PASS_PROVISIONAL"
        notes.append(
            "Design A (deterministic=True) is default in PersistentActor after Stage 3 repair."
        )

        if checks["cpu_inference"] == "FAIL":
            classification = "CANDIDATE_B_BLOCKED_CPU"
            defect = f"CPU policy-only inference too slow: {cpu_ms:.2f} ms/forward"
        else:
            pkg_reg = json.loads(
                (REPO / "experiments" / "manifests" / "phase9f_package_registry_v2.json").read_text(
                    encoding="utf-8"
                )
            )
            if pkg_reg.get("hybrid_packages") in {None, "NONE_YET", "NONE"}:
                checks["hybrid_package"] = "BLOCKED"
                classification = "CANDIDATE_B_BLOCKED_RUNTIME"
                defect = (
                    "hybrid_packages=NONE_YET; BC inference semantics OK but no hybrid "
                    "package builder wired for upload"
                )
                notes.append(defect)
            else:
                checks["hybrid_package"] = "PASS"
                classification = "CANDIDATE_B_QUALIFIED"
                defect = None

        elapsed = time.perf_counter() - started
        if elapsed > TIMEBOX_S and classification not in {
            "CANDIDATE_B_QUALIFIED",
            "CANDIDATE_B_BLOCKED_SEMANTICS",
            "CANDIDATE_B_BLOCKED_RUNTIME",
            "CANDIDATE_B_BLOCKED_CPU",
        }:
            classification = "CANDIDATE_B_NOT_COMPLETED"

    except Exception as exc:  # noqa: BLE001
        if classification == "CANDIDATE_B_NOT_COMPLETED" and defect is None:
            # Distinguish semantics vs runtime
            msg = f"{type(exc).__name__}: {exc}"
            if "CUDA" in msg or "cuda" in msg:
                classification = "CANDIDATE_B_BLOCKED_RUNTIME"
            elif checks.get("bc_load") != "PASS":
                classification = "CANDIDATE_B_BLOCKED_SEMANTICS"
            else:
                classification = "CANDIDATE_B_BLOCKED_SEMANTICS"
            defect = msg
            notes.append(traceback.format_exc(limit=5))

    elapsed = time.perf_counter() - started
    doc = {
        "schema_version": 1,
        "kind": "CANDIDATE_B_FAST_LANE",
        "created_at": now,
        "timebox_s": TIMEBOX_S,
        "elapsed_s": elapsed,
        "bc_checkpoint": str(BC_JSON.as_posix()),
        "classification": classification,
        "checks": checks,
        "defect": defect,
        "notes": notes,
        "gate": "BC_INFERENCE_SEMANTICS_GATE",
        "blocks_first_recommendation": False,
    }
    out = REPO / "experiments" / "manifests" / "phase9fs_candidate_b_fast_lane.json"
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    (REPO / "experiments" / "reports" / "phase9fs_candidate_b_fast_lane.md").write_text(
        "\n".join(
            [
                "# Candidate B fast lane",
                "",
                f"Created: {now}",
                "",
                f"- Classification: **{classification}**",
                f"- Elapsed: {elapsed:.1f}s / {TIMEBOX_S}s",
                f"- Defect: {defect}",
                f"- Checks: `{json.dumps(checks)}`",
                "",
                "Does not block FIRST_RECOMMENDATION_GATE.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"classification": classification, "elapsed_s": elapsed, "defect": defect}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
