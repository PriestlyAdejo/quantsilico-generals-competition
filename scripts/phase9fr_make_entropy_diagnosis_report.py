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


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    now = datetime.now(timezone.utc).isoformat()

    evid = {
        "BC_init": repo / "experiments" / "manifests" / "phase9fr_ppo_ratio_semantics_model.json",
        "CONTROL_update10": repo / "experiments" / "manifests" / "phase9fr_ppo_ratio_semantics_rl_control_update_10.json",
        "CONTROL_update50": repo / "experiments" / "manifests" / "phase9fr_ppo_ratio_semantics_rl_control_update_50.json",
        "CONTROL_final": repo / "experiments" / "manifests" / "phase9fr_ppo_ratio_semantics.json",
        "CURRICULUM_update10": repo / "experiments" / "manifests" / "phase9fr_ppo_ratio_semantics_rl_curriculum_update_10.json",
        "CURRICULUM_final": repo / "experiments" / "manifests" / "phase9fr_ppo_ratio_semantics_rl_curriculum.json",
    }

    rows = {}
    for k, p in evid.items():
        d = load(p)
        rows[k] = {
            "evidence_sha256": sha256(p),
            "checkpoint_json": d.get("checkpoint_json"),
            "step_zero_ratio_passed": d.get("step_zero_ratio_gate", {}).get("transitions_passed"),
            "step_zero_ratio_total": d.get("step_zero_ratio_gate", {}).get("transitions_total"),
            "step_zero_ratio_pass_rate": d.get("step_zero_ratio_gate", {}).get("pass_rate"),
            "behaviour_entropy_full_legal_mean": d.get("entropy_diagnosis", {}).get("behaviour_entropy_full_legal_mean"),
            "update_mask_entropy_same_logits_mean": d.get("entropy_diagnosis", {}).get("update_mask_entropy_same_logits_mean"),
            "update_impl_entropy_mean": d.get("entropy_diagnosis", {}).get("update_impl_entropy_mean"),
            "pass_probability_full_legal_mean": d.get("support_diagnostics", {}).get("pass_probability_full_legal_mean"),
            "update_support_sizes_unique": d.get("support_diagnostics", {}).get("update_support_sizes_unique"),
        }

    out = {
        "schema_version": 1,
        "kind": "PHASE9FR_ENTROPY_DIAGNOSIS",
        "created_at": now,
        "notes": [
            "behaviour_entropy_full_legal_mean: entropy of Categorical over FULL legal mask at collection time (old_logp distribution).",
            "update_mask_entropy_same_logits_mean: entropy of Categorical when restricting support to {chosen action, PASS} on the same logits as collection time.",
            "update_impl_entropy_mean: entropy of Categorical under ppo_update_from_fragment semantics (hidden/cell_mem reset per-transition + {chosen action, PASS} mask).",
            "High-level interpretation: if both behaviour entropy and update-time entropy are ~0, collapse is in the actual behaviour policy, not only telemetry."
        ],
        "rows": rows,
    }

    out_path = repo / "experiments" / "manifests" / "phase9fr_entropy_diagnosis.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    md = []
    md.append("# Phase 9F-R entropy diagnosis (recorded vs update-time)")
    md.append("")
    md.append(f"Created: {now}")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append("Entropy collapses along with the behaviour policy becoming effectively deterministic on `PASS`. "
              "After collapse, the PPO ratio gate becomes trivially satisfied because the selected action equals PASS, "
              "making update-time support restriction irrelevant for log-prob.")
    md.append("")
    md.append("## Evidence table (means)")
    md.append("")
    for name in evid.keys():
        r = rows[name]
        p_pass = r.get("pass_probability_full_legal_mean")
        p_pass_str = f"{p_pass:.6g}" if p_pass is not None else "NA"
        md.append(f"- {name}: step-zero-pass {r['step_zero_ratio_passed']}/{r['step_zero_ratio_total']} "
                  f"(rate {r['step_zero_ratio_pass_rate']:.2%}); "
                  f"H_behaviour={r['behaviour_entropy_full_legal_mean']:.6g}; "
                  f"H_update_mask_same={r['update_mask_entropy_same_logits_mean']:.6g}; "
                  f"H_update_impl={r['update_impl_entropy_mean']:.6g}; "
                  f"P(pass)={p_pass_str}; "
                  f"update_support_unique={r['update_support_sizes_unique']}")
    md.append("")
    md.append("## Where the 0.0 entropy comes from")
    md.append("")
    md.append("In `src/generals_bot/training/rollout.py::ppo_update_from_fragment`, update-time entropy is computed from "
              "a Categorical distribution whose logits have been masked down to the support `{acts_t, PASS}` only. "
              "This will yield entropy 0.0 whenever `acts_t == PASS` (support size 1), which is exactly what happens "
              "in the late PPO checkpoints.")
    md.append("")

    md_path = repo / "experiments" / "reports" / "phase9fr_entropy_diagnosis.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

