"""Distillation plumbing probe — hard 25-minute timebox; do not debug for hours."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home() / "quantsilico-runtime" / "emergency_rolling_v1"
PLUMBING_DEADLINE_S = 25 * 60


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


def main() -> int:
    t0 = time.perf_counter()
    steps: list[dict] = []

    def step(name: str, ok: bool, detail: str = "") -> None:
        steps.append(
            {
                "name": name,
                "ok": ok,
                "detail": detail,
                "elapsed_s": time.perf_counter() - t0,
            }
        )

    # 1) Feasibility artefact exists
    feas_path = ROOT / "experiments/manifests/competition_native_jax_early_student_feasibility.json"
    if not feas_path.exists():
        step("feasibility_manifest", False, "missing")
        status = "SKIP_BOOTSTRAP_LEARNED_DISTILLATION"
    else:
        feas = json.loads(feas_path.read_text(encoding="utf-8"))
        selected = (feas.get("selected") or feas.get("selected_student") or "student_emb96_d2_h4")
        step("feasibility_manifest", True, str(selected))
        # 2) Is there a distill train entrypoint?
        candidates = [
            ROOT / "scripts/run_competition_native_jax_distill.py",
            ROOT / "scripts/emergency_distill_student_emb96.py",
            ROOT / "train/competition_native_jax/distill_jax.py",
        ]
        found = [str(p.relative_to(ROOT)).replace("\\", "/") for p in candidates if p.exists()]
        step("distill_entrypoint", bool(found), ",".join(found) if found else "none")
        # 3) Student architecture importable?
        try:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
            # Prefer known student module if present
            import importlib

            ok_import = False
            detail = ""
            for mod in (
                "generals_bot.competition_native_jax.student_transformer_jax",
                "train.competition_native_jax.student_jax",
            ):
                try:
                    importlib.import_module(mod)
                    ok_import = True
                    detail = mod
                    break
                except Exception as e:
                    detail = str(e)[:200]
            step("student_module", ok_import, detail)
        except Exception as e:
            step("student_module", False, str(e)[:200])

        # 4) Dataset generator?
        ds = list(ROOT.glob("scripts/**/*distill*dataset*")) + list(
            ROOT.glob("scripts/**/*teacher*logits*")
        )
        step("dataset_generator", bool(ds), str(len(ds)))

        ready = all(s["ok"] for s in steps if s["name"] in ("feasibility_manifest", "distill_entrypoint", "student_module"))
        if time.perf_counter() - t0 > PLUMBING_DEADLINE_S:
            status = "SKIP_BOOTSTRAP_LEARNED_DISTILLATION"
            ready = False
        elif ready:
            status = "DISTILL_PLUMBING_READY"
        else:
            status = "SKIP_BOOTSTRAP_LEARNED_DISTILLATION"

    report = {
        "schema_version": 1,
        "kind": "EMERGENCY_DISTILL_PLUMBING_PROBE",
        "status": status,
        "plumbing_deadline_s": PLUMBING_DEADLINE_S,
        "elapsed_s": time.perf_counter() - t0,
        "steps": steps,
        "action": (
            "PAUSE_PPO_FOR_DISTILL"
            if status == "DISTILL_PLUMBING_READY"
            else "RESUME_PPO_KEEP_FALLBACK_RETRY_LATER"
        ),
        "max_distill_runs": 2,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(ROOT / "experiments/manifests/emergency_distill_plumbing.json", report)
    _atomic_write_json(RUNTIME / "programme" / "distill_plumbing.json", report)
    print(json.dumps({"status": status, "elapsed_s": report["elapsed_s"]}, indent=2))
    return 0 if status != "ERROR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
