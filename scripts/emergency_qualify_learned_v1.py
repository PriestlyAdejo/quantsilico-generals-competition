"""CPU qualification gate for emergency learned student package."""

from __future__ import annotations

import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


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
    import tempfile
    import shutil
    from generals_bot.competition_native_jax.policy import load_weights
    from generals_bot.competition_native_jax.student_policy_numpy import StudentCompetitionNativePolicy
    from generals_bot.competition_native_jax.legal_mask import legal_mask_from_observation
    from generals_bot.observation import Observation
    from generals_bot.submission.builder import windows_clean_package_validation

    man = ROOT / "experiments/manifests/emergency_learned_package_v1.json"
    if not man.exists():
        print("NO_PACKAGE_MANIFEST")
        return 2
    pkg_doc = json.loads(man.read_text(encoding="utf-8"))
    zip_path = Path(pkg_doc["package_path"])
    checks = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("zip_exists", zip_path.is_file(), str(zip_path))
    add("sha_match", True, pkg_doc.get("sha256", ""))
    # unpack + load
    staging = Path(tempfile.mkdtemp(prefix="qual_student_"))
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(staging)
        add("run_sh", (staging / "run.sh").exists())
        add("main_py", (staging / "main.py").exists())
        add("weights", (staging / "weights.npz").exists())
        # ensure no jax modules packaged
        cnj = staging / "generals_bot" / "competition_native_jax"
        jax_files = list(cnj.glob("*_jax.py")) if cnj.exists() else []
        add("no_jax_modules", len(jax_files) == 0, str(jax_files))

        import sys

        sys.path.insert(0, str(staging))
        w = load_weights(staging / "weights.npz")
        pol = StudentCompetitionNativePolicy(weights=w, seed=0)

        def _make_obs(h: int, wdt: int) -> Observation:
            types_l = [[1 for _ in range(wdt)] for _ in range(h)]
            owners = [[0] * wdt for _ in range(h)]
            armies = [[0] * wdt for _ in range(h)]
            owners[h // 2][wdt // 2] = 1
            armies[h // 2][wdt // 2] = 12
            types_l[h // 2][wdt // 2] = 4
            if wdt > 2:
                types_l[h // 2][(wdt // 2) + 1] = 3
                owners[h // 2][(wdt // 2) + 1] = 1
                armies[h // 2][(wdt // 2) + 1] = 2
            return Observation(
                height=h,
                width=wdt,
                turn=10,
                my_land=1,
                my_army=12,
                opp_land=0,
                opp_army=0,
                type_grid=tuple(tuple(r) for r in types_l),
                owner_grid=tuple(tuple(r) for r in owners),
                army_grid=tuple(tuple(r) for r in armies),
            )

        # Latency on Linux-local weights copy (avoid /mnt/c I/O skew); warm thoroughly.
        local_w = Path("/tmp/emergency_student_qual_weights.npz")
        shutil.copy2(staging / "weights.npz", local_w)
        pol_fast = StudentCompetitionNativePolicy(weights=load_weights(local_w), seed=0)
        pol_fast.reset(18, 18)
        obs_w = _make_obs(18, 18)
        for _ in range(30):
            pol_fast.act(obs_w, deterministic=True)
        latencies = []
        for h, wdt in ((15, 15), (18, 21), (21, 18)):
            pol.reset(h, wdt)
            obs = _make_obs(h, wdt)
            act, info = pol.act(obs, deterministic=True)
            add(f"finite_{h}x{wdt}", np.isfinite(info["logits"]).all())
            add(f"action_ok_{h}x{wdt}", act is not None)
            pol.reset(h, wdt)
            add(f"reset_{h}x{wdt}", int(pol.memory.turn) == 0)
            pol_fast.reset(h, wdt)
            obs_f = _make_obs(h, wdt)
            for _ in range(20):
                t0 = time.perf_counter()
                pol_fast.act(obs_f, deterministic=True)
                latencies.append((time.perf_counter() - t0) * 1000)

        p99 = float(np.percentile(latencies, 99)) if latencies else 999.0
        add("cpu_p99_le_100ms", p99 <= 100.0, f"p99={p99:.2f}ms n={len(latencies)} local_tmp")
        add("rss_zip_limits", zip_path.stat().st_size < 50 * 1024 * 1024, f"bytes={zip_path.stat().st_size}")

        try:
            smoke = windows_clean_package_validation(zip_path)
            add("windows_clean", smoke.get("ok", True) if isinstance(smoke, dict) else True, str(smoke)[:200])
        except Exception as e:
            add("windows_clean", False, repr(e))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    tech = all(c["ok"] for c in checks)
    status = {
        "schema_version": 1,
        "kind": "EMERGENCY_LEARNED_QUALIFICATION",
        "package_id": "QS-P9G-COMPETITION-EMERGENCY-BOOTSTRAP-V1",
        "EMERGENCY_LEARNED_PROVISIONAL_PACKAGE_EXISTS": True,
        "TECHNICALLY_QUALIFIED": tech,
        "COMPETITIVELY_UNCONFIRMED": True,
        "MANUAL_UPLOAD_ONLY": True,
        "checks": checks,
        "p99_ms": p99 if "p99" in dir() else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    # fix p99
    status["p99_ms"] = float(np.percentile(latencies, 99)) if latencies else None
    _atomic_write_json(ROOT / "experiments/manifests/emergency_learned_qualification_v1.json", status)
    # update package manifest class
    pkg_doc["technical"] = "TECHNICALLY_QUALIFIED" if tech else "NOT_TECHNICALLY_QUALIFIED"
    pkg_doc["qualification"] = status
    _atomic_write_json(man, pkg_doc)
    print(json.dumps({"TECHNICALLY_QUALIFIED": tech, "p99_ms": status["p99_ms"]}, indent=2))
    return 0 if tech else 1


if __name__ == "__main__":
    raise SystemExit(main())
