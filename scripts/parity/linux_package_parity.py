"""Linux-oriented submission package parity checks (also runnable under Docker).

Approximates the competition sandbox; does not claim bit-identical identity.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path


def _one_obs(h: int, w: int, turn: int = 1) -> str:
    trows = [["1"] * w for _ in range(h)]
    trows[0][0] = "4"
    types = "\n".join(" ".join(r) for r in trows) + "\n"
    owners = (
        "\n".join(
            " ".join(["1" if r == 0 and c == 0 else "0" for c in range(w)]) for r in range(h)
        )
        + "\n"
    )
    armies = (
        "\n".join(
            " ".join(["5" if r == 0 and c == 0 else "0" for c in range(w)]) for r in range(h)
        )
        + "\n"
    )
    return f"{turn} 1 5 0 0\n" + types + owners + armies


def run_parity(package_zip: Path, report_out: Path) -> dict:
    package_zip = Path(package_zip)
    report_out = Path(report_out)
    report: dict = {
        "schema_version": 1,
        "passed": False,
        "failure_reason": "",
        "package": str(package_zip),
        "approximation_note": (
            "Approximates competition Linux sandbox with pinned CPU deps; "
            "not a claim of exact competition-environment identity "
            "(image/kernel/cgroup differences may remain)."
        ),
        "run_sh_present": False,
        "lf_ok": False,
        "executable_ok": False,
        "build_sh_ran": False,
        "board_sizes": [],
        "latencies_ms": [],
        "first_action_latency_ms": None,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "latency_p99_ms": None,
        "peak_memory_mb": None,
        "protocol_faults": 0,
        "network": os.environ.get("PARITY_NETWORK", "assumed_disabled"),
        "cpu_limit_note": os.environ.get("PARITY_CPU_NOTE", "1 core / 2GB when launched via Docker/CI"),
    }

    staging = Path(tempfile.mkdtemp(prefix="generals_parity_"))
    try:
        with zipfile.ZipFile(package_zip, "r") as zf:
            zf.extractall(staging)
            names = zf.namelist()
        run_sh = staging / "run.sh"
        report["run_sh_present"] = run_sh.is_file() and "run.sh" in names
        if not report["run_sh_present"]:
            report["failure_reason"] = "run.sh missing at package root"
            report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return report

        data = run_sh.read_bytes()
        report["lf_ok"] = b"\r" not in data
        if not report["lf_ok"]:
            report["failure_reason"] = "run.sh has CRLF line endings"
            report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return report

        mode = run_sh.stat().st_mode
        if mode & 0o111 == 0:
            run_sh.chmod(mode | 0o755)
        report["executable_ok"] = (run_sh.stat().st_mode & 0o111) != 0
        if not report["executable_ok"]:
            report["failure_reason"] = "run.sh not executable"
            report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return report

        build_sh = staging / "build.sh"
        if build_sh.is_file():
            build_sh.chmod(build_sh.stat().st_mode | 0o755)
            subprocess.check_call(["bash", str(build_sh)], cwd=str(staging))
            report["build_sh_ran"] = True

        sizes = [(3, 3), (5, 8), (10, 10), (15, 12), (21, 21)]
        latencies: list[float] = []
        for h, w in sizes:
            cmd = ["bash", "./run.sh"]
            proc = subprocess.Popen(
                cmd,
                cwd=str(staging),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            payload = f"0 {h} {w}\n" + _one_obs(h, w)
            t0 = time.perf_counter()
            out, err = proc.communicate(payload, timeout=60)
            dt = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt)
            lines = [ln for ln in (out or "").splitlines() if ln.strip()]
            if not lines or len(lines[0].split()) != 5:
                report["protocol_faults"] += 1
                report["failure_reason"] = f"bad action for {h}x{w}: out={out!r} err={err!r}"
                report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                return report
            report["board_sizes"].append(
                {"h": h, "w": w, "action": lines[0], "latency_ms": dt, "returncode": proc.returncode}
            )

        report["latencies_ms"] = latencies
        report["first_action_latency_ms"] = latencies[0]
        ordered = sorted(latencies)
        report["latency_p50_ms"] = statistics.median(latencies)
        report["latency_p95_ms"] = ordered[max(0, int(0.95 * (len(ordered) - 1)))]
        report["latency_p99_ms"] = ordered[-1]
        report["passed"] = report["protocol_faults"] == 0
        if not report["passed"] and not report["failure_reason"]:
            report["failure_reason"] = "protocol faults"
        report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("package_zip")
    p.add_argument("--report", default="experiments/manifests/linux_parity_report.json")
    args = p.parse_args()
    report = run_parity(Path(args.package_zip), Path(args.report))
    print(json.dumps({"passed": report["passed"], "report": args.report, "reason": report.get("failure_reason")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
