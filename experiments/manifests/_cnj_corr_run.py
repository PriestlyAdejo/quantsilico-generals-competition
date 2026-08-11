from pathlib import Path
import json
from train.competition_native_jax.train_jax import run_gpu_correctness_gate
r = run_gpu_correctness_gate(Path("experiments/competition_native_jax/gpu_correctness"))
Path("experiments/manifests/_cnj_corr.out").write_text(json.dumps(r, indent=2), encoding="utf-8")
print("DONE", r["status"], r["device"])
