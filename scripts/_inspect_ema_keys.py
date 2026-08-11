import numpy as np
from pathlib import Path

d = np.load("experiments/competition_native_jax/v4_3_r_e6/ckpt_final/ema.npz")
print("nkeys", len(d.files))
for k in d.files:
    print(k, d[k].shape)
