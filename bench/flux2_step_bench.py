"""Time one FLUX.2 [klein] transformer step of the converted model at a given image size, its share of the matrix
units' floor, and where the time goes (kernels summed over the step).

    python bench/flux2_step_bench.py [MODEL_DIR] [SIZE]
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.flux2 import Flux2DiT, image_positions  # noqa: E402
from h3metal.gpu import Gpu  # noqa: E402
from h3metal.weights import WeightFile  # noqa: E402

I8, H16 = 42.0e12, 20.6e12   # bench/hw/mma_rate.py on an M6: int8 TOPS, half->float TFLOPS
model = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/.h3metal/flux2-klein")
side = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
gpu = Gpu()
dit = Flux2DiT(gpu, WeightFile(gpu, os.path.join(model, "dit.h3m")))
c = dit.cfg
rng = np.random.default_rng(0)
T = 512
hp = wp = side // 16
dit.prepare(rng.standard_normal((T, c["joint_dim"])).astype(np.float32), image_positions(hp, wp))
lat = rng.standard_normal((hp * wp, c["in_channels"])).astype(np.float32)
S, D, F, I = dit.S, dit.D, dit.F, hp * wp
dit.step(0.9, lat)
best = 1e9
for _ in range(3):
    t = __import__("time").time()
    dit.step(0.9, lat)
    best = min(best, __import__("time").time() - t)
gpu.profile(True)
dit.step(0.9, lat)
prof = gpu.profile_take()
gpu.profile(False)
# The floor: every product at the int8 peak, attention's two products at their peaks.
per_row_double = 2 * D * (3 * D + D + 2 * F) + 2 * F * D
ops_prod = dit.L2 * S * per_row_double + dit.L1 * S * (2 * D * (3 * D + 2 * F) + 2 * (D + F) * D)
att = (dit.L2 + dit.L1) * 2 * S * S * D
floor = ops_prod / I8 + att / I8 + att / H16
gpu_total = sum(t for _, t in prof)
print(f"{side}x{side} ({S} rows): step {best * 1e3:.0f} ms wall, {gpu_total * 1e3:.0f} ms on the GPU; floor "
      f"{floor * 1e3:.0f} ms: {floor / best:.0%} of it (GPU time {floor / gpu_total:.0%})")
agg = defaultdict(float)
cnt = defaultdict(int)
for name, t in prof:
    key = name.split("(")[0].split(":")[-1]
    if key == "gemm":  # products by their shape class
        key = "gemm " + name.split("(", 1)[1].split(")")[0][:60] if "(" in name else key
    agg[key] += t
    cnt[key] += 1
for k, t in sorted(agg.items(), key=lambda kv: -kv[1])[:14]:
    print(f"   {k:70s} {cnt[k]:4d}x {t * 1e3:8.1f} ms {100 * t / gpu_total:5.1f}%")
