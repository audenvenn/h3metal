"""Register-fragment product rates (bench/hw/mma_rate.metal): how fast the matrix units multiply per type."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from h3metal.gpu import Gpu  # noqa: E402

gpu = Gpu(os.path.dirname(os.path.abspath(__file__)))
ITERS, SG, TGS = 4096, 8, 12 * 64
out = gpu.empty((TGS * SG * 32,), "f32")
for name, ct, at in (("int8 x int8 -> int32", "int", "int8_t"), ("half x half -> float", "float", "half"),
                     ("half x half -> half", "half", "half"), ("bfloat x bfloat -> float", "float", "bfloat")):
    try:
        k = gpu.kernel("mma_rate.metal", "mma_rate", ACC_TYPE=ct, IN_TYPE=at, ITERS=ITERS)
    except Exception as e:
        print(f"{name:26s} not supported ({str(e).splitlines()[0][:80]})")
        continue
    best = 1e9
    for _ in range(4):
        gpu.launch(k, (TGS, 1, 1), (SG * 32, 1, 1), [out])
        best = min(best, gpu.wait())
    ops = 2.0 * 16 * 32 * 16 * 2 * ITERS * TGS * SG
    print(f"{name:26s} {ops / best / 1e12:6.1f} T ops/s", flush=True)
