"""Time one full-size text-encoder layer (random weights)."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu
from h3metal.ops import QWeight
from h3metal.te import TextEncoder

D, Hq, Hkv, hd, F = 5120, 64, 8, 128, 25600
gpu = Gpu(); rng = np.random.default_rng(0)
def qw(fmt, N, K):
    if fmt in ("h8", "q8"):
        return QWeight(fmt, gpu.upload(rng.integers(-127, 128, (N, K), dtype=np.int8)), gpu.upload(np.full(N, 1e-3, np.float32)), N, K)
    sc = N * (K // 128) if fmt.endswith("g") else N
    return QWeight(fmt, gpu.upload(rng.integers(0, 256, (N, K // 2), dtype=np.uint8)), gpu.upload(np.full(sc, 1e-3, np.float32)), N, K)
for fmt in sys.argv[2].split(",") if len(sys.argv) > 2 else ("h8", "h4g", "q8"):
    te = TextEncoder.__new__(TextEncoder)
    te.gpu = gpu; te.D, te.Hq, te.Hkv, te.hd, te.F = D, Hq, Hkv, hd, F
    te.cfg = {"eps": 1e-6, "rope_theta": 5e6, "mrope_section": [24, 20, 20]}
    te.fmt = fmt; te.half_act = fmt.startswith("h"); te.L = 1
    te.embed = rng.integers(0, 65535, (1000, D), dtype=np.uint16)
    te.ones = gpu.upload(np.ones((1, D), np.float32))
    mod = gpu.upload(np.stack([np.ones(D), np.zeros(D)])[None].astype(np.float32))
    te.layers = [{"qkv": qw(fmt, (Hq + 2 * Hkv) * hd, D), "out": qw(fmt, D, Hq * hd), "fc1": qw(fmt, 2 * F, D), "fc2": qw(fmt, D, F),
                  "mod1": mod, "mod2": mod, "q_norm": gpu.upload(np.ones(hd, np.float32)), "k_norm": gpu.upload(np.ones(hd, np.float32))}]
    for T in [int(x) for x in sys.argv[1].split(",")]:
        ids = rng.integers(0, 1000, T)
        te.encode(ids)
        import time
        best = 1e9
        for _ in range(3):
            t = time.time(); te.encode(ids); best = min(best, time.time() - t)
        print(f"{fmt} T={T}: {best*1e3:.1f} ms per layer -> 50 layers {best*50:.2f} s (compute only)", flush=True)
