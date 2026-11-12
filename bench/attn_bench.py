"""Flash attention throughput at the transformer's shapes (56 heads, head dim 128)."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu
from h3metal.ops import attention

gpu = Gpu()
H, D = 56, 128
Ls = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "4396").split(",")]
cfgs = [(int(a), int(b)) for a, b in (c.split("x") for c in (sys.argv[2] if len(sys.argv) > 2 else "4x32,4x64,8x32,8x64,2x64").split(","))]
for L in Ls:
    rows = (L + 127) // 128 * 128
    rng = np.random.default_rng(0)
    Q = gpu.upload(rng.standard_normal((H, rows, D)).astype(np.float16))
    K = gpu.upload(rng.standard_normal((H, rows, D)).astype(np.float16))
    V = gpu.upload(rng.standard_normal((H, rows, D)).astype(np.float16))
    O = gpu.empty((rows, H * D), "bf16")
    for wm, bk in cfgs:
        var = dict(x.split("=") for x in os.environ.get("VARIANT", "").split(",") if x)
        f = lambda: attention(gpu, Q, K, V, O, Lq=L, L=L, H=H, KVH=H, D=D, q_rows=rows, k_rows=rows, scale=D**-0.5, wm=wm, bk=bk, **var)
        f(); gpu.wait()
        best = 1e9
        for _ in range(3 if L < 10000 else 2):
            f(); best = min(best, gpu.wait())
        print(f"L={L} wm={wm} bk={bk}: {best*1e3:.1f} ms {4*H*L*L*D/best/1e12:.2f} TFLOPS", flush=True)
