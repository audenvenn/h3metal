"""Time one full-size DiT block (random weights) at a given image size; extrapolates a 6-step, 50-block run."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu
from h3metal.ops import QWeight
from h3metal.dit import DiT, Layout

D, H, hd, F = 5376, 56, 128, 14336
fmt = os.environ.get("FMT", "q8")
side = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
T = 300
gpu = Gpu()
rng = np.random.default_rng(0)

def qw(N, K):
    if fmt == "q8":
        return QWeight("q8", gpu.upload(rng.integers(-127, 128, (N, K), dtype=np.int8)), gpu.upload(np.full(N, 1e-3, np.float32)), N, K)
    return QWeight(fmt, gpu.upload(rng.integers(0, 256, (N, K // 2), dtype=np.uint8)),
                   gpu.upload(np.full(N * (K // 128 if fmt == "q4g" else 1), 1e-3, np.float32)), N, K)

blk = {"qkv": qw(3 * H * hd, D), "out": qw(D, H * hd), "fc1": qw(2 * F, D), "fc2": qw(D, F),
       "q_norm": gpu.upload(np.ones(hd, np.float32)), "k_norm": gpu.upload(np.ones(hd, np.float32))}
dit = DiT.__new__(DiT)
dit.gpu = gpu; dit.D, dit.H, dit.hd, dit.F = D, H, hd, F; dit.inner = H * hd
dit.cfg = {"norm_eps": 1e-5, "qk_eps": 1e-5, "rope_freq": 16, "rope_theta": 10000.0}
dit.patch_dim = 96; dit._bufs = None; dit.attn_cfg = dict(wm=16, bk=32); dit.qk8 = os.environ.get("H3METAL_QK8", "1") == "1"
dit.x_embed_w = None
lat = side // 16
lay = Layout(T, np.ones(T, np.uint8), [], lat, lat)
dit.prepare(lay)
S = lay.length
b = dit._bufs
b["x"].numpy()[:S] = rng.standard_normal((S, D)).astype(np.float32)
mod = gpu.upload(np.stack([np.ones((3, D)), np.zeros((3, D))], 1).astype(np.float32))
gate = gpu.upload(np.full((3, D), 1e-3, np.float32))
def run():
    dit._block(b, blk, S, b["x"], mod, mod, gate, gate, b["rt"], 96, 2 * D, D)
run(); gpu.wait()
best = 1e9
for _ in range(3):
    run(); best = min(best, gpu.wait())
gpu.profile(True)
run()
prof = gpu.profile_take()
gpu.profile(False)
print(f"{side}x{side} ({S} rows) {fmt}: block {best*1e3:.1f} ms -> 50 blocks x 6 steps = {best*300:.1f} s")
tot = sum(t for _, t in prof)
for name, t in prof:
    short = name.split("(")[0].replace("ops.metal:", "").replace("gemm.metal:", "").replace("attention.metal:", "")
    print(f"   {short:22s} {t*1e3:8.2f} ms {100*t/tot:5.1f}%")
