"""Time one full-size Z-Image block (random weights) at a given image size, and its share of the matrix units'
peak; extrapolates the eight-step transformer time.

    FMT=q8 python bench/zimage_layer_bench.py 1024
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu  # noqa: E402
from h3metal.ops import QWeight  # noqa: E402
from h3metal.zimage import ZImageDiT  # noqa: E402

I8, H16 = 42.0e12, 20.6e12   # bench/hw/mma_rate.py on an M6: int8 TOPS, half->float TFLOPS
D, H, hd, F = 3840, 30, 128, 10240
fmt = os.environ.get("FMT", "q8")
side = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
CAP = 128  # caption rows (a prompt of up to 128 tokens with the chat template)
gpu = Gpu()
rng = np.random.default_rng(0)


def qw(N, K):
    if fmt == "q8":
        return QWeight("q8", gpu.upload(rng.integers(-127, 128, (N, K), dtype=np.int8)),
                       gpu.upload(np.full(N, 1e-3, np.float32)), N, K)
    return QWeight(fmt, gpu.upload(rng.integers(0, 256, (N, K // 2), dtype=np.uint8)),
                   gpu.upload(np.full(N, 1e-3, np.float32)), N, K)


blk = {"qkv": qw(3 * D, D), "out": qw(D, D), "fc1": qw(2 * F, D), "fc2": qw(D, F),
       "q_norm": gpu.upload(np.ones(hd, np.float32)), "k_norm": gpu.upload(np.ones(hd, np.float32)),
       "pre_attn": gpu.upload(np.stack([np.ones((1, D)), np.zeros((1, D))], 1).astype(np.float32)),
       "pre_mlp": gpu.upload(np.stack([np.ones((1, D)), np.zeros((1, D))], 1).astype(np.float32)),
       "post_attn": gpu.upload(np.full((1, D), 1e-3, np.float32)),
       "post_mlp": gpu.upload(np.full((1, D), 1e-3, np.float32))}
dit = ZImageDiT.__new__(ZImageDiT)
dit.gpu = gpu
dit.D, dit.H, dit.hd, dit.F = D, H, hd, F
dit.cfg = {"norm_eps": 1e-5, "patch": 2, "axes_dims": [32, 48, 48], "rope_theta": 256.0}
dit._bufs = None
dit.patch_dim = 64
lat = side // 8
dit.prepare(CAP, lat, lat)
S = dit.S
b = dit._bufs
b["x"].numpy()[:S] = rng.standard_normal((S, D)).astype(np.float32)


def run():  # one block, its last update fused into the next block's first norm as in a real step
    dit._block(b, blk, b["x"], 0, S, 0, blk["post_mlp"].view((D,)))


run()
gpu.wait()
best = 1e9
for _ in range(3):
    run()
    best = min(best, gpu.wait())
gpu.profile(True)
run()
prof = gpu.profile_take()
gpu.profile(False)
ops_prod = 2 * S * D * (3 * D + D + 2 * F) + 2 * S * F * D
att_floor = 2 * S * S * hd * H / I8 + 2 * S * S * hd * H / H16
floor = ops_prod / I8 + att_floor
n_img = (lat // 2) ** 2
# Eight steps: 30 blocks over all rows and two refiner blocks over the image rows (the caption refiner runs once).
est = 8 * (30 * best + 2 * best * n_img / S)
print(f"{side}x{side} ({S} rows) {fmt}: block {best * 1e3:.1f} ms, {floor / best:.0%} of the floor "
      f"{floor * 1e3:.1f} ms; eight steps ~{est:.1f} s")
tot = sum(t for _, t in prof)
for name, t in prof:
    short = name.split("(")[0].replace("ops.metal:", "").replace("gemm.metal:", "").replace("attention.metal:", "")
    print(f"   {short:22s} {t * 1e3:8.2f} ms {100 * t / tot:5.1f}%")
