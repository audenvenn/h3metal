"""Mini Z-Image transformer with random weights against diffusers' ZImageTransformer2DModel
(the reference side is tests/ref/mini_zimage.py).

    python tests/test_zimage_mini.py REF_DIR [q8|q4]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu  # noqa: E402
from h3metal.weights import WeightFile  # noqa: E402
from h3metal.zimage import ZImageDiT, convert_transformer, schedule  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3METAL_REF", "ref") + "/mini_zimage"
fmt = sys.argv[2] if len(sys.argv) > 2 else "q8"
out = os.path.join(src, f"zimage_{fmt}.h3m")
convert_transformer(src, out, fmts={k: fmt for k in ("qkv", "out", "fc1", "fc2")}, log=lambda *a: None)
r = np.load(os.path.join(src, "ref.npz"))

t, sig = schedule(8)
ref_t = ((1000 - r["timesteps"]) / 1000).astype(np.float32)
assert np.abs(t - ref_t).max() < 1e-6, (t, ref_t)
assert np.abs(sig - r["sigmas"]).max() < 1e-6, (sig, r["sigmas"])

gpu = Gpu()
dit = ZImageDiT(gpu, WeightFile(gpu, out))
C, Hh, W = r["latents"].shape
dit.prepare(r["cap"].shape[0], Hh, W)
dit.encode_caption(r["cap"])
worst = 0.0
for s in range(dit.n_steps):
    o = dit.step(s, r[f"x{s}"])
    ref = r[f"out{s}"]
    rel = float(np.linalg.norm(o - ref) / np.linalg.norm(ref))
    cos = float((o * ref).sum() / np.linalg.norm(o) / np.linalg.norm(ref))
    worst = max(worst, rel)
    print(f"step {s}: rel L2 err {rel:.4f}  cosine {cos:.6f}")
assert worst < 0.02, worst
print(f"ok ({fmt})")
