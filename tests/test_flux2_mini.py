"""Mini FLUX.2 transformer with random weights against diffusers' Flux2Transformer2DModel
(the reference side is tests/ref/mini_flux2.py).

    python tests/test_flux2_mini.py REF_DIR [q8|q4]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.flux2 import Flux2DiT, convert_transformer, schedule  # noqa: E402
from h3metal.gpu import Gpu  # noqa: E402
from h3metal.weights import WeightFile  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3METAL_REF", "ref") + "/mini_flux2"
fmt = sys.argv[2] if len(sys.argv) > 2 else "q8"
out = os.path.join(src, f"flux2_{fmt}.h3m")
convert_transformer(src, out, fmts={k: fmt for k in ("qkv", "out", "fc1", "fc2")}, log=lambda *a: None)
r = np.load(os.path.join(src, "ref.npz"))

n_img = r["latents"].shape[0]
sig = schedule(4, n_img)
assert np.abs(sig - r["sigmas"]).max() < 1e-6, (sig, r["sigmas"])

gpu = Gpu()
dit = Flux2DiT(gpu, WeightFile(gpu, out))
dit.prepare(r["txt"], r["i_ids"])
sim_path = os.path.join(src, f"sim_{fmt}.json")
sim = json.load(open(sim_path)) if os.path.exists(sim_path) else None
worst = 0.0
for s in range(4):
    o = dit.step(float(r["timesteps"][s]) / 1000, r[f"x{s}"])
    ref = r[f"out{s}"]
    rel = float(np.linalg.norm(o - ref) / np.linalg.norm(ref))
    cos = float((o * ref).sum() / np.linalg.norm(o) / np.linalg.norm(ref))
    worst = max(worst, rel)
    print(f"step {s}: rel L2 err {rel:.4f}  cosine {cos:.6f}" + (f"  (quantization alone: {sim[s]:.4f})" if sim else ""))
    if sim:  # the engine may add almost nothing to what its quantization costs (tests/ref/mini_flux2_sim.py)
        assert rel <= 1.1 * sim[s] + 0.003, (s, rel, sim[s])
# The mini model is built so that its blocks make 96% of the output (tests/ref/mini_flux2.py): there int8
# rounding costs 2.6% and int4 weights 29%, simulated in float; the real model loses far less.
assert worst < (0.035 if fmt == "q8" else 0.35), worst
print(f"ok ({fmt})")
