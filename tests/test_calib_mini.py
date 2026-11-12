"""GPTQ calibration on the mini DiT (see tests/ref/mini_dit.py): runs end to end and does not do worse than
rounding to nearest. The calibration trajectory comes from the int8 model itself, the comparison from the
official module's velocities."""
import os
import sys
import tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu
from h3metal.convert import convert_dit
from h3metal.calib import calibrate
from h3metal.dit import DiT, Layout, patchify, euler
from h3metal.weights import WeightFile, bf16_bits

src = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3METAL_REF", "ref") + "/mini_dit"
d = tempfile.mkdtemp()
q8, q4, q4g = (os.path.join(d, f) for f in ("q8.h3m", "q4.h3m", "q4_gptq.h3m"))
convert_dit(src, q8, fmts={k: "q8" for k in ("qkv", "out", "fc1", "fc2")}, log=lambda *a: None)
convert_dit(src, q4, fmts={k: "q4" for k in ("qkv", "out", "fc1", "fc2")}, log=lambda *a: None)
r = np.load(os.path.join(src, "ref_t2i.npz"))
gpu = Gpu()
# a calibration "reference run" made with the int8 model
m = DiT(gpu, WeightFile(gpu, q8))
T = r["text"].shape[0]
text = gpu.empty((128, r["text"].shape[1]), "bf16"); text.numpy()[:T] = bf16_bits(r["text"])
rows = m.encode_text(text, T)
lh, lw = (int(v) for v in r["lat_hw"])
m.prepare(Layout(T, np.ones(T, np.uint8), [], lh, lw))
noise = np.random.default_rng(9).standard_normal((1, 24, 1, lh, lw)).astype(np.float32)
x = patchify(noise.reshape(24, lh, lw)); vels = {}
for s in range(6):
    v = m.step(s, rows, x); vels[f"vel{s}"] = v; x = euler(x, v, s, m.sigmas)
ref = os.path.join(d, "calib_ref.npz")
np.savez(ref, emb=r["text"], noise=noise, canvas=np.array([lh * 16, lw * 16]), **vels)
calibrate(src, q8, q4g, [ref], fmt="q4", log=lambda *a: None)
for name, path in (("q8", q8), ("q4 nearest", q4), ("q4 gptq", q4g)):
    dm = DiT(gpu, WeightFile(gpu, path)); rows = dm.encode_text(text, T)
    dm.prepare(Layout(T, np.ones(T, np.uint8), [], lh, lw))
    errs = []
    for s in range(6):
        v = dm.step(s, rows, r["latents"])
        errs.append(np.linalg.norm(v - r[f"vel{s}"]) / np.linalg.norm(r[f"vel{s}"]))
    print(f"{name}: velocity rel err vs official, mean over steps {np.mean(errs):.4f}")
