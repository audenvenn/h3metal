"""The GPU VAE encoder against the official module's posterior (see tests/ref/edit_parts.py)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu
from h3metal.vae import VAEEncoder
from h3metal.weights import WeightFile
r = np.load(os.environ.get("H3METAL_REF", "ref") + "/edit_ref.npz")
gpu = Gpu()
enc = VAEEncoder(gpu, WeightFile(gpu, sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.h3metal/model/vae.h3m")))
t = time.time()
mean, std = enc.encode(r["ref_img"])
dt = time.time() - t
em = np.linalg.norm(mean - r["post_mean"]) / np.linalg.norm(r["post_mean"])
es = np.linalg.norm(std - r["post_std"]) / np.linalg.norm(r["post_std"])
print(f"encoder {r['ref_img'].shape[0]}x{r['ref_img'].shape[1]}: mean rel err {em:.5f}, std rel err {es:.5f}, {dt:.2f} s")
assert em < 0.01 and es < 0.01
im = (np.random.default_rng(2).random((2048, 2048, 3)) * 255).astype(np.uint8)
enc.encode(im[:256, :256])
t = time.time(); enc.encode(im); print(f"2048x2048: {time.time() - t:.2f} s")
