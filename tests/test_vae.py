"""VAE decoder against the official module on the real weights (see tests/ref/vae_decode.py)."""
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu
from h3metal.vae import convert_vae, VAEDecoder
from h3metal.weights import WeightFile

W = os.environ["H3METAL_WEIGHTS"]  # the official MiniMax-H3-Image directory
ref = np.load(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3METAL_REF", "ref") + "/vae_ref.npz")
fmt = sys.argv[2] if len(sys.argv) > 2 else "f16"
out = os.path.join(os.environ.get("H3METAL_REF", "ref"), f"vae_{fmt}.h3m")
if not os.path.exists(out):
    convert_vae(W, out, fmt=fmt)
gpu = Gpu()
dec = VAEDecoder(gpu, WeightFile(gpu, out))
t = time.time()
img = dec.decode(ref["lat"], raw=True)
dt = time.time() - t
r = ref["img"]
err = np.abs(img - r)
psnr = 10 * np.log10(1.0 / np.mean(((img - r) * 0.226) ** 2))
print(f"{fmt}: max abs err {err.max():.4f}, mean {err.mean():.5f}, PSNR vs reference {psnr:.1f} dB, {dt:.2f} s")
