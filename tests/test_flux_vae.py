"""FLUX decoder (h3metal/flux_vae.py) against diffusers' AutoencoderKL (reference: tests/ref/mini_flux_vae.py).

    python tests/test_flux_vae.py REF_DIR
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.flux_vae import FluxVAEDecoder, convert_flux_vae  # noqa: E402
from h3metal.gpu import Gpu  # noqa: E402
from h3metal.weights import WeightFile  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3METAL_REF", "ref") + "/mini_zimage"
out = os.path.join(src, "flux_vae.h3m")
r = np.load(os.path.join(src, "vae_ref.npz"))
convert_flux_vae(src, out, log=lambda *a: None, encoder="enc_mean" in r)
gpu = Gpu()
dec = FluxVAEDecoder(gpu, WeightFile(gpu, out))
img = dec.decode(r["z"])
ref = r["img"]
rel = float(np.linalg.norm(img - ref) / np.linalg.norm(ref))
mse = float(((np.clip(img, -1, 1) - np.clip(ref, -1, 1)) ** 2).mean())
psnr = 10 * np.log10(4.0 / mse)
print(f"FLUX decoder {img.shape}: rel L2 err {rel:.4f}, PSNR {psnr:.1f} dB")
assert rel < 0.01, rel
if "enc_mean" in r:  # FLUX.2's encoder, for the reference images of edits
    from h3metal.flux_vae import FluxVAEEncoder  # noqa: E402
    enc = FluxVAEEncoder(gpu, WeightFile(gpu, out)).encode(r["enc_img"])
    rel = float(np.linalg.norm(enc - r["enc_mean"]) / np.linalg.norm(r["enc_mean"]))
    print(f"FLUX encoder {enc.shape}: rel L2 err {rel:.4f}")
    assert rel < 0.01, rel
print("ok")
