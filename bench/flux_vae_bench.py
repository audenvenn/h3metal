"""Time the FLUX decoder (h3metal/flux_vae.py) at the given image sides, with the weights of DIR/vae.

    python bench/flux_vae_bench.py DIR 1024 2048
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.flux_vae import FluxVAEDecoder, convert_flux_vae  # noqa: E402
from h3metal.gpu import Gpu  # noqa: E402
from h3metal.weights import WeightFile  # noqa: E402

src = sys.argv[1]
out = os.path.join(src, "vae.h3m")
if not os.path.exists(out):
    convert_flux_vae(src, out)
gpu = Gpu()
dec = FluxVAEDecoder(gpu, WeightFile(gpu, out))
for side in [int(s) for s in sys.argv[2:]] or [1024]:
    z = np.random.default_rng(0).standard_normal((16, side // 8, side // 8)).astype(np.float32)
    dec.decode(z)
    best = 1e9
    for _ in range(3):
        t = time.time()
        dec.decode(z)
        best = min(best, time.time() - t)
    print(f"FLUX decoder {side}x{side}: {best:.2f} s")
