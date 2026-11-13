"""FLUX.2's text encoder path (h3metal/te.py with layer taps and a padded prompt) against transformers
(reference: tests/ref/mini_flux2_te.py).

    python tests/test_flux2_te.py REF_DIR
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu  # noqa: E402
from h3metal.te import TextEncoder, convert_te  # noqa: E402
from h3metal.weights import WeightFile  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3METAL_REF", "ref") + "/mini_flux2_te"
r = np.load(os.path.join(src, "te_ref.npz"))
taps = tuple(int(t) for t in r["taps"])
gpu = Gpu()
for fmt in ("f16", "h8"):
    out = os.path.join(src, f"te_{fmt}.h3m")
    convert_te(os.path.join(src, "text_encoder"), out, fmt=fmt, layers=max(taps), prefix="model", log=lambda *a: None)
    te = TextEncoder(gpu, WeightFile(gpu, out))
    h = te.encode(r["ids"], taps=taps, n_valid=int(r["n_valid"]))
    ref = r["feats"]
    n = int(r["n_valid"])
    for part, sl in (("prompt rows", slice(0, n)), ("padding rows", slice(n, None))):
        rel = float(np.linalg.norm(h[sl] - ref[sl]) / np.linalg.norm(ref[sl]))
        print(f"{fmt} {part}: rel L2 err {rel:.4f}")
        assert rel < (0.01 if fmt == "f16" else 0.03), (fmt, part, rel)
print("ok")
