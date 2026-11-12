"""Z-Image's text encoder path (h3metal/te.py on a plain Qwen3, keys under "model.", all but the last layer)
against transformers (reference: tests/ref/mini_qwen3.py).

    python tests/test_qwen3_te.py REF_DIR
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu  # noqa: E402
from h3metal.te import TextEncoder  # noqa: E402
from h3metal.weights import WeightFile  # noqa: E402
from h3metal.zimage import convert_text_encoder  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3METAL_REF", "ref") + "/mini_zimage"
r = np.load(os.path.join(src, "te_ref.npz"))
gpu = Gpu()
for fmt in ("f16", "h8"):
    out = os.path.join(src, f"te_{fmt}.h3m")
    convert_text_encoder(src, out, fmt=fmt, log=lambda *a: None)
    te = TextEncoder(gpu, WeightFile(gpu, out))
    h = te.encode(r["ids"])
    ref = r["hidden"]
    rel = float(np.linalg.norm(h - ref) / np.linalg.norm(ref))
    print(f"{fmt}: rel L2 err {rel:.4f}")
    assert rel < (0.01 if fmt == "f16" else 0.03), rel
print("ok")
