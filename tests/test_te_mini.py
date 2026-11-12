"""Mini text encoder with random weights against transformers' Qwen3-VL (see tests/ref/mini_te.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu
from h3metal.te import convert_te, TextEncoder
from h3metal.weights import WeightFile
src = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3METAL_REF", "ref") + "/mini_te"
r = np.load(os.path.join(src, "ref.npz"))
gpu = Gpu()
for fmt in ("h8", "q8", "h4", "h4g", "f16"):
    out = os.path.join(src, f"te_{fmt}.h3m")
    convert_te(src, out, fmt=fmt, layers=3, log=lambda *a: None)
    te = TextEncoder(gpu, WeightFile(gpu, out))
    h = te.encode(r["ids"])
    ref = r["h3"]
    rel = np.linalg.norm(h - ref) / np.linalg.norm(ref)
    print(f"{fmt}: hidden_states[3] rel L2 err {rel:.4f}")
