"""Mini text encoder with random weights against transformers' Qwen3-VL (see tests/ref/mini_te.py)."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu
from h3metal.te import convert_te, TextEncoder
from h3metal.weights import WeightFile
src = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3METAL_REF", "ref") + "/mini_te"
r = np.load(os.path.join(src, "ref.npz"))
gpu = Gpu()
for fmt in ("q8", "q4g", "f16"):
    out = os.path.join(src, f"te_{fmt}.h3m")
    convert_te(src, out, fmt=fmt, layers=3, log=lambda *a: None)
    for act in ("f16", "i8"):
        te = TextEncoder(gpu, WeightFile(gpu, out), act=act)
        h = te.encode(r["ids"])
        ref = r["h3"]
        rel = np.linalg.norm(h - ref) / np.linalg.norm(ref)
        print(f"{fmt} weights, {act} activations ({te.fmt}): hidden_states[3] rel L2 err {rel:.4f}")
