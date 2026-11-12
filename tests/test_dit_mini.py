"""Mini DiT with random weights against the official diffusers module (see tests/ref/mini_dit.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu
from h3metal.convert import convert_dit
from h3metal.weights import WeightFile, bf16_bits
from h3metal.dit import DiT, Layout

src = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3METAL_REF", "ref") + "/mini_dit"
fmts = sys.argv[2] if len(sys.argv) > 2 else "q8"
out = os.path.join(src, f"dit_{fmts}.h3m")
convert_dit(src, out, fmts={k: fmts for k in ("qkv", "out", "fc1", "fc2")}, refiner_fmt="q8", log=lambda *a: None)
gpu = Gpu()
dit = DiT(gpu, WeightFile(gpu, out))
for case in ("t2i", "edit"):
    r = np.load(os.path.join(src, f"ref_{case}.npz"))
    T = r["text"].shape[0]
    text = gpu.empty((64, r["text"].shape[1]), "bf16")
    text.numpy()[:T] = bf16_bits(r["text"])
    rows = dit.encode_text(text, T)
    refs = [tuple(x) for x in r["refs"].tolist()]
    lay = Layout(T, np.where(r["tags"] == 1, 1, 0).astype(np.uint8), refs, int(r["lat_hw"][0]), int(r["lat_hw"][1]))
    dit.prepare(lay)
    for s in range(6):
        v = dit.step(s, rows, r["latents"])
        ref = r[f"vel{s}"]
        rel = np.linalg.norm(v - ref) / np.linalg.norm(ref)
        cos = (v * ref).sum() / np.linalg.norm(v) / np.linalg.norm(ref)
        print(f"{case} step {s}: rel L2 err {rel:.4f}  cosine {cos:.6f}")
