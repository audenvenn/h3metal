"""Writes a text-encoder file with the real shapes and random int8 weights (24 GB): python bench/synthetic_te.py OUT"""
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.weights import Writer
out = sys.argv[1]
D, Hq, Hkv, hd, F, L = 5120, 64, 8, 128, 25600, 50
rng = np.random.default_rng(0)
w = Writer(out, {"kind": "te", "fmt": "q8", "seed": 2, "synthetic": True,
                 "config": {"hidden": D, "layers": L, "heads": Hq, "kv_heads": Hkv, "head_dim": hd, "ffn": F, "eps": 1e-6,
                            "rope_theta": 5e6, "mrope_section": [24, 20, 20], "vocab": 151936, "image_token_id": 151655}})
w.tensor("embed", rng.integers(0, 1 << 14, (151936, D), dtype=np.uint16), "bf16", host_only=True)
blk = rng.integers(-40, 41, (2 * F, D), dtype=np.int8)  # reused slices: writing is the cost, not randomness
t = time.time()
for l in range(L):
    for name, N, K in (("qkv", (Hq + 2 * Hkv) * hd, D), ("out", D, Hq * hd), ("fc1", 2 * F, D), ("fc2", D, F)):
        codes = blk[:N, :K] if K == D else rng.integers(-40, 41, (N, K), dtype=np.int8)
        w.linear(f"layers.{l}.{name}", "q8", N, K, np.ascontiguousarray(codes), np.full(N, 2e-3, np.float32))
    m = np.stack([np.ones(D), np.zeros(D)])[None].astype(np.float32)
    w.tensor(f"layers.{l}.mod1", m, "f32"); w.tensor(f"layers.{l}.mod2", m, "f32")
    w.tensor(f"layers.{l}.q_norm", np.ones(hd, np.float32), "f32"); w.tensor(f"layers.{l}.k_norm", np.ones(hd, np.float32), "f32")
w.tensor("ones", np.ones((1, D), np.float32), "f32")
w.close()
print(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB) in {time.time() - t:.0f} s")
