"""Writes a transformer file with the real shapes and random weights, for timing and memory tests without the
checkpoint: python bench/synthetic_dit.py OUT.h3m q8|q4"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.weights import Writer, bf16_bits

out, fmt = sys.argv[1], sys.argv[2]
D, H, hd, F, L, steps = 5376, 56, 128, 14336, 50, 6
rng = np.random.default_rng(0)
meta = {"kind": "dit", "config": {"hidden": D, "layers": L, "heads": H, "head_dim": hd, "ffn": F, "refiner_layers": 2,
        "text_dim": 5120, "latents": 24, "patch": [1, 2, 2], "rope_freq": 16, "rope_theta": 10000.0, "norm_eps": 1e-5,
        "qk_eps": 1e-5, "final_eps": 1e-5},
        "timesteps": [0.0, 0.0325, 0.1125, 0.25, 0.4672, 0.768],
        "sigmas": [1.0, 0.9675, 0.8875, 0.75, 0.5328, 0.232, 0.0], "ref_t": 0.999, "fmts": {k: fmt for k in ("qkv", "out", "fc1", "fc2")},
        "refiner_fmt": "q8", "seed": 1, "synthetic": True}
w = Writer(out, meta)
def lin(name, f, N, K):
    if f == "q8":
        w.linear(name, "q8", N, K, rng.integers(-40, 41, (N, K), dtype=np.int8), np.full(N, 2e-3, np.float32))
    else:
        w.linear(name, "q4", N, K, rng.integers(0, 256, (N, K // 2), dtype=np.uint8), np.full(N, 4e-3, np.float32))
t = time.time()
w.tensor("x_embed.w", (rng.standard_normal((D, 96)) * 0.05).astype(np.float32), "f32")
w.tensor("x_embed.b", np.zeros(D, np.float32), "f32")
w.linear("ctx_embed", "bf16", D, 5120, bf16_bits((rng.standard_normal((D, 5120)) * 0.01).astype(np.float32)), None, bias=np.zeros(D, np.float32))
w.tensor("heads.w", (rng.standard_normal((steps, 96, D)) * 0.01).astype(np.float32), "f32")
w.tensor("heads.b", np.zeros((steps, 96), np.float32), "f32")
mod = np.zeros((steps, 3, 2, D), np.float32); mod[:, :, 0] = 1
w.tensor("final_mod", mod, "f32")
for prefix, n, f in (("refiner", 2, "q8"), ("blocks", L, fmt)):
    for i in range(n):
        lin(f"{prefix}.{i}.qkv", f, 3 * H * hd, D); lin(f"{prefix}.{i}.out", f, D, H * hd)
        lin(f"{prefix}.{i}.fc1", f, 2 * F, D); lin(f"{prefix}.{i}.fc2", f, D, F)
        w.tensor(f"{prefix}.{i}.q_norm", np.ones(hd, np.float32), "f32"); w.tensor(f"{prefix}.{i}.k_norm", np.ones(hd, np.float32), "f32")
        if prefix == "refiner":
            m = np.stack([np.ones(D), np.zeros(D)])[None].astype(np.float32)
            w.tensor(f"refiner.{i}.mod1", m, "f32"); w.tensor(f"refiner.{i}.mod2", m, "f32")
w.tensor("refiner.final_norm", np.ones(D, np.float32), "f32")
w.tensor("ones", np.ones((1, D), np.float32), "f32")
tab = np.zeros((steps, L, 3, 2, D), np.float32); tab[..., 0, :] = 1
w.tensor("mod_attn", tab, "f32"); w.tensor("mod_mlp", tab, "f32")
g = np.full((steps, L, 3, D), 0.05, np.float32)
w.tensor("gate_attn", g, "f32"); w.tensor("gate_mlp", g, "f32")
w.close()
print(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB) in {time.time() - t:.0f} s")
