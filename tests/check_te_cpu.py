"""Runs a converted text encoder (.h3m) in NumPy on the CPU and compares
hidden_states[50] with reference embeddings of the official module
(tests/ref/reference.py --te-only).

It reads nothing but the converted file and mirrors what the GPU kernels
compute (rotated inputs, folded signs and norms) in float32, so it checks a
conversion made from the real checkpoint without the GPU, for instance
before the original weights are deleted. Text-only prompts.

    python tests/check_te_cpu.py TE.h3m REF.npz [REF.npz ...]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from h3metal.quant import RB, dequant, hadamard  # noqa: E402
from h3metal.te import mrope_angles  # noqa: E402
from h3metal.weights import bf16_to_f32, read_header  # noqa: E402

H = hadamard()  # 256-point, orthonormal: the kernels' rotation


class File:
    """Reads tensors of an .h3m file with plain reads (no mapping), one at a time."""

    def __init__(self, path: str):
        hdr = read_header(path)
        self.meta, self.entries = hdr["meta"], hdr["tensors"]
        self.path = path

    def _read(self, ent: dict, dtype) -> np.ndarray:
        n = ent["nbytes"] // np.dtype(dtype).itemsize
        return np.fromfile(self.path, dtype=dtype, count=n, offset=ent["offset"])

    def tensor(self, name: str) -> np.ndarray:
        e = self.entries[name]
        dt = {"f32": np.float32, "f16": np.float16, "bf16": np.uint16, "u8": np.uint8, "i8": np.int8}[e["dtype"]]
        return self._read(e, dt).reshape(e["shape"])

    def linear(self, name: str) -> np.ndarray:
        """The float [N][K] matrix a stored linear layer represents (in its rotated input space)."""
        e = self.entries[name]
        fmt, N, K = e["fmt"], e["N"], e["K"]
        if fmt in ("q8", "h8"):
            w = self._read(e["w"], np.int8).reshape(N, K)
        elif fmt in ("q4", "q4g", "h4", "h4g"):
            w = self._read(e["w"], np.uint8).reshape(N, K // 2)
        elif fmt == "f16":
            w = self._read(e["w"], np.float16).reshape(N, K)
        else:
            w = self._read(e["w"], np.uint16).reshape(N, K)
        sw = self._read(e["sw"], np.float32) if "sw" in e else None
        return dequant(fmt, w, sw, N, K, e.get("G", 128))


def rot(x: np.ndarray) -> np.ndarray:
    """Rows times blockdiag(H), as the kernels rotate a product's input."""
    T, K = x.shape
    return (x.reshape(T, K // RB, RB) @ H).reshape(T, K)


def rms(x: np.ndarray, w, eps: float) -> np.ndarray:
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x [T][heads][hd], rotate-half pairs (i, i + hd/2); cos, sin [T][hd/2]."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def forward(f: File, ids: np.ndarray, log=None) -> np.ndarray:
    c = f.meta["config"]
    D, Hq, Hkv, hd, F, L, eps = c["hidden"], c["heads"], c["kv_heads"], c["head_dim"], c["ffn"], c["layers"], c["eps"]
    T, rep = len(ids), Hq // Hkv
    x = bf16_to_f32(f.tensor("embed")[ids]).astype(np.float32)
    ang = mrope_angles(np.tile(np.arange(T), (3, 1)), hd, c["rope_theta"], c["mrope_section"])
    cos, sin = np.cos(ang), np.sin(ang)
    mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
    for l in range(L):
        p = f"layers.{l}."
        m1, m2 = f.tensor(p + "mod1")[0], f.tensor(p + "mod2")[0]
        qkv = rot(rms(x, m1[0], eps) + m1[1]) @ f.linear(p + "qkv").T
        q = qkv[:, :Hq * hd].reshape(T, Hq, hd)
        k = qkv[:, Hq * hd:(Hq + Hkv) * hd].reshape(T, Hkv, hd)
        v = qkv[:, (Hq + Hkv) * hd:].reshape(T, Hkv, hd)  # rows carry the signs the output product expects
        q = rope(rms(q, f.tensor(p + "q_norm"), eps), cos, sin)
        k = rope(rms(k, f.tensor(p + "k_norm"), eps), cos, sin)
        o = np.empty((T, Hq, hd), np.float32)
        for h in range(Hq):
            s = q[:, h] @ k[:, h // rep].T * np.float32(hd ** -0.5) + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            o[:, h] = (s / s.sum(-1, keepdims=True)) @ v[:, h // rep]
        x = x + rot(o.reshape(T, Hq * hd)) @ f.linear(p + "out").T
        ff = rot(rms(x, m2[0], eps) + m2[1]) @ f.linear(p + "fc1").T
        up, gate = ff[:, :F], ff[:, F:]  # up rows carry the signs of fc2's input
        x = x + rot(gate / (1.0 + np.exp(-gate)) * up) @ f.linear(p + "fc2").T
        if log:
            log(l)
    return x


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    f = File(sys.argv[1])
    lowest = 1.0
    for path in sys.argv[2:]:
        r = np.load(path)
        if "tags" in r and (r["tags"] == 0).any():
            print(f"{os.path.basename(path)}: an edit presentation, skipped (text-only check)")
            continue
        t = time.time()
        h = forward(f, r["ids"])
        ref = r["emb"].astype(np.float32)
        rel = np.linalg.norm(h - ref) / np.linalg.norm(ref)
        cos = (h * ref).sum(1) / (np.linalg.norm(h, axis=1) * np.linalg.norm(ref, axis=1) + 1e-12)
        rest = np.linalg.norm(h[1:] - ref[1:]) / np.linalg.norm(ref[1:])
        print(f"{os.path.basename(path)} ({len(r['ids'])} tokens, {time.time() - t:.0f} s): rel error {rel:.4f} "
              f"(after the first token {rest:.4f}), token cosine mean {cos.mean():.5f} min {cos.min():.5f}",
              flush=True)
        lowest = min(lowest, float(cos[1:].mean()))
    # A wrong layout, sign or scale leaves the tokens nearly unrelated to the reference (cosine far below 0.9);
    # rounding the weights alone keeps them aligned.
    ok = lowest >= 0.95
    print("PASS" if ok else "FAIL", f"(lowest mean token cosine {lowest:.5f})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
