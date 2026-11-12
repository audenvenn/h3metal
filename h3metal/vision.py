"""The Qwen3-VL vision tower on the engine (edits only).

Turns each reference image into the features of its vision block in the
text encoder's input, plus the three DeepStack feature sets. Products run in
half precision on the matrix units with float accumulation; the residual
stream stays float. The tower's head dim (72) is padded to 96 for the
attention kernel: the padded channels are zero in queries, keys and values,
and the output projection has zero columns there, so nothing changes.
"""

from __future__ import annotations

import math

import numpy as np

from . import ops
from .edit import MERGE, pos_interp, vision_rope
from .gpu import Gpu
from .ops import QWeight
from .weights import WeightFile


def _pad(n: int, m: int) -> int:
    return -(-n // m) * m


class VisionTower:
    def __init__(self, gpu: Gpu, wf: WeightFile):
        self.gpu = gpu
        cfg = wf.meta["config"]
        self.depth, self.H, self.dim = cfg["depth"], cfg["num_heads"], cfg["hidden_size"]
        self.hd = self.dim // self.H
        self.hdp = _pad(self.hd, 32)
        self.ds_index = cfg["deepstack_visual_indexes"]
        self.side = int(math.isqrt(cfg["num_position_embeddings"]))
        h = wf.host

        def lin(name, k_map=None, n_pad=128, k_pad=None):
            """A float16 product with bias; k_map scatters the input columns into a wider (padded) input."""
            w = h(name + ".weight").astype(np.float32)
            w = w.reshape(w.shape[0], -1)
            b = h(name + ".bias").astype(np.float32)
            N, K = w.shape
            if k_map is not None:
                wide = np.zeros((N, k_map[1]), np.float32)
                wide[:, k_map[0]] = w
                w, K = wide, k_map[1]
            Np, Kp = _pad(N, n_pad), _pad(K, k_pad or 32)
            full = np.zeros((Np, Kp), np.float16)
            full[:N, :K] = w
            bias = np.zeros(Np, np.float32)
            bias[:N] = b
            return QWeight("f16", gpu.upload(full), None, Np, Kp, bias=gpu.upload(bias)), N

        def norm(name):
            return gpu.upload(h(name + ".weight").astype(np.float32)), gpu.upload(h(name + ".bias").astype(np.float32))

        # The output projection reads the padded attention output: head h's channels sit at h * hdp + [0, hd).
        cols = (np.arange(self.H)[:, None] * self.hdp + np.arange(self.hd)[None, :]).reshape(-1)
        ff = None
        self.blocks = []
        for i in range(self.depth):
            p = f"blocks.{i}."
            fc1, ff = lin(p + "mlp.linear_fc1")
            self.blocks.append({
                "n1": norm(p + "norm1"), "n2": norm(p + "norm2"),
                "qkv": lin(p + "attn.qkv")[0],
                "proj": lin(p + "attn.proj", k_map=(cols, self.H * self.hdp))[0],
                "fc1": fc1,
                "fc2": lin(p + "mlp.linear_fc2", k_pad=fc1.N)[0],  # reads fc1's padded output (zeros past ff)
            })
        self.ff_pad = self.blocks[0]["fc1"].N

        def merger(p):
            return {"n": norm(p + "norm"), "fc1": lin(p + "linear_fc1")[0], "fc2": lin(p + "linear_fc2")}
        self.merger = merger("merger.")
        self.ds = [merger(f"deepstack_merger_list.{i}.") for i in range(len(self.ds_index))]
        self.patch = lin("patch_embed.proj")[0]
        self.pos_embed = h("pos_embed.weight").astype(np.float32)

    def _merge(self, x, m, N: int, post: bool) -> np.ndarray:
        """2 x 2 patch merge: LayerNorm (after the merge with post, else per patch), MLP with exact GELU."""
        g, D = self.gpu, self.dim
        n, W = N // (MERGE * MERGE), D * MERGE * MERGE
        h = g.empty((ops.pad_rows(n, 64), W), "f16")
        if post:
            ops.layer_norm(g, x.view((n, W)), m["n"][0], m["n"][1], h, n, W, 1e-6)
        else:
            ops.layer_norm(g, x, m["n"][0], m["n"][1], h.view((N, D)), N, D, 1e-6)
        f1 = m["fc1"]
        t = g.empty((ops.pad_rows(n, 64), f1.N), "f32")
        ops.gemm(g, h, f1, t, n)
        a = g.empty((ops.pad_rows(n, 64), f1.N), "f16")
        ops.gelu(g, t, a, t.shape[0] * f1.N, exact=True)
        f2, n_out = m["fc2"]
        out = g.empty((ops.pad_rows(n, 64), f2.N), "f32")
        ops.gemm(g, a, f2, out, n)
        g.wait()
        return out.numpy()[:n, :n_out].copy()

    def __call__(self, patches: np.ndarray, grid):
        """patches [N][1536] of one image (edit.vision_patches) -> (embeds [N/4][out], [deepstack features])."""
        g, D, H, hd, hdp = self.gpu, self.dim, self.H, self.hd, self.hdp
        _, gh, gw = grid
        N = patches.shape[0]
        rows = ops.pad_rows(N, 128)
        # Residual stream starts as the interpolated position embedding; the patch embedding adds onto it.
        idx, wts = pos_interp(gh, gw, self.side)
        x = g.empty((rows, D), "f32")
        x.numpy()[:N] = (self.pos_embed[idx.reshape(-1)].reshape(N, 4, D) * wts[:, :, None]).sum(1)
        p16 = g.empty((rows, self.patch.K), "f16")
        p16.numpy()[:N, :patches.shape[1]] = patches
        ones = g.upload(np.ones((3, max(D, self.ff_pad)), np.float32))
        rt = g.upload(np.zeros(rows, np.uint8))
        ops.gemm(g, p16, self.patch, x, N, epi=1, rtype=rt, gate=ones)
        cos, sin = vision_rope(gh, gw, hd)
        cos, sin = g.upload(cos), g.upload(sin)
        h16 = g.empty((rows, max(D, self.ff_pad)), "f16")
        qkv = g.empty((rows, self.blocks[0]["qkv"].N), "f32")
        Q, K, V = (g.empty((H, rows, hdp), "f16") for _ in range(3))
        o = g.empty((rows, H * hdp), "f16")
        t = g.empty((rows, self.ff_pad), "f32")
        feats = []
        for i, b in enumerate(self.blocks):
            ops.layer_norm(g, x, b["n1"][0], b["n1"][1], h16.view((rows, D)), N, D, 1e-6)
            ops.gemm(g, h16.view((rows, D)), b["qkv"], qkv, N)
            ops.vit_qkv_rope(g, qkv, cos, sin, Q, K, V, N=N, H=H, hd=hd, hdp=hdp, rows=rows)
            ops.attention(g, Q, K, V, o, Lq=N, L=N, H=H, KVH=H, D=hdp, q_rows=rows, k_rows=rows,
                          scale=hd ** -0.5, wm=8, bk=32)
            ops.gemm(g, o, b["proj"], x, N, epi=1, rtype=rt, gate=ones)
            ops.layer_norm(g, x, b["n2"][0], b["n2"][1], h16.view((rows, D)), N, D, 1e-6)
            ops.gemm(g, h16.view((rows, D)), b["fc1"], t, N)
            ops.gelu(g, t, h16.view((rows, self.ff_pad)), rows * self.ff_pad, exact=False)
            ops.gemm(g, h16.view((rows, self.ff_pad)), b["fc2"], x, N, epi=1, rtype=rt, gate=ones)
            if i in self.ds_index:
                feats.append(self._merge(x, self.ds[self.ds_index.index(i)], N, post=True))
            g.commit()
        out = self._merge(x, self.merger, N, post=False)
        return out, feats
