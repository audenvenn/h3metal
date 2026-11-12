"""The MiniMax-H3-Image VAE decoder: a 24-layer ViT applied to overlapping
256-pixel tiles (16 x 16 latents each), blended linearly where they overlap,
exactly as the released pipeline decodes.

All tiles of a chunk run as one batch: their tokens are stacked, the matrix
products see every token at once, and attention runs per tile (the kernels
take a batch of short sequences).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from . import ops
from .gpu import Gpu, Tensor
from .st import SafeTensors
from .weights import WeightFile, Writer

PIXEL_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
PIXEL_STD = np.array([0.229, 0.224, 0.225], np.float32)


def convert_vae(src: str, out: str, fmt: str = "f16", log=print):
    """src: the MiniMax-H3-Image directory (vae/)."""
    from .quant import quantize
    with open(os.path.join(src, "vae", "config.json")) as f:
        cfg = json.load(f)
    st = SafeTensors(os.path.join(src, "vae", "diffusion_pytorch_model.safetensors"))
    kw = cfg["vit_decoder_kwargs"]
    heads, hd, layers = kw["heads"], kw["dim_head"], kw["num_layers"]
    D = heads * hd
    meta = {"kind": "vae", "config": {"dim": D, "heads": heads, "head_dim": hd, "layers": layers,
                                      "latent_channels": cfg["latent_channels"], "patch": 16,
                                      "registers": 4, "rope_theta": kw["rope_theta"],
                                      "rope_dim_ratio": kw["rope_dim_ratio"], "eps": 1e-5,
                                      "tile": cfg["vae_tile_size"], "tile_overlap": cfg["vae_tile_overlap_min"]},
            "latents_mean": cfg["latents_mean"], "latents_std": cfg["latents_std"], "fmt": fmt}
    w = Writer(out, meta)
    # post_quant_conv (1x1) folded into the token embedding: W = W_in W_pq, b = W_in b_pq + b_in.
    wpq = st.get("post_quant_conv.weight").reshape(24, 24)
    bpq = st.get("post_quant_conv.bias")
    win, bin_ = st.get("decoder.x_embedder.weight"), st.get("decoder.x_embedder.bias")
    w.tensor("embed.w", (win.astype(np.float64) @ wpq.astype(np.float64)).astype(np.float32), "f32")
    w.tensor("embed.b", (win.astype(np.float64) @ bpq.astype(np.float64) + bin_).astype(np.float32), "f32")
    w.tensor("registers", st.get("decoder.register_tokens").reshape(-1, D), "f32")

    def lin(name, mat, bias):
        codes, scales = quantize(fmt, mat)
        w.linear(name, fmt, mat.shape[0], mat.shape[1], codes, scales, bias=bias)

    for i in range(layers):
        p = f"decoder.transformer_blocks.{i}"
        qkv = st.get(f"{p}.attn.to_qkv.weight").reshape(heads, 3, hd, D).transpose(1, 0, 2, 3).reshape(3 * D, D)
        qb = st.get(f"{p}.attn.to_qkv.bias").reshape(heads, 3, hd).transpose(1, 0, 2).reshape(3 * D)
        lin(f"blocks.{i}.qkv", qkv, qb)
        lin(f"blocks.{i}.out", st.get(f"{p}.attn.to_out.weight"), st.get(f"{p}.attn.to_out.bias"))
        w1, b1 = st.get(f"{p}.ff.w1.weight"), st.get(f"{p}.ff.w1.bias")  # [gate; up]
        F = w1.shape[0] // 2
        lin(f"blocks.{i}.fc1", np.concatenate([w1[F:], w1[:F]]), np.concatenate([b1[F:], b1[:F]]))
        lin(f"blocks.{i}.fc2", st.get(f"{p}.ff.w2.weight"), st.get(f"{p}.ff.w2.bias"))
        w.tensor(f"blocks.{i}.mod1", np.stack([st.get(f"{p}.norm1.weight"), np.zeros(D, np.float32)])[None], "f32")
        w.tensor(f"blocks.{i}.mod2", np.stack([st.get(f"{p}.norm2.weight"), np.zeros(D, np.float32)])[None], "f32")
        w.tensor(f"blocks.{i}.scale1", st.get(f"{p}.scale1")[None], "f32")
        w.tensor(f"blocks.{i}.scale2", st.get(f"{p}.scale2")[None], "f32")
    w.tensor("norm_out.w", st.get("decoder.norm_out.weight"), "f32")
    w.tensor("norm_out.b", st.get("decoder.norm_out.bias"), "f32")
    lin("proj_out", st.get("decoder.proj_out.weight"), st.get("decoder.proj_out.bias"))
    w.tensor("ones", np.ones(hd, np.float32), "f32")
    # The encoder (edits only), as stored.
    for k in st.keys():
        if k.startswith(("encoder.", "quant_conv.")):
            w.tensor(k, np.ascontiguousarray(st.get(k), np.float32), "f32")
    w.close()
    log(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB)")


def split_tiles(length: int, tile: int = 256, min_overlap: int = 64, ratio: int = 16):
    """diffusers AutoencoderKLMiniMaxH3Image._split_tiles."""
    if tile >= length:
        return [0], [length], []
    n = math.ceil(length / tile)
    while tile * n - min_overlap * (n - 1) - length < 0:
        n += 1
    overlaps = [min_overlap] * (n - 1)
    remaining = tile * n - sum(overlaps) - length
    for i in range(remaining // ratio):
        overlaps[i % (n - 1)] += ratio
    starts = [0]
    for i in range(n - 1):
        starts.append(starts[-1] + tile - overlaps[i])
    return starts, [tile] * n, overlaps


def _blend(a: np.ndarray, b: np.ndarray, extent: int, axis: int) -> np.ndarray:
    extent = min(a.shape[axis], b.shape[axis], extent)
    pos = np.arange(extent, dtype=np.float32)
    shape = [1] * a.ndim
    shape[axis] = extent
    wa = (1 - pos / extent).reshape(shape)
    wb = (pos / extent).reshape(shape)
    sa = [slice(None)] * a.ndim
    sa[axis] = slice(-extent, None)
    sb = [slice(None)] * b.ndim
    sb[axis] = slice(0, extent)
    blended = a[tuple(sa)] * wa + b[tuple(sb)] * wb
    if extent == b.shape[axis]:
        return blended
    sr = [slice(None)] * b.ndim
    sr[axis] = slice(extent, None)
    return np.concatenate([blended, b[tuple(sr)]], axis=axis)


def stitch(tiles, h_overlaps, w_overlaps) -> np.ndarray:
    """diffusers AutoencoderKLMiniMaxH3Image._stitch_tiles, tiles [C][h][w]."""
    rows = []
    for i, row in enumerate(tiles):
        out_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = _blend(tiles[i - 1][j], tile, h_overlaps[i - 1], axis=-2)
            if j > 0:
                tile = _blend(row[j - 1], tile, w_overlaps[j - 1], axis=-1)
            if i < len(tiles) - 1:
                tile = tile[..., : -h_overlaps[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, : -w_overlaps[j]]
            out_row.append(tile)
        rows.append(np.concatenate(out_row, axis=-1))
    return np.concatenate(rows, axis=-2)


class VAEDecoder:
    def __init__(self, gpu: Gpu, wf: WeightFile, chunk_tokens: int = 1 << 15):
        self.gpu, self.wf = gpu, wf
        c = wf.meta["config"]
        self.cfg = c
        self.D, self.H, self.hd, self.L = c["dim"], c["heads"], c["head_dim"], c["layers"]
        self.mean = np.array(wf.meta["latents_mean"], np.float32)
        self.std = np.array(wf.meta["latents_std"], np.float32)
        self.embed_w, self.embed_b = wf.host("embed.w").copy(), wf.host("embed.b").copy()
        self.registers = wf.host("registers").copy()
        lin = wf.linear
        self.blocks = [{k: lin(f"blocks.{i}.{k}") for k in ("qkv", "out", "fc1", "fc2")} |
                       {k: wf.tensor(f"blocks.{i}.{k}") for k in ("mod1", "mod2", "scale1", "scale2")}
                       for i in range(self.L)]
        self.norm_w, self.norm_b = wf.tensor("norm_out.w"), wf.tensor("norm_out.b")
        self.proj_out = lin("proj_out")
        self.ones = wf.tensor("ones")
        self.chunk_tokens = chunk_tokens
        self.extra = c["registers"] + 1

    def _rope(self, th: int, tw: int):
        """cos/sin [rows][rot/2] for a th x tw tile plus its register and zero tokens (all at position 0)."""
        rot = int(self.hd * self.cfg["rope_dim_ratio"])
        n_axes = 2
        inv = (1.0 / self.cfg["rope_theta"] ** np.arange(0, 1, 2 * n_axes / rot, dtype=np.float32)).astype(np.float32)
        gh = (2.0 * (np.arange(0.5, th, dtype=np.float32) / th) - 1.0).astype(np.float32)
        gw = (2.0 * (np.arange(0.5, tw, dtype=np.float32) / tw) - 1.0).astype(np.float32)
        hh, ww = np.meshgrid(gh, gw, indexing="ij")
        pos = np.stack([hh.reshape(-1), ww.reshape(-1)], -1)
        pos = np.concatenate([pos, np.zeros((self.extra, 2), np.float32)])
        ang = (np.float32(2.0 * math.pi) * pos[:, :, None] * inv[None, None, :]).astype(np.float32)
        ang = ang.reshape(len(pos), -1)  # [rows][rot/2]
        return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32), rot

    def _run_tiles(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        """Decodes same-shaped latent tiles [C][th][tw] -> pixel tiles [3][th*16][tw*16] (normalized space)."""
        g, D, H, hd = self.gpu, self.D, self.H, self.hd
        C, th, tw = tiles[0].shape
        n_tok = th * tw
        seq = n_tok + self.extra
        nt = len(tiles)
        M = nt * seq
        Mp = ops.pad_rows(M, 128)
        rows_pad = (seq + 31) // 32 * 32
        cos, sin, rot = self._rope(th, tw)
        x = g.empty((Mp, D), "f32")
        xv = x.numpy()
        for t, z in enumerate(tiles):
            tok = z.reshape(C, n_tok).T  # row-major over (h, w)
            base = t * seq
            xv[base:base + n_tok] = tok @ self.embed_w.T + self.embed_b
            xv[base + n_tok:base + n_tok + self.extra - 1] = self.registers
            xv[base + seq - 1] = 0.0
        hn = g.empty((Mp, D), "f16")
        wide = g.empty((Mp, 8 * D), "f16")
        o = g.empty((Mp, D), "f16")
        Q = g.empty((nt * H, rows_pad, hd), "f16")
        K = g.empty((nt * H, rows_pad, hd), "f16")
        V = g.empty((nt * H, rows_pad, hd), "f16")
        rt = g.empty((Mp,), "u8")
        ffo = g.empty((Mp, 4 * D), "f16")
        cs, sn = g.upload(cos), g.upload(sin)
        eps = self.cfg["eps"]
        for blk in self.blocks:
            ops.norm_mod_q8(g, x, blk["mod1"], rt, hn, None, M, D, eps=eps, out_h=True)
            qkv = wide.view((Mp, 3 * D))
            ops.gemm(g, hn, blk["qkv"], qkv, M)
            ops.qk_norm_rope(g, qkv, self.ones, self.ones, cs, sn, Q, K, V, M=M, Hq=H, Hk=H, Hv=H, hd=hd,
                             out_rows=rows_pad, rot=rot, eps=eps, seq_len=seq)
            ops.attention(g, Q, K, V, o, Lq=seq, L=seq, H=nt * H, KVH=nt * H, D=hd, q_rows=rows_pad,
                          k_rows=rows_pad, scale=hd ** -0.5, wm=4, bk=32, seq_heads=H, seq_rows=seq)
            ops.gemm(g, o, blk["out"], x, M, epi=1, rtype=rt, gate=blk["scale1"])
            ops.norm_mod_q8(g, x, blk["mod2"], rt, hn, None, M, D, eps=eps, out_h=True)
            ff = wide.view((Mp, 8 * D))
            ops.gemm(g, hn, blk["fc1"], ff, M)
            ops.swiglu(g, ff, ffo, M, 4 * D)
            ops.gemm(g, ffo, blk["fc2"], x, M, epi=1, rtype=rt, gate=blk["scale2"])
            g.commit()
        ops.layer_norm(g, x, self.norm_w, self.norm_b, hn, M, D, eps)
        out = g.empty((Mp, self.proj_out.N), "f32")
        ops.gemm(g, hn, self.proj_out, out, M)
        g.wait()
        res = out.numpy()
        p = self.cfg["patch"]
        pix = []
        for t in range(nt):
            r = res[t * seq:t * seq + n_tok].reshape(th, tw, 3, p, p).transpose(2, 0, 3, 1, 4)
            pix.append(r.reshape(3, th * p, tw * p).copy())
        return pix

    def decode(self, latents: np.ndarray, raw: bool = False) -> np.ndarray:
        """Normalized latents [C][h][w] -> image [3][16h][16w] float in [0, 1] (raw: the decoder's own
        ImageNet-normalized output, before the pipeline maps it back to pixels)."""
        z = latents * self.std[:, None, None] + self.mean[:, None, None]
        p = self.cfg["patch"]
        Hp, Wp = z.shape[1] * p, z.shape[2] * p
        ys, yl, yo = split_tiles(Hp, self.cfg["tile"], self.cfg["tile_overlap"], p)
        xs, xl, xo = split_tiles(Wp, self.cfg["tile"], self.cfg["tile_overlap"], p)
        jobs = [(i, j, z[:, y // p:(y + ly) // p, x // p:(x + lx) // p])
                for i, (y, ly) in enumerate(zip(ys, yl)) for j, (x, lx) in enumerate(zip(xs, xl))]
        out = {}
        per = max(1, self.chunk_tokens // (jobs[0][2].shape[1] * jobs[0][2].shape[2] + self.extra))
        for c in range(0, len(jobs), per):
            batch = jobs[c:c + per]
            for (i, j, _), pix in zip(batch, self._run_tiles([b[2] for b in batch])):
                out[(i, j)] = pix
        tiles = [[out[(i, j)] for j in range(len(xs))] for i in range(len(ys))]
        img = stitch(tiles, yo, xo)
        if raw:
            return img
        return np.clip(img * PIXEL_STD[:, None, None] + PIXEL_MEAN[:, None, None], 0.0, 1.0)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """[3][H][W] in [0, 1] -> [H][W][3] uint8, truncating like the released pipeline."""
    return np.clip(img * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)
