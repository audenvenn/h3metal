"""The MiniMax-H3-Image transformer (DiT) on the GPU.

One packed sequence holds the text rows, the rows of the reference images
(edits) and the rows of the generated image. Every row has a type that
selects its precomputed modulation (h3metal/convert.py):

    0  image rows at the step's timestep (also the vision-block rows of the text)
    1  text rows at the step's timestep
    2  reference-image rows, held at t = 0.999
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import ops
from .gpu import Gpu, Tensor
from .weights import WeightFile

ROPE_SPATIAL_SCALE = 32


def spatial_grid(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    """One aspect-normalized rotary axis (float64, like the released pipeline)."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * ROPE_SPATIAL_SCALE


def frame_grid(lat_h: int, lat_w: int, patch: int = 2) -> np.ndarray:
    sqrt_area = np.sqrt(lat_h * lat_w)
    gh = spatial_grid(lat_h, patch, sqrt_area)
    gw = spatial_grid(lat_w, patch, sqrt_area)
    hh, ww = np.meshgrid(gh, gw, indexing="ij")
    return np.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1)


@dataclass
class Layout:
    """The packed sequence of one request."""
    n_text: int
    text_types: np.ndarray        # [n_text] 1 text, 0 vision block
    refs: list                    # [(lat_h, lat_w)] of the reference images
    lat_h: int
    lat_w: int
    patch: int = 2

    @property
    def n_ref_rows(self) -> int:
        return sum((h // self.patch) * (w // self.patch) for h, w in self.refs)

    @property
    def n_img_rows(self) -> int:
        return (self.lat_h // self.patch) * (self.lat_w // self.patch)

    @property
    def length(self) -> int:
        return self.n_text + self.n_ref_rows + self.n_img_rows

    def positions(self) -> np.ndarray:
        """(t, h, w) rotary coordinates of every row, float64 (build_image_packed_sequence)."""
        pos = np.zeros((self.length, 3), np.float64)
        pos[:self.n_text, 0] = np.arange(self.n_text)
        cur, t = self.n_text, float(self.n_text)
        for h, w in self.refs:
            n = (h // self.patch) * (w // self.patch)
            pos[cur:cur + n, 0] = t
            pos[cur:cur + n, 1:] = frame_grid(h, w, self.patch)
            cur += n
            t += 1.0
        pos[cur:, 0] = t
        pos[cur:, 1:] = frame_grid(self.lat_h, self.lat_w, self.patch)
        return pos

    def row_types(self) -> np.ndarray:
        rt = np.zeros(self.length, np.uint8)
        rt[:self.n_text] = self.text_types
        rt[self.n_text:self.n_text + self.n_ref_rows] = 2
        return rt


class DiT:
    def __init__(self, gpu: Gpu, wf: WeightFile):
        self.gpu, self.wf = gpu, wf
        c = wf.meta["config"]
        self.cfg = c
        self.D, self.H, self.hd, self.F = c["hidden"], c["heads"], c["head_dim"], c["ffn"]
        self.L = c["layers"]
        self.inner = self.H * self.hd
        self.timesteps = wf.meta["timesteps"]
        self.sigmas = wf.meta["sigmas"]
        self.n_steps = len(self.timesteps)
        lin = wf.linear
        self.blocks = [{k: lin(f"blocks.{l}.{k}") for k in ("qkv", "out", "fc1", "fc2")} |
                       {"q_norm": wf.tensor(f"blocks.{l}.q_norm"), "k_norm": wf.tensor(f"blocks.{l}.k_norm")}
                       for l in range(self.L)]
        self.refiner = [{k: lin(f"refiner.{i}.{k}") for k in ("qkv", "out", "fc1", "fc2")} |
                        {k: wf.tensor(f"refiner.{i}.{k}") for k in ("q_norm", "k_norm", "mod1", "mod2")}
                        for i in range(c["refiner_layers"])]
        self.ctx_embed = lin("ctx_embed")
        tab = np.zeros((1, 2, self.D), np.float32)
        tab[0, 0] = wf.host("refiner.final_norm")
        self.refiner_final_mod = gpu.upload(tab)
        self.ones = wf.tensor("ones")
        self.x_embed_w, self.x_embed_b = wf.tensor("x_embed.w"), wf.tensor("x_embed.b")
        self.mod_attn, self.mod_mlp = wf.tensor("mod_attn"), wf.tensor("mod_mlp")
        self.gate_attn, self.gate_mlp = wf.tensor("gate_attn"), wf.tensor("gate_mlp")
        self.final_mod = wf.tensor("final_mod")
        self.heads_w, self.heads_b = wf.tensor("heads.w"), wf.tensor("heads.b")
        self.patch_dim = self.x_embed_w.shape[1]
        self._bufs = None
        self.attn_cfg = dict(wm=16, bk=32)

    # -- buffers ---------------------------------------------------------------------------------------------------
    def _alloc(self, S: int):
        g, D = self.gpu, self.D
        Sp = ops.pad_rows(S, 128)
        if self._bufs and self._bufs["Sp"] >= Sp:
            return self._bufs
        self._bufs = None
        big = max(3 * self.inner, 2 * self.F)
        b = {
            "Sp": Sp,
            "x": g.empty((Sp, D), "f32"),
            "a8": g.empty((Sp, max(self.F, self.inner, D)), "i8"),
            "sa": g.empty((Sp,), "f32"),
            "wide": g.empty((Sp, big), "bf16"),
            "o": g.empty((Sp, self.inner), "bf16"),
            "Q": g.empty((self.H, Sp, self.hd), "f16"),
            "K": g.empty((self.H, Sp, self.hd), "f16"),
            "V": g.empty((self.H, Sp, self.hd), "f16"),
            "rt": g.empty((Sp,), "u8"),
            "rt0": g.empty((Sp,), "u8"),
            "cos": g.empty((Sp, 3 * self.cfg["rope_freq"]), "f32"),
            "sin": g.empty((Sp, 3 * self.cfg["rope_freq"]), "f32"),
            "y": g.empty((Sp, D), "f32"),
            "lat": g.empty((Sp, self.patch_dim), "f32"),
            "vel": g.empty((Sp, self.patch_dim), "f32"),
        }
        self._bufs = b
        return b

    def free(self):
        self._bufs = None

    def _prefetch(self, l: int):
        """Asks the kernel to read block l's weights ahead (a no-op when they are already in memory)."""
        if 0 <= l < self.L:
            self.wf.prefetch([f"blocks.{l}.{k}" for k in ("qkv", "out", "fc1", "fc2")])

    # -- pieces ----------------------------------------------------------------------------------------------------
    def _block(self, b, blk, S: int, x: Tensor, mod1: Tensor, mod2: Tensor, g1: Tensor, g2: Tensor, rt: Tensor,
               rot: int, mod_stride: int, g_stride: int):
        g, D, H, hd, F, inner = self.gpu, self.D, self.H, self.hd, self.F, self.inner
        Sp = b["Sp"]
        eps = self.cfg["norm_eps"]
        a8, sa = b["a8"], b["sa"]
        ops.norm_mod_q8(g, x, mod1, rt, a8.view((Sp, D)), sa, S, D, eps=eps, mod_stride=mod_stride)
        qkv = b["wide"].view((Sp, 3 * inner))
        ops.gemm(g, a8.view((Sp, D)), blk["qkv"], qkv, S, sa=sa, **_tile(blk["qkv"]))
        ops.qk_norm_rope(g, qkv, blk["q_norm"], blk["k_norm"], b["cos"], b["sin"], b["Q"], b["K"], b["V"], M=S,
                         Hq=H, Hk=H, Hv=H, hd=hd, out_rows=Sp, rot=rot, eps=self.cfg["qk_eps"])
        ops.attention(g, b["Q"], b["K"], b["V"], b["o"], Lq=S, L=S, H=H, KVH=H, D=hd, q_rows=Sp, k_rows=Sp,
                      scale=hd ** -0.5, **self.attn_cfg)
        ops.rot_q8(g, b["o"], a8.view((Sp, inner)), sa, S, inner)
        ops.gemm(g, a8.view((Sp, inner)), blk["out"], x, S, sa=sa, epi=1, rtype=rt, gate=g1, g_stride=g_stride,
                 **_tile(blk["out"]))
        ops.norm_mod_q8(g, x, mod2, rt, a8.view((Sp, D)), sa, S, D, eps=eps, mod_stride=mod_stride)
        h = b["wide"].view((Sp, 2 * F))
        ops.gemm(g, a8.view((Sp, D)), blk["fc1"], h, S, sa=sa, **_tile(blk["fc1"]))
        ops.rot_q8(g, h, a8.view((Sp, F)), sa, S, F, swiglu=True)
        ops.gemm(g, a8.view((Sp, F)), blk["fc2"], x, S, sa=sa, epi=1, rtype=rt, gate=g2, g_stride=g_stride,
                 **_tile(blk["fc2"]))

    def encode_text(self, text: Tensor, T: int) -> Tensor:
        """Projects and refines the text-encoder hidden states [T][text_dim] (bf16) -> [T][D] float rows."""
        g, D = self.gpu, self.D
        b = self._alloc(T)
        Tp = b["Sp"]
        out = g.empty((Tp, D), "f32")
        ops.gemm(g, text, self.ctx_embed, out, T, tile=(64, 128, 4))
        for blk in self.refiner:
            self._block(b, blk, T, out, blk["mod1"], blk["mod2"], self.ones, self.ones, b["rt0"], 0,
                        2 * D, D)
        res = g.empty((Tp, D), "f32")
        ops.norm_mod_q8(g, out, self.refiner_final_mod, b["rt0"], res, None, T, D,
                        eps=self.cfg["final_eps"], out_f=True)
        g.wait()
        return res

    def prepare(self, layout: Layout):
        """Per-request constants: row types and rotary tables."""
        S = layout.length
        b = self._alloc(S)
        rt = b["rt"].numpy()
        rt[:] = 0
        rt[:S] = layout.row_types()
        pos = layout.positions().astype(np.float32)
        rf = self.cfg["rope_freq"]
        inv = (1.0 / (self.cfg["rope_theta"] ** (np.arange(0, 2 * rf, 2, dtype=np.float32) / (2 * rf)))).astype(
            np.float32)
        ang = (pos[:, :, None] * inv[None, None, :]).astype(np.float32).reshape(S, 3 * rf)  # [S][48]
        cos, sin = b["cos"].numpy(), b["sin"].numpy()
        cos[:] = 1.0
        sin[:] = 0.0
        cos[:S, :3 * rf] = np.cos(ang)
        sin[:S, :3 * rf] = np.sin(ang)
        self.layout = layout
        self.rot = 2 * 3 * rf

    def step(self, s: int, text_rows: Tensor, latents: np.ndarray) -> np.ndarray:
        """One denoising step: latents [n_ref + n_img][patch_dim] (references first) -> velocity of all image rows."""
        g, D, lay = self.gpu, self.D, self.layout
        S, T = lay.length, lay.n_text
        b = self._bufs
        Sp, x = b["Sp"], b["x"]
        n_img = lay.n_ref_rows + lay.n_img_rows
        lat = b["lat"].numpy()
        lat[:n_img] = latents
        ops.copy_f32(g, text_rows, x, T * D)
        ops.linear_f32(g, b["lat"], self.x_embed_w, self.x_embed_b, x.row(T, n_img), M=n_img, N=D,
                       K=self.patch_dim)
        L3 = 3 * 2 * D  # elements per (step, layer) in the modulation tables
        self._prefetch(0)
        for l, blk in enumerate(self.blocks):
            self._prefetch(l + 1)
            o = (s * self.L + l)
            mod1 = self.mod_attn.view((3, 2, D), offset_elems=o * L3)
            mod2 = self.mod_mlp.view((3, 2, D), offset_elems=o * L3)
            g1 = self.gate_attn.view((3, D), offset_elems=o * 3 * D)
            g2 = self.gate_mlp.view((3, D), offset_elems=o * 3 * D)
            self._block(b, blk, S, x, mod1, mod2, g1, g2, b["rt"], self.rot, 2 * D, D)
            g.commit()  # one command buffer per block: only its weights are pinned while it runs
        # Output norm and this step's head, over the image rows.
        fm = self.final_mod.view((3, 2, D), offset_elems=s * L3)
        y = b["y"]
        rt_img = Tensor(b["rt"].buf, b["rt"].offset + T, (n_img,), "u8")
        ops.norm_mod_q8(g, x.row(T, n_img), fm, rt_img, y, None, n_img, D, eps=self.cfg["final_eps"], out_f=True)
        hw = self.heads_w.view((self.patch_dim, D), offset_elems=s * self.patch_dim * D)
        hb = self.heads_b.view((self.patch_dim,), offset_elems=s * self.patch_dim)
        ops.linear_f32(g, y, hw, hb, b["vel"], M=n_img, N=self.patch_dim, K=D)
        g.wait()
        return b["vel"].numpy()[:n_img].copy()


def _tile(w) -> dict:
    """Tile shape and dispatch order for a weight, from measurements on an M6 (bench/gemm_bench.py)."""
    if w.K >= 12000:  # long inner dimension (fc2): wide tiles reuse each activation tile more
        return {"tile": (64, 256, 8)} if w.fmt == "q8" else {"tile": (32, 128, 2)}
    if w.K <= 6000:   # qkv, fc1: sweep several row tiles per weight tile while it is cached
        return {"tile": (64, 128, 4), "swz": 4}
    return {"tile": (64, 128, 4)}


def patchify(lat: np.ndarray, p: int = 2) -> np.ndarray:
    """[C][H][W] -> rows [(H/p)(W/p)][C p p], row-major over the patch grid (channel, then dy, dx)."""
    C, Hh, W = lat.shape
    x = lat.reshape(C, Hh // p, p, W // p, p).transpose(1, 3, 0, 2, 4)
    return np.ascontiguousarray(x.reshape(-1, C * p * p))


def unpatchify(rows: np.ndarray, C: int, Hh: int, W: int, p: int = 2) -> np.ndarray:
    x = rows.reshape(Hh // p, W // p, C, p, p).transpose(2, 0, 3, 1, 4)
    return np.ascontiguousarray(x.reshape(C, Hh, W))


def euler(latents: np.ndarray, vel: np.ndarray, s: int, sigmas) -> np.ndarray:
    """x_next = x + (sigma - sigma_next) * v, float32 (the released velocity update)."""
    d = np.float32(np.float32(sigmas[s]) - np.float32(sigmas[s + 1]))
    return (latents.astype(np.float32) + d * vel.astype(np.float32)).astype(np.float32)


def num_rows(h: int, w: int) -> int:
    return (h // 32) * (w // 32)


def canvas_size(height: int, width: int, min_area: int = 2048 * 2048, multiple: int = 32):
    scale = max(1.0, math.sqrt(min_area / (width * height)))
    return (max(multiple, round(height * scale / multiple) * multiple),
            max(multiple, round(width * scale / multiple) * multiple))
