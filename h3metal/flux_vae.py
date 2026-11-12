"""The FLUX autoencoder's decoder (diffusers AutoencoderKL with 16 latent channels), as Z-Image uses it.

A convolutional decoder: conv_in, a middle block (ResNet block, single-head self-attention over all latent
pixels, ResNet block), four up blocks of three ResNet blocks each (the first three end in a 2x nearest
upsampling and a convolution), GroupNorm + SiLU and conv_out.

On the engine the 3x3 convolutions run as nine accumulating half-precision products per output tile whose
left operands are views into the input stored with a one-pixel zero border (kernels/conv.metal): GroupNorm +
SiLU writes straight into that layout, and the 2x upsampling writes its padded output in one pass. Only
conv_in (16 input channels) goes through im2col. The
attention has one head of 512 channels, too wide for the flash-attention kernel's registers, so its scores
are materialized a chunk of query rows at a time: Q K^T, a row softmax, P V, all on the matrix units.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from . import ops
from .gpu import Gpu, Tensor
from .ops import QWeight
from .st import SafeTensors
from .weights import WeightFile, Writer

STRIPE_BYTES = 256 << 20
CONV_TILE = (64, 128, 8)  # conv3x3: output pixels and channels per threadgroup, SIMD-groups (swept on an M6)


def convert_flux_vae(src: str, out: str, log=print):
    """src: a directory with vae/config.json and vae/diffusion_pytorch_model.safetensors; the decoder only."""
    d = os.path.join(src, "vae")
    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)
    st = SafeTensors(os.path.join(d, "diffusion_pytorch_model.safetensors"))
    keep = {k: cfg[k] for k in ("block_out_channels", "layers_per_block", "latent_channels", "norm_num_groups",
                                "scaling_factor", "shift_factor")}
    w = Writer(out, {"kind": "flux-vae", "config": keep})
    for k in sorted(st.keys()):
        if k.startswith("decoder."):
            w.tensor(k, np.ascontiguousarray(st.get(k), np.float32), "f32")
    w.close()
    log(f"wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB)")


class FluxVAEDecoder:
    def __init__(self, gpu: Gpu, wf: WeightFile):
        self.gpu = gpu
        c = wf.meta["config"]
        self.cfg = c
        h = wf.host
        self.G = c["norm_num_groups"]

        def conv(name):
            w = h(name + ".weight").astype(np.float32)
            b = h(name + ".bias").astype(np.float32)
            cout, cin, kh, kw = w.shape
            m = w.transpose(0, 2, 3, 1).reshape(cout, kh * kw * cin)
            K = -(-m.shape[1] // 32) * 32
            N = -(-cout // 128) * 128
            full = np.zeros((N, K), np.float16)
            full[:cout, :m.shape[1]] = m
            bias = np.zeros(N, np.float32)
            bias[:cout] = b
            return QWeight("f16", gpu.upload(full), None, N, K, bias=gpu.upload(bias)), kh, cin, cout

        def lin(name, bias=None):
            w = h(name + ".weight").astype(np.float32)
            if w.ndim == 4:  # a 1x1 convolution stored as such
                w = w[:, :, 0, 0]
            b = h(name + ".bias").astype(np.float32) if bias is None else bias
            return QWeight("f16", gpu.upload(w.astype(np.float16)), None, w.shape[0], w.shape[1],
                           bias=gpu.upload(b.astype(np.float32)))

        def norm(name):
            return gpu.upload(h(name + ".weight").astype(np.float32)), gpu.upload(h(name + ".bias").astype(np.float32))

        def resnet(p):
            r = {"n1": norm(p + ".norm1"), "c1": conv(p + ".conv1"), "n2": norm(p + ".norm2"), "c2": conv(p + ".conv2")}
            if p + ".conv_shortcut.weight" in wf:
                r["sc"] = conv(p + ".conv_shortcut")
            return r

        self.conv_in = conv("decoder.conv_in")
        self.mid = [resnet("decoder.mid_block.resnets.0"), resnet("decoder.mid_block.resnets.1")]
        a = "decoder.mid_block.attentions.0"
        self.attn_norm = norm(a + ".group_norm")
        self.wq, self.wk = lin(a + ".to_q"), lin(a + ".to_k")
        wv = h(a + ".to_v.weight").astype(np.float32)
        bv = h(a + ".to_v.bias").astype(np.float32)
        wo = h(a + ".to_out.0.weight").astype(np.float32)
        # Softmax rows sum to one, so the value bias passes through the attention: fold it into to_out's bias.
        self.wo = lin(a + ".to_out.0", bias=wo @ bv + h(a + ".to_out.0.bias").astype(np.float32))
        self.wv_act = gpu.upload(wv.astype(np.float16))  # V^T = W_v x^T: W_v as the left operand
        n_up = len(c["block_out_channels"])
        self.ups = []
        for i in range(n_up):
            p = f"decoder.up_blocks.{i}"
            res = [resnet(f"{p}.resnets.{j}") for j in range(c["layers_per_block"] + 1)]
            up = conv(f"{p}.upsamplers.0.conv") if f"{p}.upsamplers.0.conv.weight" in wf else None
            self.ups.append((res, up))
        self.norm_out = norm("decoder.conv_norm_out")
        self.conv_out = conv("decoder.conv_out")
        self._scratch = {}
        self._ones = {}
        self.conv_tile = CONV_TILE

    # -- helpers -----------------------------------------------------------------------------------------------
    # The matrix-product kernel runs whole 64-row tiles: every [rows][channels] buffer has room for the padded
    # rows, and the stripes of a convolution start on tile boundaries.
    def _buf(self, role: str, shape, dtype: str) -> Tensor:
        shape = (pad(shape[0]),) + tuple(shape[1:])
        need = int(math.prod(shape)) * (2 if dtype == "f16" else 4)
        cur = self._scratch.get(role)
        if cur is None or cur.buf.nbytes < need:
            if cur is not None:
                self.gpu.wait()  # commands already encoded may still read the smaller buffer
            cur = self.gpu.empty((need // (2 if dtype == "f16" else 4),), dtype)
            self._scratch[role] = cur
        return Tensor(cur.buf, 0, tuple(shape), dtype)

    def _gate(self, N: int, M: int):
        key = (N, pad(M))
        if key not in self._ones:
            rt = self.gpu.empty((pad(M),), "u8")
            rt.numpy()[:] = 0
            self._ones[key] = (self.gpu.upload(np.ones((1, N), np.float32)), rt)
        return self._ones[key]

    def _conv(self, xh: Tensor, H: int, W: int, cw, out: Tensor, accumulate: bool = False):
        """3x3 (zero-padded, through im2col in stripes) or 1x1 convolution of a half NHWC image -> float out
        [H*W][N] (+= with accumulate). conv_in only: its 16 input channels are too few for conv3x3."""
        g = self.gpu
        wt, k, cin, cout = cw
        M = H * W
        ones, rt = self._gate(wt.N, M) if accumulate else (None, None)
        if k == 1:
            if accumulate:
                ops.gemm(g, xh, wt, out, M, epi=1, rtype=rt, gate=ones)
            else:
                ops.gemm(g, xh, wt, out, M)
            return
        step = 64 // math.gcd(W, 64)  # stripes of a whole number of 64-row tiles
        rows = max(step, min(H, STRIPE_BYTES // (W * wt.K * 2)) // step * step)
        col = self._buf("col", (rows * W, wt.K), "f16")
        for r0 in range(0, H, rows):
            n = min(rows, H - r0)
            ops.im2col(g, xh, col, N=1, H=H, W=W, C=cin, KH=3, KW=3, stride=1, pad_t=1, pad_l=1, K=wt.K,
                       zero_pad=True, row0=r0, rows=n)
            o = out.row(r0 * W, n * W)
            if accumulate:
                ops.gemm(g, col, wt, o, n * W, epi=1, rtype=rt, gate=ones)
            else:
                ops.gemm(g, col, wt, o, n * W)

    def _pbuf(self, role: str, H: int, W: int, C: int) -> Tensor:
        """A padded half image [(H + 2)(W + 2) + slack][C] (conv3x3 reads up to a tile past the last row)."""
        n = (H + 2) * (W + 2) + 128
        need = n * C * 2
        cur = self._scratch.get(role)
        if cur is None or cur.buf.nbytes < need:
            if cur is not None:
                self.gpu.wait()
            cur = self.gpu.empty((need // 2,), "f16")
            self._scratch[role] = cur
        return Tensor(cur.buf, 0, (n, C), "f16")

    def _gn_pad(self, x: Tensor, H: int, W: int, C: int, nb, role: str = "gnp") -> Tensor:
        """GroupNorm + SiLU of x [H*W][C] into a padded half image (a conv3x3 input)."""
        g = self.gpu
        xp = self._pbuf(role, H, W, C)
        ops.fill_border(g, xp, H=H, W=W, C=C)
        stats = g.empty((1, self.G, 2), "f32")
        stats.numpy()[...] = 0
        ops.group_norm(g, x, stats, nb[0], nb[1], xp, N=1, HW=H * W, C=C, G=self.G, eps=1e-6, pad_w=W)
        return xp

    def _conv3(self, xp: Tensor, H: int, W: int, cw, out: Tensor, accumulate: bool = False):
        wt, k, cin, cout = cw
        tile = self.conv_tile if wt.N % self.conv_tile[1] == 0 else (64, 128, 4)
        ops.conv3x3(self.gpu, xp, wt, out, H=H, W=W, C=cin, epi=1 if accumulate else 0, tile=tile)

    def _gn(self, x: Tensor, HW: int, C: int, nb, out: Tensor, silu: bool = True):
        g = self.gpu
        stats = g.empty((1, self.G, 2), "f32")
        stats.numpy()[...] = 0
        ops.group_norm(g, x, stats, nb[0], nb[1], out, N=1, HW=HW, C=C, G=self.G, eps=1e-6, silu=silu)
        return out

    def _resnet(self, r, x: Tensor, role: str, H: int, W: int, C: int):
        """-> (output float [H*W][cout], cout, its buffer role)."""
        g = self.gpu
        HW = H * W
        cout = r["c1"][3]
        hn = self._gn_pad(x, H, W, C, r["n1"])
        h1 = self._buf("h1", (HW, cout), "f32")
        self._conv3(hn, H, W, r["c1"], h1)
        h1n = self._gn_pad(h1, H, W, cout, r["n2"])
        if "sc" in r:
            xs = self._buf("half", (HW, C), "f16")
            ops.to_half(g, x, xs, HW * C)
            role = other(role)
            out = self._buf(role, (HW, cout), "f32")
            self._conv(xs, H, W, r["sc"], out)
        else:
            out = x
        self._conv3(h1n, H, W, r["c2"], out, accumulate=True)
        return out, cout, role

    def _attention(self, x: Tensor, H: int, W: int, C: int):
        """x += to_out(softmax(q k^T / sqrt(C)) v), q, k, v from GroupNorm(x); one head of C channels."""
        g = self.gpu
        L = H * W
        Lp = pad(L)
        hn = self._gn(x, L, C, self.attn_norm, self._buf("gn", (Lp, C), "f16"), silu=False)
        q = self._buf("q", (Lp, C), "f16")
        ops.gemm(g, hn, self.wq, q, Lp)
        kk = self._buf("k", (Lp, C), "f16")
        ops.gemm(g, hn, self.wk, kk, Lp)
        vt = self._buf("vt", (C, Lp), "f16")  # V^T = W_v hn^T, without the value bias (folded into to_out)
        ops.gemm(g, self.wv_act, QWeight("f16", hn, None, Lp, C), vt, C)
        kw = QWeight("f16", kk, None, Lp, C)
        vw = QWeight("f16", vt, None, C, Lp)
        o = self._buf("attn_o", (Lp, C), "f16")
        chunk = max(128, (64 << 20) // (Lp * 4) // 128 * 128)
        s = self._buf("scores", (chunk, Lp), "f32")
        p = self._buf("probs", (chunk, Lp), "f16")
        for r0 in range(0, L, chunk):
            n = min(chunk, Lp - r0)
            ops.gemm(g, q.row(r0, n), kw, s, n)
            ops.softmax_rows(g, s, p, n, Lp, valid=L, scale=C ** -0.5)  # padded keys get probability 0
            ops.gemm(g, p, vw, o.row(r0, n), n)
        ones, rt = self._gate(C, L)
        ops.gemm(g, o, self.wo, x, L, epi=1, rtype=rt, gate=ones)

    # -- decode ------------------------------------------------------------------------------------------------
    def decode(self, z: np.ndarray) -> np.ndarray:
        """z: the transformer's latents [16][h][w] (before the VAE's scaling) -> image [8h][8w][3] in [-1, 1]."""
        g, c = self.gpu, self.cfg
        lat = z.astype(np.float32) / np.float32(c["scaling_factor"]) + np.float32(c["shift_factor"])
        Cl, H, W = lat.shape
        g.wait()  # the scratch buffers below are written from the host
        xin = self._buf("lat", (H * W, Cl), "f16")
        xin.numpy()[:H * W] = lat.transpose(1, 2, 0).reshape(H * W, Cl).astype(np.float16)
        role = "xa"
        C = self.conv_in[3]
        x = self._buf(role, (H * W, C), "f32")
        self._conv(xin, H, W, self.conv_in, x)
        x, C, role = self._resnet(self.mid[0], x, role, H, W, C)
        self._attention(x, H, W, C)
        x, C, role = self._resnet(self.mid[1], x, role, H, W, C)
        for res, up in self.ups:
            for r in res:
                x, C, role = self._resnet(r, x, role, H, W, C)
            if up is not None:
                H, W = 2 * H, 2 * W
                xp = self._pbuf("upp", H, W, C)
                ops.pad_half(g, x, xp, H=H, W=W, C=C, up=2)
                role = other(role)
                x = self._buf(role, (H * W, C), "f32")
                self._conv3(xp, H, W, up, x)
            g.commit()
        hn = self._gn_pad(x, H, W, C, self.norm_out)
        img = self._buf("img", (H * W, self.conv_out[0].N), "f32")
        self._conv3(hn, H, W, self.conv_out, img)
        g.wait()
        return img.numpy()[:H * W, :3].reshape(H, W, 3).copy()


def pad(m: int) -> int:
    return -(-m // 128) * 128


def other(role: str) -> str:
    return "xb" if role == "xa" else "xa"
