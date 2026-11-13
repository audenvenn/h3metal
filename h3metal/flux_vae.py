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


def convert_flux_vae(src: str, out: str, log=print, encoder: bool = False):
    """src: a directory with vae/config.json and vae/diffusion_pytorch_model.safetensors; the decoder, and with
    encoder the encoder too (reference images of FLUX.2 edits)."""
    d = os.path.join(src, "vae")
    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)
    st = SafeTensors(os.path.join(d, "diffusion_pytorch_model.safetensors"))
    keep = {k: cfg[k] for k in ("block_out_channels", "layers_per_block", "latent_channels", "norm_num_groups")}
    keep["scaling_factor"] = cfg.get("scaling_factor") or 1.0  # FLUX.2's VAE normalizes with batch-norm statistics
    keep["shift_factor"] = cfg.get("shift_factor") or 0.0
    if "bn.running_mean" in st.keys():  # FLUX.2: latents are normalized per channel of their 2 x 2 patches
        keep["patch"] = list(cfg.get("patch_size", [2, 2]))
        keep["bn_eps"] = cfg.get("batch_norm_eps", 1e-4)
    w = Writer(out, {"kind": "flux-vae", "config": keep})
    for k in sorted(st.keys()):
        if k.startswith("decoder.") or (encoder and k.startswith(("encoder.", "quant_conv."))):
            w.tensor(k, np.ascontiguousarray(st.get(k), np.float32), "f32")
        elif k in ("bn.running_mean", "bn.running_var") or (k.startswith("post_quant_conv.") and
                                                            cfg.get("use_post_quant_conv", False)):
            w.tensor(k, np.ascontiguousarray(st.get(k), np.float32), "f32", host_only=True)
    w.close()
    log(f"wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB)")


class FluxVAEDecoder:
    def __init__(self, gpu: Gpu, wf: WeightFile):
        self.gpu = gpu
        c = wf.meta["config"]
        self.cfg = c
        h = wf.host
        self.G = c["norm_num_groups"]
        self.bn = (h("bn.running_mean").astype(np.float32), h("bn.running_var").astype(np.float32)) \
            if "bn.running_mean" in wf else None
        self.pq = (h("post_quant_conv.weight")[:, :, 0, 0].astype(np.float32),
                   h("post_quant_conv.bias").astype(np.float32)) if "post_quant_conv.weight" in wf else None

        self.wf = wf
        self.conv_in = self._conv_w("decoder.conv_in")
        self.mid = [self._resnet_w("decoder.mid_block.resnets.0"), self._resnet_w("decoder.mid_block.resnets.1")]
        self._attention_w("decoder.mid_block.attentions.0")
        n_up = len(c["block_out_channels"])
        self.ups = []
        for i in range(n_up):
            p = f"decoder.up_blocks.{i}"
            res = [self._resnet_w(f"{p}.resnets.{j}") for j in range(c["layers_per_block"] + 1)]
            up = self._conv_w(f"{p}.upsamplers.0.conv") if f"{p}.upsamplers.0.conv.weight" in wf else None
            self.ups.append((res, up))
        self.norm_out = self._norm_w("decoder.conv_norm_out")
        self.conv_out = self._conv_w("decoder.conv_out")
        self._scratch = {}
        self._ones = {}
        self.conv_tile = CONV_TILE

    # -- weights -----------------------------------------------------------------------------------------------
    def _conv_w(self, name: str, fold=None):
        """A convolution as a half [N][kh kw cin] product with bias (N padded to 128, K to 32); fold: a following
        1 x 1 convolution (W, b) folded in, W' = W_f W, b' = W_f b + b_f."""
        h = self.wf.host
        w = h(name + ".weight").astype(np.float32)
        b = h(name + ".bias").astype(np.float32)
        cout, cin, kh, kw = w.shape
        m = w.transpose(0, 2, 3, 1).reshape(cout, kh * kw * cin)
        if fold is not None:
            m, b = fold[0] @ m, fold[0] @ b + fold[1]
            cout = m.shape[0]
        K = -(-m.shape[1] // 32) * 32
        N = -(-cout // 128) * 128
        full = np.zeros((N, K), np.float16)
        full[:cout, :m.shape[1]] = m
        bias = np.zeros(N, np.float32)
        bias[:cout] = b
        return QWeight("f16", self.gpu.upload(full), None, N, K, bias=self.gpu.upload(bias)), kh, cin, cout

    def _lin_w(self, name: str, bias=None):
        h = self.wf.host
        w = h(name + ".weight").astype(np.float32)
        if w.ndim == 4:  # a 1x1 convolution stored as such
            w = w[:, :, 0, 0]
        b = h(name + ".bias").astype(np.float32) if bias is None else bias
        return QWeight("f16", self.gpu.upload(w.astype(np.float16)), None, w.shape[0], w.shape[1],
                       bias=self.gpu.upload(b.astype(np.float32)))

    def _norm_w(self, name: str):
        h = self.wf.host
        return (self.gpu.upload(h(name + ".weight").astype(np.float32)),
                self.gpu.upload(h(name + ".bias").astype(np.float32)))

    def _resnet_w(self, p: str):
        r = {"n1": self._norm_w(p + ".norm1"), "c1": self._conv_w(p + ".conv1"), "n2": self._norm_w(p + ".norm2"),
             "c2": self._conv_w(p + ".conv2")}
        if p + ".conv_shortcut.weight" in self.wf:
            r["sc"] = self._conv_w(p + ".conv_shortcut")
        return r

    def _attention_w(self, a: str):
        h = self.wf.host
        self.attn_norm = self._norm_w(a + ".group_norm")
        self.wq, self.wk = self._lin_w(a + ".to_q"), self._lin_w(a + ".to_k")
        wv = h(a + ".to_v.weight").astype(np.float32)
        bv = h(a + ".to_v.bias").astype(np.float32)
        wo = h(a + ".to_out.0.weight").astype(np.float32)
        # Softmax rows sum to one, so the value bias passes through the attention: fold it into to_out's bias.
        self.wo = self._lin_w(a + ".to_out.0", bias=wo @ bv + h(a + ".to_out.0.bias").astype(np.float32))
        self.wv_act = self.gpu.upload(wv.astype(np.float16))  # V^T = W_v x^T: W_v as the left operand

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
        """-> (output [H*W][cout] in x's precision, cout, its buffer role)."""
        g = self.gpu
        HW = H * W
        cout = r["c1"][3]
        dt = x.dtype
        hn = self._gn_pad(x, H, W, C, r["n1"])
        h1 = self._buf("h1", (HW, cout), dt)
        self._conv3(hn, H, W, r["c1"], h1)
        h1n = self._gn_pad(h1, H, W, cout, r["n2"])
        if "sc" in r:
            if dt == "f16":
                xs = x
            else:
                xs = self._buf("half", (HW, C), "f16")
                ops.to_half(g, x, xs, HW * C)
            role = other(role)
            out = self._buf(role, (HW, cout), dt)
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

    # -- decode ----------------------------------------------------------------------------------------------
    def decode(self, z: np.ndarray) -> np.ndarray:
        """z: the transformer's latents [16][h][w] (before the VAE's scaling), or FLUX.2's [128][h/2][w/2] (2 x 2
        patches, normalized with the VAE's batch-norm statistics) -> image [8h][8w][3] in [-1, 1].

        conv_in and the middle block run at the latents' size and keep float activations. The up blocks, up to
        64 times as many pixels, keep theirs in half precision (sums still in float): at 2048 x 2048 a float
        activation of 256 channels is 4.3 GB. Half precision rounds to about 1/2000, far below the image's
        8 bits."""
        g, c = self.gpu, self.cfg
        lat = z.astype(np.float32) / np.float32(c["scaling_factor"]) + np.float32(c["shift_factor"])
        if self.bn is not None:  # FLUX.2: z is [C*4][h/2][w/2], normalized per channel of the 2 x 2 patches
            mean, var = self.bn
            lat = lat * np.sqrt(var + np.float32(c["bn_eps"]))[:, None, None] + mean[:, None, None]
            C4, h2, w2 = lat.shape
            lat = lat.reshape(C4 // 4, 2, 2, h2, w2).transpose(0, 3, 1, 4, 2).reshape(C4 // 4, 2 * h2, 2 * w2)
        if self.pq is not None:  # the 1 x 1 convolution before the decoder (CPU: a few MFLOP)
            pw, pb = self.pq
            lat = (np.einsum("oc,chw->ohw", pw, lat) + pb[:, None, None]).astype(np.float32)
        Cl, H, W = lat.shape
        g.wait()  # the scratch buffers below are written from the host
        xin = self._buf("lat", (H * W, Cl), "f16")
        xin.numpy()[:H * W] = lat.transpose(1, 2, 0).reshape(H * W, Cl).astype(np.float16)
        C = self.conv_in[3]
        x = self._buf("mid_a", (H * W, C), "f32")
        self._conv(xin, H, W, self.conv_in, x)
        x, C, _ = self._resnet(self.mid[0], x, "mid_a", H, W, C)
        self._attention(x, H, W, C)
        x, C, _ = self._resnet(self.mid[1], x, "mid_a", H, W, C)
        role = "xa"
        xh = self._buf(role, (H * W, C), "f16")
        ops.to_half(g, x, xh, H * W * C)
        x = xh
        for res, up in self.ups:
            for r in res:
                x, C, role = self._resnet(r, x, role, H, W, C)
            if up is not None:
                H, W = 2 * H, 2 * W
                xp = self._pbuf("gnp", H, W, C)  # the norms' bordered buffer, free between them
                ops.pad_half(g, x, xp, H=H, W=W, C=C, up=2)
                role = other(role)
                x = self._buf(role, (H * W, C), "f16")
                self._conv3(xp, H, W, up, x)
            g.commit()
        hn = self._gn_pad(x, H, W, C, self.norm_out)
        img = self._buf("img", (H * W, 3), "f32")
        wt = self.conv_out[0]
        ops.conv3x3(g, hn, wt, img, H=H, W=W, C=C, tile=CONV_TILE, out_channels=3)
        g.wait()
        return img.numpy()[:H * W].reshape(H, W, 3).copy()

    def release(self):
        """Frees the scratch buffers (gigabytes at 2048 x 2048)."""
        self._scratch = {}
        self._ones = {}


def pad(m: int) -> int:
    return -(-m // 128) * 128


def other(role: str) -> str:
    return "xb" if role == "xa" else "xa"


class FluxVAEEncoder(FluxVAEDecoder):
    """The FLUX VAE's encoder (FLUX.2's reference images), on the decoder's convolutions. The first levels, at up to
    the image's full size, keep half-precision activations; the last level and the middle block, at 1/8 of each
    side, keep float ones (the attention adds into its input in float). The 1 x 1 quant_conv is folded into
    conv_out; the latent distribution's mean is the encoding (the pipeline's argmax)."""

    def __init__(self, gpu: Gpu, wf: WeightFile):
        self.gpu, self.wf = gpu, wf
        c = wf.meta["config"]
        self.cfg = c
        h = wf.host
        self.G = c["norm_num_groups"]
        self.bn = (h("bn.running_mean").astype(np.float32), h("bn.running_var").astype(np.float32)) \
            if "bn.running_mean" in wf else None
        if "encoder.conv_in.weight" not in wf:
            raise ValueError(f"{wf.path} has no encoder: convert the model again")
        self.conv_in = self._conv_w("encoder.conv_in")
        self.downs = []
        for i in range(len(c["block_out_channels"])):
            p = f"encoder.down_blocks.{i}"
            res = [self._resnet_w(f"{p}.resnets.{j}") for j in range(c["layers_per_block"])]
            down = self._conv_w(f"{p}.downsamplers.0.conv") if f"{p}.downsamplers.0.conv.weight" in wf else None
            self.downs.append((res, down))
        self.mid = [self._resnet_w("encoder.mid_block.resnets.0"), self._resnet_w("encoder.mid_block.resnets.1")]
        self._attention_w("encoder.mid_block.attentions.0")
        self.norm_out = self._norm_w("encoder.conv_norm_out")
        q = h("quant_conv.weight")[:, :, 0, 0].astype(np.float32)
        self.conv_out = self._conv_w("encoder.conv_out", fold=(q, h("quant_conv.bias").astype(np.float32)))
        self._scratch = {}
        self._ones = {}
        self.conv_tile = CONV_TILE

    def encode(self, img: np.ndarray) -> np.ndarray:
        """img: float [H][W][3] in [-1, 1], sides multiples of 16 -> the latent mean [C][H/8][W/8] (float)."""
        g = self.gpu
        H, W, _ = img.shape
        xh = self._buf("img", (H * W, 3), "f16")
        xh.numpy()[:H * W] = img.reshape(H * W, 3).astype(np.float16)
        C = self.conv_in[3]
        role = "xa"
        x = self._buf(role, (H * W, C), "f16")
        self._conv(xh, H, W, self.conv_in, x)
        n = len(self.downs)
        for i, (res, down) in enumerate(self.downs):
            for r in res:
                x, C, role = self._resnet(r, x, role, H, W, C)
            if down is not None:
                xp = self._pbuf("downp", H, W, C)
                ops.pad_half(g, x, xp, H=H, W=W, C=C)  # zeros on the bottom and right, as the reference pads
                Ho, Wo = (H + 1 - 3) // 2 + 1, (W + 1 - 3) // 2 + 1
                role = other(role)
                y = self._buf(role, (Ho * Wo, down[0].N), "f32" if i + 1 == n - 1 else "f16")
                ops.conv3x3(g, xp, down[0], y, H=Ho, W=Wo, C=C, stride=2, in_hw=(H, W), pad=0,
                            tile=self.conv_tile if down[0].N % self.conv_tile[1] == 0 else (64, 128, 4))
                x, H, W = y, Ho, Wo
            g.commit()
        x, C, role = self._resnet(self.mid[0], x, role, H, W, C)
        self._attention(x, H, W, C)
        x, C, role = self._resnet(self.mid[1], x, role, H, W, C)
        hn = self._gn_pad(x, H, W, C, self.norm_out)
        mom = self._buf("mom", (H * W, self.conv_out[0].N), "f32")
        self._conv3(hn, H, W, self.conv_out, mom)
        g.wait()
        lat = self.cfg["latent_channels"]
        return mom.numpy()[:H * W, :lat].reshape(H, W, lat).transpose(2, 0, 1).copy()

    def encode_packed(self, img: np.ndarray) -> np.ndarray:
        """FLUX.2: the latent mean in 2 x 2 patches, normalized with the batch-norm statistics -> [C*4][H/16][W/16]
        (the rows the transformer reads for a reference image)."""
        lat = self.encode(img)
        C, h, w = lat.shape
        packed = lat.reshape(C, h // 2, 2, w // 2, 2).transpose(0, 2, 4, 1, 3).reshape(C * 4, h // 2, w // 2)
        mean, var = self.bn
        return ((packed - mean[:, None, None]) / np.sqrt(var + np.float32(self.cfg["bn_eps"]))[:, None, None]
                ).astype(np.float32)
