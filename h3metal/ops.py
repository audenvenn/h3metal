"""Kernel launchers: one Python function per GPU operation.

Every function records its dispatches into the open command buffer of `gpu`;
results are available after `gpu.wait()`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .gpu import Gpu, Tensor, pack

FMT_ID = {"q8": 1, "q4": 2, "q4g": 3, "f16": 4, "bf16": 5, "h8": 6, "h4": 7, "h4g": 8}
OUT_DT = {"f16": "half", "bf16": "bfloat", "f32": "float"}


@dataclass
class QWeight:
    """A linear layer's weight on the GPU: W[N][K] in one of the formats of kernels/gemm.metal."""
    fmt: str           # q8 | q4 | q4g | f16 | bf16 | h8 | h4 (h*: int weights, half activations)
    w: Tensor          # codes, [N][K] row-major (int4: two per byte)
    sw: Tensor | None  # float scales: [N] (q8, q4) or [K/G][N] (q4g)
    N: int
    K: int
    G: int = 128
    bias: Tensor | None = None
    ws: int | None = None  # row stride of the codes in elements, when it differs from K (a slice of K)

    @property
    def w_stride(self) -> int:
        return self.ws or self.K


# Whether the accelerator's register layout is the one the fast group-scale
# epilogue of kernels/gemm.metal assumes (checked by tests/test_gemm.py).
FAST_LAYOUT_OK = True


def pad_rows(m: int, tile: int = 128) -> int:
    return (m + tile - 1) // tile * tile


def gemm(gpu: Gpu, a: Tensor, wt: QWeight, out: Tensor, M: int, *, sa: Tensor | None = None, epi: int = 0,
         rtype: Tensor | None = None, gate: Tensor | None = None, g_stride: int = 0, c_stride: int | None = None,
         a_stride: int | None = None, tile=(64, 128, 4), swz: int = 1, ksplit: int = 1):
    """out = A x W^T (epi 0), or out += gate[rtype] * (A x W^T) (epi 1, out float).

    ksplit: accumulate the product in that many dispatches over consecutive slices of K (epi 1 and per-row
    scales only). Long products run faster this way: each dispatch's working set fits the caches better.
    """
    if ksplit > 1:
        if epi != 1 or wt.fmt not in ("q8", "q4", "h8", "h4") or wt.K % (ksplit * 256):
            raise ValueError(f"cannot split K of a {wt.fmt} product (epi {epi}, K {wt.K}) {ksplit} ways")
        kp = wt.K // ksplit
        per_byte = 2 if wt.fmt in ("q4", "h4") else 1
        for i in range(ksplit):
            part = replace(wt, w=wt.w.view((wt.N, kp // per_byte), offset_elems=i * kp // per_byte), K=kp,
                           ws=wt.w_stride, bias=wt.bias if i == 0 else None)
            gemm(gpu, a.view((a.shape[0], kp), offset_elems=i * kp), part, out, M, sa=sa, epi=1, rtype=rtype,
                 gate=gate, g_stride=g_stride, c_stride=c_stride, a_stride=a_stride or wt.K, tile=tile, swz=swz)
        return
    tm, tn, nsg = tile
    out_t = OUT_DT[out.dtype] if epi == 0 else "float"
    k = gpu.kernel("gemm.metal", "gemm", WFMT=FMT_ID[wt.fmt], EPI=epi, OUT_T=out_t, TM=tm, TN=tn, NSG=nsg,
                   G=wt.G, HAS_BIAS=int(wt.bias is not None),
                   FAST_LAYOUT=int(wt.fmt in ("q4g", "h4g") and tile == (64, 128, 4) and FAST_LAYOUT_OK), SWZ=swz)
    m_pad = (M + tm - 1) // tm * tm
    params = pack("8i", m_pad, wt.N, wt.K, a_stride or wt.K, c_stride or wt.N, wt.w_stride, wt.K // wt.G,
                  g_stride or wt.N)
    gpu.launch(k, (wt.N // tn, m_pad // tm, 1), (32 * nsg, 1, 1),
               [a, wt.w, wt.sw, sa, out, wt.bias, rtype, gate], params)


def attention(gpu: Gpu, q: Tensor, k: Tensor, v: Tensor, out: Tensor, *, Lq: int, L: int, H: int, KVH: int,
              D: int, q_rows: int, k_rows: int, scale: float, causal: bool = False, causal_off: int = 0,
              wm: int = 8, bk: int = 32, seq_heads: int | None = None, seq_rows: int = 0,
              q_scale: Tensor | None = None, k_scale: Tensor | None = None, **variant):
    """out = softmax(q k^T * scale) v per head (kernels/attention.metal); k_rows a multiple of bk."""
    import math
    out_t = OUT_DT[out.dtype]
    variant.pop("TG", None)
    kern = gpu.kernel("attention.metal", "attention", HD=D, WM=wm, BK=bk, CAUSAL=int(causal), O_T=out_t,
                      QK8=int(q_scale is not None), **variant)
    sh = seq_heads or H
    params = pack("7if3i", Lq, L, H, KVH, q_rows, k_rows, sh * D, scale * 1.4426950408889634, causal_off, sh,
                  seq_rows)
    gpu.launch(kern, (math.ceil(Lq / (16 * wm)), H, 1), (32 * wm, 1, 1), [q, k, v, out, None, q_scale, k_scale],
               params, params_index=4)

def _rows_kernel_cfg(D: int):
    """SIMD-groups and rotation blocks per SIMD-group for a row of D channels."""
    nb = D // 256
    nsg = 8 if nb >= 8 else nb
    return nsg, -(-nb // nsg)


def norm_mod_q8(gpu: Gpu, x: Tensor, mod: Tensor, rtype: Tensor, out: Tensor, sa: Tensor | None, M: int, D: int,
                *, eps: float, mod_stride: int | None = None, x_stride: int | None = None, out_f: bool = False,
                out_h: bool = False, out_hr: bool = False):
    """out = int8(rot(rmsnorm(x) * mod[type][0] + mod[type][1])) and its row scales (or the plain modulated
    row as float with out_f, as half with out_h)."""
    nsg, maxb = _rows_kernel_cfg(D)
    k = gpu.kernel("ops.metal", "norm_mod_q8", NSG=nsg, MAXB=maxb, OUT_F=int(out_f), OUT_H=int(out_h),
                   OUT_HR=int(out_hr))
    params = pack("4if", M, D, x_stride or D, mod_stride or 2 * D, eps)
    gpu.launch(k, (M, 1, 1), (32 * nsg, 1, 1), [x, mod, rtype, out, sa], params)


def rot_q8(gpu: Gpu, x: Tensor, out: Tensor, sa: Tensor | None, M: int, D: int, *, swiglu: bool = False,
           x_stride: int | None = None, out_hr: bool = False):
    """out = int8(rot(x)) (or of value * silu(gate) with the gates in columns D..2D-1) and its row scales;
    with out_hr the rotated rows in half precision instead."""
    nsg, maxb = _rows_kernel_cfg(D)
    in_t = OUT_DT[x.dtype]
    k = gpu.kernel("ops.metal", "rot_q8", NSG=nsg, MAXB=maxb, IN_T=in_t, OUT_HR=int(out_hr))
    params = pack("4i", M, D, x_stride or (2 * D if swiglu else D), 1 if swiglu else 0)
    gpu.launch(k, (M, 1, 1), (32 * nsg, 1, 1), [x, out, sa], params)


def qk_norm_rope(gpu: Gpu, qkv: Tensor, wq: Tensor | None, wk: Tensor | None, cos: Tensor | None,
                 sin: Tensor | None, Q: Tensor, K: Tensor, V: Tensor, *, M: int, Hq: int, Hk: int, Hv: int,
                 hd: int, out_rows: int, rot: int, eps: float, norm: bool = True, in_stride: int | None = None,
                 seq_len: int | None = None, q_scale: Tensor | None = None, k_scale: Tensor | None = None):
    in_t = OUT_DT[qkv.dtype]
    nsg = 8
    k = gpu.kernel("ops.metal", "qk_norm_rope", NSG=nsg, HDIM=hd, IN_T=in_t, QK8=int(q_scale is not None))
    slots = Hq + Hk + Hv
    params = pack("8ifi", M, Hq, Hk, Hv, in_stride or slots * hd, out_rows, rot, int(norm), eps, seq_len or M)
    gpu.launch(k, (M, -(-slots // nsg), 1), (32 * nsg, 1, 1), [qkv, wq, wk, cos, sin, Q, K, V, None, q_scale, k_scale],
               params, params_index=8)


def linear_f32(gpu: Gpu, x: Tensor, w: Tensor, b: Tensor | None, out: Tensor, *, M: int, N: int, K: int,
               in_stride: int | None = None, out_stride: int | None = None):
    nsg = 8
    k = gpu.kernel("ops.metal", "linear_f32", NSG=nsg)
    params = pack("6i", M, N, K, in_stride or K, out_stride or N, int(b is not None))
    gpu.launch(k, (-(-N // 32), -(-M // nsg), 1), (32 * nsg, 1, 1), [x, w, b, out], params)


def copy_f32(gpu: Gpu, src: Tensor, dst: Tensor, n: int):
    """Copies n floats (a multiple of 4)."""
    k = gpu.kernel("ops.metal", "copy4")
    n4 = n // 4
    gpu.launch(k, (-(-n4 // 256), 1, 1), (256, 1, 1), [src, dst], pack("i", n4))


def layer_norm(gpu: Gpu, x: Tensor, w: Tensor, b: Tensor, out: Tensor, M: int, D: int, eps: float):
    k = gpu.kernel("ops.metal", "layer_norm", NSG=8)
    gpu.launch(k, (M, 1, 1), (256, 1, 1), [x, w, b, out], pack("3if", M, D, D, eps))


def swiglu(gpu: Gpu, x: Tensor, out: Tensor, M: int, F: int):
    k = gpu.kernel("ops.metal", "swiglu")
    gpu.launch(k, (-(-F // 256), M, 1), (256, 1, 1), [x, out], pack("2i", M, F))


def im2col(gpu: Gpu, x: Tensor, col: Tensor, *, N: int, H: int, W: int, C: int, KH: int, KW: int, stride: int,
           pad_t: int, pad_l: int, K: int):
    """col[N*Ho*Wo][K] <- the (ky, kx, c) taps of each output pixel of an NHWC half image, reflect-padded."""
    Ho = (H + 2 * pad_t - KH) // stride + 1 if stride == 1 else (H + 1 - KH) // stride + 1
    Wo = (W + 2 * pad_l - KW) // stride + 1 if stride == 1 else (W + 1 - KW) // stride + 1
    k = gpu.kernel("ops.metal", "im2col")
    total = -(-C // 4) * KH * KW * N * Ho * Wo
    gpu.launch(k, (-(-total // 256), 1, 1), (256, 1, 1), [x, col],
               pack("12i", N, H, W, C, Ho, Wo, KH, KW, stride, pad_t, pad_l, K))
    return Ho, Wo


def group_norm(gpu: Gpu, x: Tensor, stats: Tensor, w: Tensor, b: Tensor, y: Tensor, *, N: int, HW: int, C: int,
               G: int = 32, eps: float = 1e-6, silu: bool = True):
    """y = half(silu?(groupnorm(x))), x float NHWC; stats [N][G][2] float, zeroed by the caller."""
    ks = gpu.kernel("ops.metal", "gn_stats")
    for p in (0, 1):
        gpu.launch(ks, (G, N, -(-HW // 64)), (1, 1, 1), [x, stats], pack("4ifii", N, HW, C, G, eps, int(silu), p))
    ka = gpu.kernel("ops.metal", "gn_apply")
    gpu.launch(ka, (-(-C // 256), N * HW, 1), (min(256, C), 1, 1), [x, stats, w, b, y],
               pack("4ifii", N, HW, C, G, eps, int(silu), 0))


def to_half(gpu: Gpu, x: Tensor, y: Tensor, n: int):
    k = gpu.kernel("ops.metal", "to_half")
    gpu.launch(k, (-(-n // 256), 1, 1), (256, 1, 1), [x, y], pack("i", n))
