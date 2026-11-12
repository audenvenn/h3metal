"""Weight rotation and quantization for the formats of kernels/gemm.metal.

Rotation: the input dimension of a weight is multiplied by a block-diagonal
orthonormal matrix R = diag(signs) * blockdiag(H_256 / 16) (H the Sylvester
Hadamard matrix). The engine rotates the activations by the same R before
quantizing them to int8 (kernels/ops.metal), so x W^T = (x R)(W R)^T.
"""

from __future__ import annotations

import numpy as np

RB = 256


def hadamard(n: int = RB) -> np.ndarray:
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float32)


_H = hadamard()


def random_signs(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)


def rotate_in(w: np.ndarray, signs: np.ndarray | None = None) -> np.ndarray:
    """W [N][K] -> W diag(signs) blockdiag(H): rotates the input dimension."""
    N, K = w.shape
    if K % RB:
        raise ValueError(f"input dimension {K} is not a multiple of {RB}")
    x = w.astype(np.float32, copy=signs is None)
    if signs is not None:
        x = x * signs[None, :]
    out = np.empty_like(x)
    step = max(1, (1 << 24) // K)  # rows per batch, to bound temporaries
    for r in range(0, N, step):
        blk = x[r:r + step].reshape(-1, K // RB, RB)
        out[r:r + step] = (blk @ _H).reshape(-1, K)  # H symmetric
    return out


def pack_int4(q: np.ndarray) -> np.ndarray:
    """Integer codes in -8..7, [N][K] -> [N][K/2] bytes, even k in the low nibble."""
    u = (q.astype(np.int16) & 15).astype(np.uint8)
    return np.ascontiguousarray(u[:, 0::2] | (u[:, 1::2] << 4))


def unpack_int4(b: np.ndarray) -> np.ndarray:
    lo = (b & 15).astype(np.int8)
    hi = (b >> 4).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo)
    hi = np.where(hi > 7, hi - 16, hi)
    q = np.empty((b.shape[0], b.shape[1] * 2), dtype=np.int8)
    q[:, 0::2], q[:, 1::2] = lo, hi
    return q


def _search_scale(w: np.ndarray, amax: np.ndarray, qmax: int, qmin: int, grid=(1.0, .95, .9, .85, .8, .75, .7)):
    """Per row (last axis reduced): the clipped scale that minimizes the squared error of round-to-nearest."""
    best_err = np.full(amax.shape, np.inf, dtype=np.float64)
    best = np.zeros_like(amax)
    for c in grid:
        s = np.maximum(amax * c / qmax, 1e-12)
        q = np.clip(np.rint(w / s[..., None]), qmin, qmax)
        err = ((q * s[..., None] - w) ** 2).sum(-1, dtype=np.float64)
        upd = err < best_err
        best_err = np.where(upd, err, best_err)
        best = np.where(upd, s, best)
    return best


def quant_q8(w: np.ndarray, search: bool = True):
    """Per-row symmetric int8: codes [N][K] int8, scales [N]."""
    amax = np.abs(w).max(1)
    s = _search_scale(w, amax, 127, -127, grid=(1.0, .97, .94, .9, .85)) if search else np.maximum(amax / 127, 1e-12)
    q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
    return q, s.astype(np.float32)


def quant_q4(w: np.ndarray, search: bool = True):
    """Per-row symmetric int4 (levels -8..7 scaled so 7 maps to the clip): packed [N][K/2], scales [N]."""
    amax = np.abs(w).max(1)
    s = _search_scale(w, amax, 7, -8) if search else np.maximum(amax / 7, 1e-12)
    q = np.clip(np.rint(w / s[:, None]), -8, 7)
    return pack_int4(q), s.astype(np.float32)


def quant_q4g(w: np.ndarray, G: int = 128, search: bool = True):
    """Per-(row, group of G inputs) int4: packed [N][K/2], scales [K/G][N] (group-major)."""
    N, K = w.shape
    wg = w.reshape(N, K // G, G)
    amax = np.abs(wg).max(-1)
    s = _search_scale(wg, amax, 7, -8) if search else np.maximum(amax / 7, 1e-12)
    q = np.clip(np.rint(wg / s[..., None]), -8, 7).reshape(N, K)
    return pack_int4(q), np.ascontiguousarray(s.T.astype(np.float32))


def dequant(fmt: str, w: np.ndarray, sw: np.ndarray | None, N: int, K: int, G: int = 128) -> np.ndarray:
    """The float matrix [N][K] a stored linear layer represents (in its rotated input space)."""
    if fmt in ("q8", "h8"):
        return w.astype(np.float32) * sw[:, None]
    if fmt in ("q4", "h4"):
        return unpack_int4(w).astype(np.float32) * sw[:, None]
    if fmt in ("q4g", "h4g"):
        return unpack_int4(w).astype(np.float32) * np.repeat(sw.reshape(K // G, N).T, G, axis=1)
    if fmt == "f16":
        return w.astype(np.float32)
    if fmt == "bf16":
        return (w.astype(np.uint32) << 16).view(np.float32)
    raise ValueError(fmt)


def quantize(fmt: str, w: np.ndarray, G: int = 128):
    """-> (stored codes, scales or None)."""
    if fmt in ("q8", "h8"):
        return quant_q8(w)
    if fmt in ("q4", "h4"):
        return quant_q4(w)
    if fmt in ("q4g", "h4g"):
        return quant_q4g(w, G)
    if fmt == "f16":
        return w.astype(np.float16), None
    if fmt == "bf16":
        u = np.ascontiguousarray(w, dtype=np.float32).view(np.uint32)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16), None
    raise ValueError(fmt)
