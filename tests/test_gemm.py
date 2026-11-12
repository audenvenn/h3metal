"""GEMM kernels against NumPy, on random data."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from h3metal.gpu import Gpu  # noqa: E402
from h3metal.ops import QWeight, gemm  # noqa: E402


def pack_int4(q: np.ndarray) -> np.ndarray:
    """[N][K] int codes in -8..7 -> [N][K/2] bytes, even k in the low nibble."""
    u = (q.astype(np.int16) & 15).astype(np.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)


def check(fmt, M=200, N=256, K=512, epi=0, out_dtype="f16", seed=0, tile=(64, 128, 4), ksplit=1):
    rng = np.random.default_rng(seed)
    gpu = Gpu()
    m_pad = (M + 127) // 128 * 128
    G = 128
    a = rng.integers(-127, 128, size=(m_pad, K), dtype=np.int8)
    sa = rng.uniform(0.001, 0.01, size=m_pad).astype(np.float32)
    if fmt == "q8":
        q = rng.integers(-127, 128, size=(N, K), dtype=np.int8)
        sw = rng.uniform(0.001, 0.01, size=N).astype(np.float32)
        wt = QWeight("q8", gpu.upload(q), gpu.upload(sw), N, K)
        ref = (a.astype(np.float64) @ q.T.astype(np.float64)) * sa[:, None] * sw[None, :]
    elif fmt == "q4":
        q = rng.integers(-8, 8, size=(N, K))
        sw = rng.uniform(0.001, 0.01, size=N).astype(np.float32)
        wt = QWeight("q4", gpu.upload(pack_int4(q)), gpu.upload(sw), N, K)
        ref = (a.astype(np.float64) @ q.T.astype(np.float64)) * sa[:, None] * sw[None, :]
    elif fmt == "q4g":
        q = rng.integers(-8, 8, size=(N, K))
        sw = rng.uniform(0.001, 0.01, size=(N, K // G)).astype(np.float32)
        wt = QWeight("q4g", gpu.upload(pack_int4(q)), gpu.upload(np.ascontiguousarray(sw.T)), N, K, G)
        wf = q.astype(np.float64) * np.repeat(sw, G, axis=1)
        ref = (a.astype(np.float64) @ wf.T) * sa[:, None]
    elif fmt == "h4g":
        af = rng.standard_normal((m_pad, K)).astype(np.float16)
        q = rng.integers(-8, 8, size=(N, K))
        sw = rng.uniform(0.001, 0.01, size=(N, K // G)).astype(np.float32)
        wt = QWeight("h4g", gpu.upload(pack_int4(q)), gpu.upload(np.ascontiguousarray(sw.T)), N, K, G)
        a, sa = af, None
        ref = af.astype(np.float64) @ (q.astype(np.float64) * np.repeat(sw, G, axis=1)).T
    elif fmt in ("h8", "h4"):
        af = rng.standard_normal((m_pad, K)).astype(np.float16)
        if fmt == "h8":
            q = rng.integers(-127, 128, size=(N, K), dtype=np.int8); codes = q
        else:
            q = rng.integers(-8, 8, size=(N, K)); codes = pack_int4(q)
        sw = rng.uniform(0.001, 0.01, size=N).astype(np.float32)
        wt = QWeight(fmt, gpu.upload(codes), gpu.upload(sw), N, K)
        a, sa = af, None
        ref = (af.astype(np.float64) @ q.T.astype(np.float64)) * sw[None, :]
    elif fmt == "f16":
        af = rng.standard_normal((m_pad, K)).astype(np.float16)
        wf = (rng.standard_normal((N, K)) * 0.05).astype(np.float16)
        wt = QWeight("f16", gpu.upload(wf), None, N, K)
        a, sa = af, None
        ref = af.astype(np.float64) @ wf.T.astype(np.float64)
    A = gpu.upload(a)
    SA = gpu.upload(sa) if sa is not None else None
    if epi == 0:
        out = gpu.empty((m_pad, N), out_dtype)
        gemm(gpu, A, wt, out, M, sa=SA, tile=tile)
        gpu.wait()
        got = out.numpy()[:M].astype(np.float64) if out_dtype != "bf16" else \
            (out.numpy()[:M].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    else:
        x0 = rng.standard_normal((m_pad, N)).astype(np.float32)
        X = gpu.upload(x0)
        types = rng.integers(0, 3, size=m_pad).astype(np.uint8)
        gates = rng.standard_normal((3, N)).astype(np.float32)
        gemm(gpu, A, wt, X, M, sa=SA, epi=1, rtype=gpu.upload(types), gate=gpu.upload(gates), tile=tile,
             ksplit=ksplit)
        gpu.wait()
        got = X.numpy()[:M].astype(np.float64)
        ref = x0.astype(np.float64) + gates[types][:, :] * ref
    ref = ref[:M]
    err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)
    print(f"{fmt:4s} epi={epi} out={out_dtype}{f' ksplit={ksplit}' if ksplit > 1 else ''}: max rel err {err:.2e}")
    assert err < (2e-3 if out_dtype != "bf16" else 1e-2), err


if __name__ == "__main__":
    for fmt in ("q8", "q4", "q4g", "f16", "h8", "h4", "h4g"):
        check(fmt)
    check("q4g", epi=1)
    check("q8", out_dtype="bf16")
    check("q4g", out_dtype="f32", M=130, N=384, K=1024)
    check("q4g", out_dtype="f32", M=130, N=384, K=1024, tile=(128, 128, 8))
    for fmt in ("q8", "q4", "h8", "h4"):  # K in two dispatches, accumulated into the residual
        check(fmt, epi=1, K=1024, ksplit=2)
    check("q8", epi=1, K=2048, ksplit=4, tile=(64, 256, 8))
    print("ok")
