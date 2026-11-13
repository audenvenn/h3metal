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


def check_partial_tiles_refused():
    """The grid covers whole tiles, so a tile width that does not divide N would leave the last N % tn columns
    unwritten (a 32 x 512 tile on 5376 columns skipped 256 of them): gemm must refuse it."""
    gpu = Gpu()
    for N, tn in ((5376, 512), (3840, 512), (1152, 256), (1216, 128)):
        wt = QWeight("q8", gpu.upload(np.zeros((N, 256), np.int8)), gpu.upload(np.ones(N, np.float32)), N, 256)
        try:
            gemm(gpu, gpu.upload(np.zeros((64, 256), np.int8)), wt, gpu.empty((64, N), "f16"), 64,
                 sa=gpu.upload(np.ones(64, np.float32)), tile=(32, tn, 8))
        except ValueError:
            continue
        raise AssertionError(f"a {tn}-column tile on {N} columns was not refused")
    print("tiles that do not divide N: refused")


def check_dit_tiles(model: str = "minimax-h3"):
    """Every tile and dispatch split a transformer uses (h3metal/dit.py: _tile, h3metal/zimage.py: _ztile), at
    its real shapes and row counts (1024, 2048 and 4096 pixels square), against a float64 reference on a sample
    of rows."""
    if model == "minimax-h3":
        from h3metal.dit import _tile
        D, inner, F = 5376, 7168, 14336
        products = (("qkv", D, 3 * inner, 0), ("out", inner, D, 1), ("fc1", D, 2 * F, 0), ("fc2", F, D, 1))
        sizes = (1324, 4396, 16684)
    elif model == "z-image":  # every product stores its result (the norm after it needs the whole row)
        from h3metal.zimage import _ztile as _tile
        D, F = 3840, 10240
        products = (("qkv", D, 3 * D, 0), ("out", D, D, 0), ("fc1", D, 2 * F, 0), ("fc2", F, D, 0))
        sizes = (4160, 16448)  # 1024 and 2048 pixels square (64 caption rows)
    else:  # FLUX.2 [klein] 4B or 9B: the double blocks' image and caption rows run apart, the single blocks' together
        from h3metal.flux2 import _ftile as _tile
        D = {"flux2-4b": 3072, "flux2-9b": 4096}[model]
        F = 3 * D
        products = (("qkv", D, 3 * D, 0), ("out", D, D, 1), ("fc1", D, 2 * F, 0), ("fc2", F, D, 1),
                    ("qkv_mlp", D, 3 * D + 2 * F, 0))
        sizes = (512, 4096, 4608)  # caption rows, image rows at 1024 x 1024, both
    gpu = Gpu()
    rng = np.random.default_rng(1)
    for fmt in (("q8",) if model.startswith("flux2") else ("q8", "q4")):
        for name, K, N, epi in products:
            for M in sizes:
                cfg = _tile(QWeight(fmt, None, None, N, K), M)
                m_pad = (M + 127) // 128 * 128
                a = rng.integers(-127, 128, size=(m_pad, K), dtype=np.int8)
                sa = rng.uniform(0.001, 0.01, size=m_pad).astype(np.float32)
                q = rng.integers(-127, 128, size=(N, K), dtype=np.int8) if fmt == "q8" else \
                    rng.integers(-8, 8, size=(N, K)).astype(np.int8)
                sw = rng.uniform(0.001, 0.01, size=N).astype(np.float32)
                wt = QWeight(fmt, gpu.upload(q if fmt == "q8" else pack_int4(q)), gpu.upload(sw), N, K)
                rows = np.unique(np.r_[0:40, M - 40:M, rng.integers(0, M, 64)])
                ref = (a[rows].astype(np.float64) @ q.T.astype(np.float64)) * sa[rows, None] * sw[None, :]
                if epi == 0:
                    out = gpu.empty((m_pad, N), "f16")
                    gemm(gpu, gpu.upload(a), wt, out, M, sa=gpu.upload(sa), **cfg)
                    gpu.wait()
                    got = out.numpy()[rows].astype(np.float64)
                else:
                    x0 = rng.standard_normal((m_pad, N)).astype(np.float32)
                    types = rng.integers(0, 3, size=m_pad).astype(np.uint8)
                    gates = rng.standard_normal((3, N)).astype(np.float32)
                    X = gpu.upload(x0)
                    gemm(gpu, gpu.upload(a), wt, X, M, sa=gpu.upload(sa), epi=1, rtype=gpu.upload(types),
                         gate=gpu.upload(gates), **cfg)
                    gpu.wait()
                    got = X.numpy()[rows].astype(np.float64)
                    ref = x0[rows].astype(np.float64) + gates[types[rows]] * ref
                err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)
                print(f"{fmt} {name:3s} M={M:5d} {cfg}: max rel err {err:.2e}")
                assert err < 2e-3, (fmt, name, M, cfg, err)


def check_low_rank(fmt="q8", M=200, N=768, K=512, epi=0, out_dtype="f16", tile=(64, 128, 4), swz=1, ksplit=1,
                   seed=3):
    """A product with a low-rank term (LoRA): out = A x W^T + (A x A_l^T) B_l^T, against float64 with the term's
    stored codes (tight), and against the exact term (loose: the term's input side is stored in int8).

    The term has a rank-16 block per third of the columns (like separate q, k and v adapters on a fused
    product), a rank-8 block over all of them, and nothing on the last 128 columns."""
    from h3metal.ops import Scratch, low_rank
    rng = np.random.default_rng(seed)
    gpu = Gpu()
    m_pad = (M + 127) // 128 * 128
    third = (N - 128) // 3
    r_parts = [16, 16, 16, 8]
    A_l = rng.standard_normal((sum(r_parts), K)).astype(np.float32) / np.sqrt(K)
    B_l = np.zeros((N, sum(r_parts)), np.float32)
    for i in range(3):
        B_l[i * third:(i + 1) * third, 16 * i:16 * (i + 1)] = rng.standard_normal((third, 16)) * 0.5
    B_l[:N - 128, 48:56] = rng.standard_normal((N - 128, 8)) * 0.5
    if fmt in ("q8", "q4", "q4g"):
        a = rng.integers(-127, 128, size=(m_pad, K), dtype=np.int8)
        sa = rng.uniform(0.001, 0.01, size=m_pad).astype(np.float32)
        x = a.astype(np.float64) * sa[:, None].astype(np.float64)
        act = "q8"
    else:
        a = rng.standard_normal((m_pad, K)).astype(np.float16)
        sa = None
        x = a.astype(np.float64)
        act = "f16"
    if fmt == "q8":
        q = rng.integers(-127, 128, size=(N, K), dtype=np.int8)
        sw = rng.uniform(0.001, 0.01, size=N).astype(np.float32) / 10
        wt = QWeight("q8", gpu.upload(q), gpu.upload(sw), N, K)
        wf = q.astype(np.float64) * sw[:, None]
    elif fmt == "q4":
        q = rng.integers(-8, 8, size=(N, K))
        sw = rng.uniform(0.001, 0.01, size=N).astype(np.float32)
        wt = QWeight("q4", gpu.upload(pack_int4(q)), gpu.upload(sw), N, K)
        wf = q.astype(np.float64) * sw[:, None]
    elif fmt == "q4g":
        G = 128
        q = rng.integers(-8, 8, size=(N, K))
        sw = rng.uniform(0.001, 0.01, size=(N, K // G)).astype(np.float32)
        wt = QWeight("q4g", gpu.upload(pack_int4(q)), gpu.upload(np.ascontiguousarray(sw.T)), N, K, G)
        wf = q.astype(np.float64) * np.repeat(sw, G, axis=1)
    else:
        wf16 = (rng.standard_normal((N, K)) * 0.05).astype(np.float16)
        wt = QWeight("f16", gpu.upload(wf16), None, N, K)
        wf = wf16.astype(np.float64)
    lo = low_rank(gpu, A_l, B_l, Scratch(gpu), act=act)
    wt.low = lo
    main = x @ wf.T
    exact = x @ A_l.T.astype(np.float64) @ B_l.T.astype(np.float64)
    # what the kernel computes: T from the term's stored input side, rounded to half; B in half
    R = lo.R
    if act == "q8":
        a_codes = lo.a.w.numpy().astype(np.float64)
        T = (a.astype(np.float64) @ a_codes.T) * sa[:, None] * lo.a.sw.numpy()[None, :]
    else:
        T = a.astype(np.float64) @ lo.a.w.numpy().astype(np.float64).T
    stored = T.astype(np.float16).astype(np.float64) @ lo.b.numpy().astype(np.float64).T
    assert lo.b.shape == (N, R)
    A = gpu.upload(a)
    SA = gpu.upload(sa) if sa is not None else None
    if epi == 0:
        out = gpu.empty((m_pad, N), out_dtype)
        gemm(gpu, A, wt, out, M, sa=SA, tile=tile, swz=swz)
        gpu.wait()
        got = out.numpy()[:M].astype(np.float64) if out_dtype != "bf16" else \
            (out.numpy()[:M].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        ref_t, ref_e = main + stored, main + exact
    else:
        x0 = rng.standard_normal((m_pad, N)).astype(np.float32)
        X = gpu.upload(x0)
        types = rng.integers(0, 3, size=m_pad).astype(np.uint8)
        gates = rng.standard_normal((3, N)).astype(np.float32)
        gemm(gpu, A, wt, X, M, sa=SA, epi=1, rtype=gpu.upload(types), gate=gpu.upload(gates), tile=tile,
             swz=swz, ksplit=ksplit)
        gpu.wait()
        got = X.numpy()[:M].astype(np.float64)
        ref_t = x0.astype(np.float64) + gates[types] * (main + stored)
        ref_e = x0.astype(np.float64) + gates[types] * (main + exact)
        ref_t, ref_e = ref_t, ref_e
    ref_t, ref_e = ref_t[:M], ref_e[:M]
    term = np.abs(exact[:M]).max() / np.abs(main[:M]).max()
    err_t = np.abs(got - ref_t).max() / np.abs(ref_t).max()
    err_e = np.abs(got - ref_e).max() / np.abs(ref_e).max()
    print(f"low rank {fmt:4s} epi={epi} out={out_dtype} tile={tile}{f' ksplit={ksplit}' if ksplit > 1 else ''}: "
          f"term/main {term:.2f}, max rel err {err_t:.2e} (stored term), {err_e:.2e} (exact term)")
    assert term > 0.3, term  # the term is large enough for the check to see it
    assert err_t < (2e-3 if out_dtype != "bf16" else 1e-2), err_t
    assert err_e < 1e-2, err_e


if __name__ == "__main__":
    check_low_rank("q8")
    check_low_rank("q8", out_dtype="bf16", tile=(64, 256, 8))
    check_low_rank("q8", out_dtype="f32", tile=(32, 256, 4), swz=4)
    check_low_rank("q8", epi=1, K=1024, ksplit=2)
    check_low_rank("q8", epi=1, K=1024, tile=(32, 256, 4), ksplit=2)
    check_low_rank("q4", epi=1)
    check_low_rank("q4g", K=1024, out_dtype="f32")
    check_low_rank("f16", M=130, N=384 + 128, K=256)
    for fmt in ("q8", "q4", "q4g", "f16", "h8", "h4", "h4g"):
        check(fmt)
    check("q4g", epi=1)
    check("q8", out_dtype="bf16")
    check("q4g", out_dtype="f32", M=130, N=384, K=1024)
    check("q4g", out_dtype="f32", M=130, N=384, K=1024, tile=(128, 128, 8))
    for fmt in ("q8", "q4", "h8", "h4"):  # K in two dispatches, accumulated into the residual
        check(fmt, epi=1, K=1024, ksplit=2)
    check("q8", epi=1, K=2048, ksplit=4, tile=(64, 256, 8))
    check("q8", N=1024, tile=(32, 512, 8))  # wide tiles are exact where they divide N
    check("q8", epi=1, N=1024, K=2048, ksplit=4, tile=(32, 512, 8))
    check_partial_tiles_refused()
    check_dit_tiles()
    check_dit_tiles("z-image")
    check_dit_tiles("flux2-4b")
    check_dit_tiles("flux2-9b")
    print("ok")
