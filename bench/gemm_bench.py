"""Throughput of the GEMM kernels at the transformer's shapes."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu
from h3metal.ops import QWeight, gemm

SHAPES = [("qkv", 5376, 21504), ("out", 7168, 5376), ("fc1", 5376, 28672), ("fc2", 14336, 5376)]

def run(gpu, fmt, M, K, N, tile, epi=0, reps=3, G=128, swz=1):
    rng = np.random.default_rng(0)
    m_pad = (M + tile[0] - 1) // tile[0] * tile[0]
    A = gpu.upload(rng.integers(-127, 128, size=(m_pad, K), dtype=np.int8)) if fmt != "f16" else \
        gpu.upload(rng.standard_normal((m_pad, K)).astype(np.float16))
    SA = gpu.upload(np.full(m_pad, 0.01, np.float32))
    if fmt == "q8":
        wt = QWeight("q8", gpu.upload(rng.integers(-127, 128, size=(N, K), dtype=np.int8)), gpu.upload(np.full(N, .01, np.float32)), N, K)
    elif fmt in ("q4", "q4g"):
        codes = gpu.upload(rng.integers(0, 256, size=(N, K // 2), dtype=np.uint8))
        sw = gpu.upload(np.full(N * (K // G if fmt == "q4g" else 1), .01, np.float32))
        wt = QWeight(fmt, codes, sw, N, K, G)
    else:
        wt = QWeight("f16", gpu.upload((rng.standard_normal((N, K)) * .02).astype(np.float16)), None, N, K)
    if epi == 0:
        out = gpu.empty((m_pad, N), "bf16")
        args = {}
    else:
        out = gpu.empty((m_pad, N), "f32")
        args = dict(epi=1, rtype=gpu.upload(np.zeros(m_pad, np.uint8)), gate=gpu.upload(np.ones((3, N), np.float32)))
    gemm(gpu, A, wt, out, M, sa=SA, tile=tile, swz=swz, **args); gpu.wait()
    best = 1e9
    for _ in range(reps):
        for _ in range(3):
            gemm(gpu, A, wt, out, M, sa=SA, tile=tile, swz=swz, **args)
        best = min(best, gpu.wait() / 3)
    return best, 2.0 * M * K * N / best / 1e12

if __name__ == "__main__":
    gpu = Gpu()
    Ms = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "4396").split(",")]
    fmts = (sys.argv[2] if len(sys.argv) > 2 else "q8,q4,q4g").split(",")
    tiles = [(64, 128, 4)]
    G = int(os.environ.get("G", "128"))
    for M in Ms:
        for name, K, N in SHAPES:
            for fmt in fmts:
                res = []
                for t in tiles:
                    try:
                        dt, tf = run(gpu, fmt, M, K, N, t, G=G, swz=int(os.environ.get("SWZ", "1")))
                        res.append((tf, dt, t))
                    except Exception as e:
                        pass
                tf, dt, t = max(res)
                line = f"M={M:6d} {name:4s} {fmt:4s} best tile {t}: {dt*1e3:8.2f} ms {tf:6.2f} TOPS   " + \
                       " ".join(f"{r[2][0]}x{r[2][1]}x{r[2][2]}:{r[0]:.1f}" for r in res)
                print(line, flush=True)
