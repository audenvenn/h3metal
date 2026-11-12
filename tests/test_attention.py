"""Flash attention kernel against NumPy."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu
from h3metal.ops import attention

def ref_attn(q, k, v, scale, causal=False, off=0):
    H, Lq, D = q.shape
    KVH = k.shape[0]
    out = np.zeros((Lq, H * D))
    for h in range(H):
        kh = h // (H // KVH)
        s = q[h].astype(np.float64) @ k[kh].T.astype(np.float64) * scale
        if causal:
            i = np.arange(Lq)[:, None]; j = np.arange(k.shape[1])[None, :]
            s = np.where(j > i + off, -np.inf, s)
        s -= s.max(1, keepdims=True); p = np.exp(s); p /= p.sum(1, keepdims=True)
        out[:, h * D:(h + 1) * D] = p @ v[kh].astype(np.float64)
    return out

def check(Lq, L, H, KVH, D, causal=False, bk=32, wm=4, seed=0, **var):
    rng = np.random.default_rng(seed)
    gpu = Gpu()
    q_rows = (Lq + 63) // 64 * 64; k_rows = (L + bk - 1) // bk * bk
    q = (rng.standard_normal((H, Lq, D)) * 1.0).astype(np.float16)
    k = (rng.standard_normal((KVH, L, D)) * 1.0).astype(np.float16)
    v = rng.standard_normal((KVH, L, D)).astype(np.float16)
    Q = gpu.empty((H, q_rows, D), "f16"); Q.numpy()[:, :Lq] = q
    K = gpu.empty((KVH, k_rows, D), "f16"); K.numpy()[:, :L] = k
    Vt = gpu.empty((KVH, k_rows, D), "f16"); Vt.numpy()[:, :L] = v
    O = gpu.empty((q_rows, H * D), "f16")
    scale = D ** -0.5
    attention(gpu, Q, K, Vt, O, Lq=Lq, L=L, H=H, KVH=KVH, D=D, q_rows=q_rows, k_rows=k_rows, scale=scale,
              causal=causal, bk=bk, wm=wm, **var)
    gpu.wait()
    got = O.numpy()[:Lq].astype(np.float64)
    ref = ref_attn(q, k, v, scale, causal)
    err = np.abs(got - ref).max()
    print(f"Lq={Lq} L={L} H={H} KVH={KVH} D={D} causal={causal} bk={bk} {var}: max abs err {err:.2e}")
    assert err < 5e-3, err

if __name__ == "__main__":
    check(100, 100, 2, 2, 128)
    check(77, 300, 3, 3, 128, bk=64, KPAD=0)
    check(64, 64, 4, 1, 64)
    check(130, 130, 8, 2, 128, causal=True)
    check(1000, 1037, 2, 2, 128)
    for var in ({"KPAD": 0},):
        check(100, 100, 2, 2, 128, wm=8, **var)
        check(77, 300, 3, 3, 128, bk=32, **var)
        check(64, 64, 4, 1, 64, **var)
        check(130, 130, 8, 2, 128, causal=True, **var)
        check(1000, 1037, 2, 2, 128, wm=8, **var)
    print("ok")



def check_q8(Lq, L, H, KVH, D, causal=False, wm=8, seed=0):
    """Integer score product: per-row int8 queries and keys, against the float reference of the dequantized
    inputs."""
    rng = np.random.default_rng(seed)
    gpu = Gpu()
    q_rows = (Lq + 255) // 256 * 256
    k_rows = (L + 31) // 32 * 32
    q = rng.standard_normal((H, Lq, D)).astype(np.float32)
    k = rng.standard_normal((KVH, L, D)).astype(np.float32)
    v = rng.standard_normal((KVH, L, D)).astype(np.float16)

    def q8(x):
        s = np.abs(x).max(-1, keepdims=True) / 127
        return np.rint(x / s).astype(np.int8), s[..., 0].astype(np.float32)
    qi, qs = q8(q)
    ki, ks = q8(k)
    Q = gpu.empty((H, q_rows, D), "i8"); Q.numpy()[:, :Lq] = qi
    K = gpu.empty((KVH, k_rows, D), "i8"); K.numpy()[:, :L] = ki
    Vt = gpu.empty((KVH, k_rows, D), "f16"); Vt.numpy()[:, :L] = v
    QS = gpu.empty((H, q_rows), "f32"); QS.numpy()[:, :Lq] = qs
    KS = gpu.empty((KVH, k_rows), "f32"); KS.numpy()[:, :L] = ks
    O = gpu.empty((q_rows, H * D), "f16")
    attention(gpu, Q, K, Vt, O, Lq=Lq, L=L, H=H, KVH=KVH, D=D, q_rows=q_rows, k_rows=k_rows, scale=D ** -0.5,
              causal=causal, wm=wm, q_scale=QS, k_scale=KS)
    gpu.wait()
    got = O.numpy()[:Lq].astype(np.float64)
    ref = ref_attn(qi * qs[..., None], ki * ks[..., None], v, D ** -0.5, causal)
    err = np.abs(got - ref).max()
    print(f"q8 Lq={Lq} L={L} H={H} KVH={KVH} D={D} causal={causal}: max abs err {err:.2e}")
    assert err < 5e-3


if __name__ == "__main__":
    check_q8(100, 100, 2, 2, 128)
    check_q8(300, 1037, 3, 3, 128)
    check_q8(130, 130, 8, 2, 128, causal=True)
    check_q8(64, 64, 4, 4, 64)
    print("ok q8")
