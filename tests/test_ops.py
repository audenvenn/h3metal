"""Row kernels (norm + modulation + rotation + int8, rotation of SwiGLU outputs, q/k norm + rope, fp32 linear)."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal.gpu import Gpu
from h3metal import ops

def hadamard(n=256):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)

def rot(x):
    H = hadamard()
    M, D = x.shape
    return (x.reshape(M, D // 256, 256) @ H.T).reshape(M, D)

def bf16(a):
    u = np.asarray(a, np.float32).view(np.uint32)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return u

def from_bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)

def test_norm_mod(gpu, M=37, D=5376, ln=False):
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((M, D)) * 3 + (50 if ln else 0)).astype(np.float32)  # LN: a large mean to remove
    mod = rng.standard_normal((3, 2, D)).astype(np.float32)
    rt = rng.integers(0, 3, size=M).astype(np.uint8)
    X, MOD, RT = gpu.upload(x), gpu.upload(mod), gpu.upload(rt)
    out = gpu.empty((M, D), "i8"); sa = gpu.empty((M,), "f32")
    ops.norm_mod_q8(gpu, X, MOD, RT, out, sa, M, D, eps=1e-5, ln=ln)
    of = gpu.empty((M, D), "f32")
    ops.norm_mod_q8(gpu, X, MOD, RT, of, None, M, D, eps=1e-5, out_f=True, ln=ln)
    gpu.wait()
    xc = x.astype(np.float64) - (x.astype(np.float64).mean(1, keepdims=True) if ln else 0)
    xn = xc / np.sqrt((xc ** 2).mean(1, keepdims=True) + 1e-5)
    y = xn * mod[rt, 0] + mod[rt, 1]
    assert np.abs(of.numpy() - y).max() < 1e-3 * np.abs(y).max(), "out_f"
    r = rot(y)
    got = out.numpy().astype(np.float64) * sa.numpy()[:, None]
    err = np.abs(got - r).max() / np.abs(r).max()
    print(f"norm_mod_q8{' LN' if ln else ''} D={D}: rel err {err:.3e} (int8 step {1/127:.3e})")
    assert err < 0.6 / 127
    oh = gpu.empty((M, D), "f16")
    ops.norm_mod_q8(gpu, X, MOD, RT, oh, None, M, D, eps=1e-5, out_hr=True, ln=ln)
    gpu.wait()
    e2 = np.abs(oh.numpy().astype(np.float64) - r).max() / np.abs(r).max()
    print(f"norm_mod rotated half: rel err {e2:.2e}")
    assert e2 < 2e-3

def test_rot(gpu, M=19, D=14336, swiglu=True):
    rng = np.random.default_rng(2)
    xin = (rng.standard_normal((M, 2 * D if swiglu else D)) * 2).astype(np.float32)
    X = gpu.upload(bf16(xin), "bf16")
    xf = from_bf16(bf16(xin)).astype(np.float64)
    out = gpu.empty((M, D), "i8"); sa = gpu.empty((M,), "f32")
    ops.rot_q8(gpu, X, out, sa, M, D, swiglu=swiglu)
    gpu.wait()
    if swiglu:
        g = xf[:, D:]
        y = xf[:, :D] * g / (1 + np.exp(-g))
    else:
        y = xf
    r = rot(y)
    got = out.numpy().astype(np.float64) * sa.numpy()[:, None]
    err = np.abs(got - r).max() / np.abs(r).max()
    print(f"rot_q8 swiglu={swiglu} D={D}: rel err {err:.3e}")
    assert err < 0.6 / 127
    oh = gpu.empty((M, D), "f16")
    ops.rot_q8(gpu, X, oh, None, M, D, swiglu=swiglu, out_hr=True)
    gpu.wait()
    e2 = np.abs(oh.numpy().astype(np.float64) - r).max() / np.abs(r).max()
    print(f"rot rotated half: rel err {e2:.2e}")
    assert e2 < 2e-3

def test_qk(gpu, M=21, Hq=56, Hk=56, Hv=56, hd=128, rot_dims=96, norm=True):
    rng = np.random.default_rng(3)
    S = Hq + Hk + Hv
    x = (rng.standard_normal((M, S * hd)) * 2).astype(np.float32)
    X = gpu.upload(bf16(x), "bf16")
    xf = from_bf16(bf16(x)).astype(np.float64)
    wq = rng.uniform(0.5, 1.5, hd).astype(np.float32); wk = rng.uniform(0.5, 1.5, hd).astype(np.float32)
    ang = rng.uniform(0, 6.28, (M, rot_dims // 2))
    cs, sn = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    rows = 32
    Q = gpu.empty((Hq, rows, hd), "f16"); K = gpu.empty((Hk, rows, hd), "f16"); V = gpu.empty((Hv, rows, hd), "f16")
    ops.qk_norm_rope(gpu, X, gpu.upload(wq), gpu.upload(wk), gpu.upload(cs), gpu.upload(sn), Q, K, V, M=M, Hq=Hq,
                     Hk=Hk, Hv=Hv, hd=hd, out_rows=rows, rot=rot_dims, eps=1e-5, norm=norm)
    gpu.wait()
    def ref(h, w):
        v = h
        if norm:
            v = v / np.sqrt((v ** 2).mean(-1, keepdims=True) + 1e-5) * w
        r = rot_dims // 2
        x1, x2 = v[..., :r].copy(), v[..., r:2 * r].copy()
        c, s = cs[:, None, :] if v.ndim == 3 else cs, sn[:, None, :] if v.ndim == 3 else sn
        v = v.copy()
        v[..., :r] = x1 * c - x2 * s
        v[..., r:2 * r] = x2 * c + x1 * s
        return v
    xs = xf.reshape(M, S, hd)
    q_ref = ref(xs[:, :Hq], wq); k_ref = ref(xs[:, Hq:Hq + Hk], wk); v_ref = xs[:, Hq + Hk:]
    eq = np.abs(Q.numpy()[:, :M].transpose(1, 0, 2) - q_ref).max()
    ek = np.abs(K.numpy()[:, :M].transpose(1, 0, 2) - k_ref).max()
    ev = np.abs(V.numpy()[:, :M].transpose(1, 0, 2) - v_ref).max()
    print(f"qk_norm_rope hd={hd} rot={rot_dims}: max abs err q {eq:.2e} k {ek:.2e} v {ev:.2e}")
    assert max(eq, ek) < 5e-3 * max(1, np.abs(q_ref).max()) and ev < 2e-2

def test_linear(gpu, M=33, N=96, K=5376):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((M, K)).astype(np.float32); w = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    out = gpu.empty((M, N), "f32")
    ops.linear_f32(gpu, gpu.upload(x), gpu.upload(w), gpu.upload(b), out, M=M, N=N, K=K)
    gpu.wait()
    err = np.abs(out.numpy() - (x @ w.T + b)).max()
    print(f"linear_f32 N={N} K={K}: max abs err {err:.2e}")
    assert err < 1e-3

if __name__ == "__main__":
    gpu = Gpu()
    test_norm_mod(gpu)
    test_norm_mod(gpu, D=5120)
    test_norm_mod(gpu, D=3072, ln=True)  # FLUX.2's LayerNorm
    test_rot(gpu)
    test_rot(gpu, D=7168, swiglu=False)
    test_qk(gpu)
    test_qk(gpu, Hq=64, Hk=8, Hv=8, rot_dims=128)
    test_qk(gpu, Hq=16, Hk=16, Hv=16, hd=64, rot_dims=32)
    test_linear(gpu)
    test_linear(gpu, M=40, N=5376, K=96)
    print("ok")
