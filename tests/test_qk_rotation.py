"""Queries and keys rotated before their int8 rounding (ops.qk_norm_rope, qk_rot): attention is unchanged in exact
arithmetic, and with a few dominant channels per head, as real ones have, the int8 scores lose less.

    python tests/test_qk_rotation.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from h3metal import ops  # noqa: E402
from h3metal.gpu import Gpu  # noqa: E402


def ref_attn(q, k, v):
    s = np.einsum("hqd,hkd->hqk", q, k) / np.sqrt(q.shape[-1])
    s -= s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hqk,hkd->hqd", p, v)


def run(gpu, qkv, H, S, hd, rot, cs, sn, qk_rot):
    Sp = -(-S // 64) * 64
    Q, K = gpu.empty((H, Sp, hd), "i8"), gpu.empty((H, Sp, hd), "i8")
    V = gpu.empty((H, Sp, hd), "f16")
    qs, ks = gpu.empty((H * Sp,), "f32"), gpu.empty((H * Sp,), "f32")
    ones = gpu.upload(np.ones(hd, np.float32))
    ops.qk_norm_rope(gpu, qkv, ones, ones, cs, sn, Q, K, V, M=S, Hq=H, Hk=H, Hv=H, hd=hd, out_rows=Sp, rot=rot,
                     eps=1e-6, norm=False, q_scale=qs, k_scale=ks, qk_rot=qk_rot)
    o = gpu.empty((Sp, H * hd), "f16")
    ops.attention(gpu, Q, K, V, o, Lq=S, L=S, H=H, KVH=H, D=hd, q_rows=Sp, k_rows=Sp, scale=hd ** -0.5, wm=10,
                  bk=32, DB=1, PVORDER=1, VX=1, q_scale=qs, k_scale=ks)
    gpu.wait()
    return o.numpy()[:S].astype(np.float64).reshape(S, H, hd).transpose(1, 0, 2)


def main():
    gpu = Gpu()
    rng = np.random.default_rng(0)
    H, S, hd, rot = 4, 1000, 128, 96
    # Rows dominated by a few channels with a shared direction, the way real queries and keys are.
    base = rng.standard_normal((3, H, S, hd))
    loud = rng.choice(hd, 8, replace=False)
    base[:2, :, :, loud] = base[:2, :, :, loud] * 12 + 20 * rng.standard_normal((2, H, 1, 8))
    q, k, v = base
    qkv = np.concatenate([q.transpose(1, 0, 2).reshape(S, -1), k.transpose(1, 0, 2).reshape(S, -1),
                          v.transpose(1, 0, 2).reshape(S, -1)], 1).astype(np.float16)
    q, k, v = (x.astype(np.float16).astype(np.float64) for x in (q, k, v))
    pos = np.arange(S)[:, None] / 10000 ** (np.arange(rot // 2)[None] / (rot // 2))
    cs, sn = np.cos(pos).astype(np.float32), np.sin(pos).astype(np.float32)
    h = rot // 2
    rq, rk = q.copy(), k.copy()
    for x in (rq, rk):
        a, b = x[..., :h].copy(), x[..., h:rot].copy()
        x[..., :h], x[..., h:rot] = a * cs - b * sn, b * cs + a * sn
    ref = ref_attn(rq, rk, v)
    errs = {}
    for qk_rot in (False, True):
        got = run(gpu, gpu.upload(qkv), H, S, hd, rot, gpu.upload(cs), gpu.upload(sn), qk_rot)
        errs[qk_rot] = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        print(f"int8 scores, {'rotated' if qk_rot else 'plain'}: attention rel err {errs[qk_rot]:.4f}")
    assert errs[True] < 0.6 * errs[False], errs


if __name__ == "__main__":
    main()
