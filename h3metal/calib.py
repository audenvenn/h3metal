"""GPTQ for the transformer's matrices, calibrated on real denoising trajectories.

A second conversion pass. It takes an already converted file (any format;
its refiner, embeddings, modulation tables and heads are reused) and the
original checkpoint, and rewrites the 50 blocks' matrices in another format,
choosing each matrix's codes with GPTQ (Frantar et al., 2022) instead of
rounding to nearest:

* Calibration states come from reference runs of the official pipeline
  (tests/ref/reference.py): their text embeddings and the latents of every
  step, rebuilt from the saved noise and velocities. They enter block 0 the
  way the engine builds a step's sequence.
* Blocks are processed in order. For each, the inputs of its four products
  (rotated, in half precision, exactly what the engine quantizes) are
  collected over all states into Hessians H = sum x^T x, every matrix is
  quantized with GPTQ against its H, and the block is then run with the
  quantized matrices through the engine's own kernels to produce the next
  block's states (sequential quantization).
"""

from __future__ import annotations

import os
import time

import numpy as np

from . import dit as ditm
from . import ops
from .convert import _signs
from .gpu import Gpu
from .ops import QWeight
from .quant import pack_int4, rotate_in
from .st import Sharded
from .weights import WeightFile, Writer, bf16_bits


def gptq(w: np.ndarray, H: np.ndarray, bits: int, damp: float = 0.01, block: int = 128,
         act_order: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """W [N][K] float, H [K][K] -> (integer codes [N][K], row scales [N]) with symmetric per-row scales."""
    W = w.astype(np.float32, copy=True)
    N, K = W.shape
    qmax, qmin = (127, -127) if bits == 8 else (7, -8)
    H = H.astype(np.float64, copy=True)
    dead = np.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    # Per-row scales from a clipping search on the unquantized rows (weighted by the Hessian diagonal).
    d = np.diag(H)
    amax = np.abs(W).max(1)
    best_err = np.full(N, np.inf)
    scale = np.maximum(amax / qmax, 1e-12)
    for c in (1.0, .95, .9, .85, .8, .75, .7) if bits == 4 else (1.0, .97, .94, .9):
        s = np.maximum(amax * c / qmax, 1e-12)
        q = np.clip(np.rint(W / s[:, None]), qmin, qmax)
        err = (((q * s[:, None] - W) ** 2) * d[None, :]).sum(1)
        upd = err < best_err
        best_err = np.where(upd, err, best_err)
        scale = np.where(upd, s, scale)
    perm = np.argsort(-np.diag(H)) if act_order else np.arange(K)
    W = W[:, perm]
    H = H[perm][:, perm]
    H[np.diag_indices(K)] += damp * np.mean(np.diag(H))
    L = np.linalg.cholesky(np.linalg.inv(H))          # lower
    Hinv = L.T.astype(np.float32)                      # upper Cholesky factor of H^-1
    Q = np.zeros((N, K), np.int8)
    s = scale.astype(np.float32)[:, None]
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].copy()
        Err = np.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = W1[:, i]
            q = np.clip(np.rint(col / s[:, 0]), qmin, qmax)
            Q[:, i1 + i] = q.astype(np.int8)
            e = (col - q * s[:, 0]) / Hi[i, i]
            W1[:, i:] -= e[:, None] * Hi[i, i:][None, :]
            Err[:, i] = e
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    inv = np.argsort(perm)
    return Q[:, inv], scale.astype(np.float32)


class _Hessian:
    """Accumulates x^T x for one product's inputs on the GPU (MLX)."""

    def __init__(self, K: int):
        import mlx.core as mx
        self.mx = mx
        self.H = mx.zeros((K, K), dtype=mx.float32)
        self.n = 0

    def add(self, x: np.ndarray):
        mx = self.mx
        a = mx.array(x).astype(mx.float32)
        self.H = self.H + a.T @ a
        self.n += x.shape[0]
        mx.eval(self.H)

    def get(self) -> np.ndarray:
        return np.array(self.H).astype(np.float64) / max(self.n, 1)


def trajectory(ref: dict, sigmas) -> list[np.ndarray]:
    """The latents [C][h][w] the reference run denoised at each step, rebuilt from its noise and velocities."""
    ch, cw = (int(v) for v in ref["canvas"])
    lat = ref["noise"].reshape(-1, ch // 16, cw // 16).astype(np.float32)
    rows = ditm.patchify(lat)
    out = []
    for s in range(len(sigmas) - 1):
        out.append(rows.copy())
        rows = ditm.euler(rows, ref[f"vel{s}"], s, sigmas)
    return out


def record_run(dit_path: str, emb: np.ndarray, canvas, seed: int, out: str):
    """A calibration run made with the engine itself (any converted DiT, int8 in practice): the text embeddings,
    the starting noise and every step's velocity, in the format reference runs are saved in (what calibrate()
    reads). Much faster than a reference run of the official modules, and as good for collecting statistics."""
    from .rng import TorchRandn
    g = Gpu()
    wf = WeightFile(g, dit_path)
    m = ditm.DiT(g, wf)
    T = emb.shape[0]
    text = g.empty((ops.pad_rows(T, 128), emb.shape[1]), "bf16")
    text.numpy()[:T] = bf16_bits(emb)
    rows = m.encode_text(text, T)
    ch, cw = (int(v) for v in canvas)
    lh, lw = ch // 16, cw // 16
    C = m.cfg["latents"]
    m.prepare(ditm.Layout(T, np.ones(T, np.uint8), [], lh, lw))
    noise = TorchRandn(seed).normal((1, C, 1, lh, lw))
    x = ditm.patchify(noise.reshape(C, lh, lw))
    vels = {}
    for s in range(m.n_steps):
        v = m.step(s, rows, x)
        vels[f"vel{s}"] = v
        x = ditm.euler(x, v, s, m.sigmas)
    np.savez(out, emb=emb.astype(np.float32), noise=noise, canvas=np.array([ch, cw]), **vels)
    m.free()
    wf.close()


def calibrate(src: str, base: str, out: str, refs: list[str], fmt="q4", log=print):
    """Rewrites the blocks of `base` (a converted DiT) into `out` with GPTQ codes.

    fmt: "q4" or "q8" for all four products, or one per product, e.g.
    {"qkv": "q8", "out": "q8", "fc1": "q4", "fc2": "q4"}.
    """
    fmts = dict(fmt) if isinstance(fmt, dict) else {k: fmt for k in ("qkv", "out", "fc1", "fc2")}
    if set(fmts) != {"qkv", "out", "fc1", "fc2"} or not set(fmts.values()) <= {"q4", "q8"}:
        raise ValueError(f"formats per product must be q4 or q8: {fmt}")
    g = Gpu()
    wf = WeightFile(g, base)
    model = ditm.DiT(g, wf)
    D, H, hd, F, inner = model.D, model.H, model.hd, model.F, model.inner
    seed = wf.meta["seed"]
    st = Sharded(os.path.join(src, "transformer"))
    # Calibration states at the input of block 0: one per (reference run, step).
    states = []
    for path in refs:
        r = dict(np.load(path))
        emb = r["emb"].astype(np.float32)
        T = emb.shape[0]
        text = g.empty((ops.pad_rows(T, 128), emb.shape[1]), "bf16")
        text.numpy()[:T] = bf16_bits(emb)
        text_rows = model.encode_text(text, T).numpy()[:T].copy()
        ch, cw = (int(v) for v in r["canvas"])
        lay = ditm.Layout(T, np.ones(T, np.uint8), [], ch // 16, cw // 16)
        for s, lat_rows in enumerate(trajectory(r, model.sigmas)):
            x = np.empty((lay.length, D), np.float32)
            x[:T] = text_rows
            x[T:] = lat_rows @ wf.host("x_embed.w").T + wf.host("x_embed.b")
            states.append({"x": x, "layout": lay, "step": s})
    log(f"{len(states)} calibration states from {len(refs)} reference runs")
    meta = dict(wf.meta)
    meta["fmts"] = fmts
    meta["gptq"] = {"refs": [os.path.basename(p) for p in refs], "states": len(states)}
    w = Writer(out, meta)
    for name, e in wf.entries.items():  # everything but the blocks' matrices is copied
        if name.startswith("blocks.") and name.split(".")[-1] in ("qkv", "out", "fc1", "fc2"):
            continue
        if e.get("kind") == "linear":
            def raw(t):
                return wf.map.host(t["offset"], t["nbytes"]).copy()
            w.linear(name, e["fmt"], e["N"], e["K"], raw(e["w"]),
                     raw(e["sw"]).view(np.float32) if "sw" in e else None,
                     bias=raw(e["bias"]).view(np.float32) if "bias" in e else None, G=e.get("G", 128))
        else:
            w.tensor(name, wf.host(name).copy(), e["dtype"], host_only=e.get("host_only", False))
    t0 = time.time()
    for l in range(model.L):
        p = f"blocks.{l}"
        s_in, s_o = _signs(seed, l, 1, D), _signs(seed, l, 2, inner)
        s_mlp, s_f = _signs(seed, l, 3, D), _signs(seed, l, 4, F)
        qkv = st.get(f"{p}.attn.qkv_proj.weight").reshape(H, 3, hd, D).transpose(1, 0, 2, 3).reshape(3 * inner, D)
        qkv[2 * inner:] *= s_o[:, None]
        fc1 = st.get(f"{p}.mlp.fc1.weight")
        mats = {"qkv": rotate_in(qkv, s_in), "out": rotate_in(st.get(f"{p}.attn.out_proj.weight"), s_o),
                "fc1": rotate_in(np.concatenate([fc1[F:] * s_f[:, None], fc1[:F]]), s_mlp),
                "fc2": rotate_in(st.get(f"{p}.mlp.fc2.weight"), s_f)}
        del qkv, fc1
        # 1. Inputs of the four products over all states, through the block with float (half) weights.
        hw = {k: QWeight("f16", g.upload(m.astype(np.float16)), None, m.shape[0], m.shape[1])
              for k, m in mats.items()}
        hs = {"qkv": _Hessian(D), "out": _Hessian(inner), "fc1": _Hessian(D), "fc2": _Hessian(F)}
        blk = dict(model.blocks[l])
        for st_ in states:
            _block_capture(model, blk | hw, st_, l, hs)
        # 2. GPTQ.
        qw = {}
        for k, m in mats.items():
            bits = 4 if fmts[k] == "q4" else 8
            q, sc = gptq(m, hs[k].get(), bits)
            codes = q if bits == 8 else pack_int4(q)
            w.linear(f"{p}.{k}", fmts[k], m.shape[0], m.shape[1], codes, sc)
            qw[k] = QWeight(fmts[k], g.upload(codes), g.upload(sc), m.shape[0], m.shape[1])
        del hw, mats
        # 3. The next block's states, through the engine's kernels with the quantized matrices.
        for st_ in states:
            st_["x"] = _block_run(model, blk | qw, st_, l)
        log(f"  block {l + 1}/{model.L} ({time.time() - t0:.0f} s)")
    w.close()
    wf.close()
    log(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB)")


def _prepare(model, st_):
    model.prepare(st_["layout"])
    b = model._bufs
    S = st_["layout"].length
    b["x"].numpy()[:S] = st_["x"]
    return b, S


def _tables(model, st_, l):
    D, s, L3 = model.D, st_["step"], 3 * 2 * model.D
    o = s * model.L + l
    return (model.mod_attn.view((3, 2, D), offset_elems=o * L3), model.mod_mlp.view((3, 2, D), offset_elems=o * L3),
            model.gate_attn.view((3, D), offset_elems=o * 3 * D), model.gate_mlp.view((3, D), offset_elems=o * 3 * D))


def _block_capture(model, blk, st_, l, hs):
    """Runs block l in half precision and adds the four products' inputs to their Hessians."""
    g, D, H, hd, F, inner = model.gpu, model.D, model.H, model.hd, model.F, model.inner
    b, S = _prepare(model, st_)
    Sp = b["Sp"]
    mod1, mod2, g1, g2 = _tables(model, st_, l)
    eps = model.cfg["norm_eps"]
    x, rt = b["x"], b["rt"]
    a = g.empty((Sp, max(D, inner, F)), "f16")
    ops.norm_mod_q8(g, x, mod1, rt, a.view((Sp, D)), None, S, D, eps=eps, out_hr=True)
    g.wait()
    hs["qkv"].add(a.view((Sp, D)).numpy()[:S])
    qkv = b["wide"].view((Sp, 3 * inner))
    ops.gemm(g, a.view((Sp, D)), blk["qkv"], qkv, S)
    ops.qk_norm_rope(g, qkv, blk["q_norm"], blk["k_norm"], b["cos"], b["sin"], b["Q"], b["K"], b["V"], M=S, Hq=H,
                     Hk=H, Hv=H, hd=hd, out_rows=Sp, rot=model.rot, eps=model.cfg["qk_eps"])
    ops.attention(g, b["Q"], b["K"], b["V"], b["o"], Lq=S, L=S, H=H, KVH=H, D=hd, q_rows=Sp, k_rows=Sp,
                  scale=hd ** -0.5, **model.attn_cfg)
    ops.rot_q8(g, b["o"], a.view((Sp, inner)), None, S, inner, out_hr=True)
    g.wait()
    hs["out"].add(a.view((Sp, inner)).numpy()[:S])
    ops.gemm(g, a.view((Sp, inner)), blk["out"], x, S, epi=1, rtype=rt, gate=g1, g_stride=D)
    ops.norm_mod_q8(g, x, mod2, rt, a.view((Sp, D)), None, S, D, eps=eps, out_hr=True)
    g.wait()
    hs["fc1"].add(a.view((Sp, D)).numpy()[:S])
    h = b["wide"].view((Sp, 2 * F))
    ops.gemm(g, a.view((Sp, D)), blk["fc1"], h, S)
    ops.rot_q8(g, h, a.view((Sp, F)), None, S, F, swiglu=True, out_hr=True)
    g.wait()
    hs["fc2"].add(a.view((Sp, F)).numpy()[:S])


def _block_run(model, blk, st_, l) -> np.ndarray:
    """Runs block l through the engine's kernels; returns the new residual rows."""
    b, S = _prepare(model, st_)
    mod1, mod2, g1, g2 = _tables(model, st_, l)
    model._block(b, blk, S, b["x"], mod1, mod2, g1, g2, b["rt"], model.rot, 2 * model.D, model.D)
    model.gpu.wait()
    return b["x"].numpy()[:S].copy()
