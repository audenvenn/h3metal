"""Z-Image (Tongyi-MAI/Z-Image-Turbo, Apache-2.0) on h3metal's engine: the transformer.

Z-Image's transformer is a single-stream DiT: the image's patches and the caption's rows run through the
same blocks. Each block normalizes its input (RMSNorm, scaled by 1 + the timestep's modulation), runs
attention or a SwiGLU MLP, normalizes the result again (RMSNorm) and adds it to the residual stream through a
tanh gate. Two refiner blocks run over the image rows alone (modulated), two over the caption rows alone (not
modulated: those depend on the prompt only and run once per prompt).

The conversion does what h3metal/convert.py does for MiniMax-H3-Image:

* The modulation depends only on the timestep, and Z-Image-Turbo samples a fixed eight-step schedule, so every
  modulation vector is computed here, once; the AdaLN projections never reach the device.
* The norm weights and the gates are folded into those tables.
* Every quantized matrix is rotated on its input side (h3metal/quant.py), with the rotations' random sign
  flips folded into the tables and into the rows of the producing matrix where a sign can pass through.

Z-Image's rotary embedding rotates pairs of adjacent channels (as complex numbers); the engine's kernel pairs
channel i with i + 64. Permuting the query and key rows of every head (even channels first, then odd ones)
turns one into the other and leaves every score unchanged, so the conversion does that.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from . import ops
from .convert import silu, timestep_embedding
from .dit import _tile, patchify, unpatchify
from .gpu import Gpu, Tensor
from .quant import quantize, random_signs, rotate_in
from .st import Sharded
from .weights import WeightFile, Writer

SEQ_MULTI = 32       # Z-Image pads the image rows and the caption rows to multiples of 32
ADALN_DIM = 256


def schedule(steps: int = 8, shift: float = 3.0):
    """Z-Image's sampler: sigmas linspace(1, 1/steps) shifted by 3 (FlowMatchEulerDiscreteScheduler), the
    transformer's timestep 1 - sigma, float32 like the scheduler. -> (t [steps], sigmas [steps + 1])."""
    base = np.linspace(1.0, 1.0 / steps, steps, dtype=np.float32)
    sig = (np.float32(shift) * base / (np.float32(1) + np.float32(shift - 1) * base)).astype(np.float32)
    timesteps = (sig * np.float32(1000)).astype(np.float32)
    t = ((np.float32(1000) - timesteps) / np.float32(1000)).astype(np.float32)
    return t, np.concatenate([sig, np.zeros(1, np.float32)])


def _signs(seed: int, layer: int, role: int, n: int) -> np.ndarray:
    return random_signs(n, seed * 1_000_003 + layer * 101 + role)


def _pair_perm(H: int, hd: int) -> np.ndarray:
    """Row order that turns adjacent-pair rotation into half-split rotation, per head."""
    one = np.concatenate([np.arange(0, hd, 2), np.arange(1, hd, 2)])
    return (np.arange(H)[:, None] * hd + one[None, :]).reshape(-1)


def convert_transformer(src: str, out: str, *, fmts=None, steps: int = 8, seed: int = 1, log=print):
    """src: the Z-Image directory (transformer/ with config.json and safetensors shards)."""
    f = {"qkv": "q8", "out": "q8", "fc1": "q8", "fc2": "q8", **(fmts or {})}
    with open(os.path.join(src, "transformer", "config.json")) as fh:
        cfg = json.load(fh)
    st = Sharded(os.path.join(src, "transformer"))
    D, H = cfg["dim"], cfg["n_heads"]
    hd = D // H
    F = int(D / 3 * 8)
    L, R = cfg["n_layers"], cfg["n_refiner_layers"]
    eps = cfg["norm_eps"]
    t_steps, sigmas = schedule(steps)

    # Timestep embedding (float32): t_embedder(t * t_scale), the input of every block's AdaLN projection.
    w1, b1 = st.get("t_embedder.mlp.0.weight"), st.get("t_embedder.mlp.0.bias")
    w2, b2 = st.get("t_embedder.mlp.2.weight"), st.get("t_embedder.mlp.2.bias")
    temb = timestep_embedding(t_steps * np.float32(cfg["t_scale"]), ADALN_DIM) @ w1.T + b1
    temb = (silu(temb) @ w2.T + b2).astype(np.float32)  # [steps][256]

    meta = {"kind": "zimage-dit",
            "config": {"dim": D, "heads": H, "head_dim": hd, "ffn": F, "layers": L, "refiner_layers": R,
                       "in_channels": cfg["in_channels"], "patch": 2, "cap_dim": cfg["cap_feat_dim"],
                       "axes_dims": cfg["axes_dims"], "rope_theta": cfg["rope_theta"], "norm_eps": eps},
            "timesteps": t_steps.tolist(), "sigmas": sigmas.tolist(), "seed": seed, "fmts": f}
    w = Writer(out, meta)
    t0 = time.time()

    # Z-Image orders a patch's features (dy, dx, channel); h3metal's patchify (h3metal/dit.py) orders them
    # (channel, dy, dx), so the embedding's columns and the final layer's rows are reordered to match.
    C, p = cfg["in_channels"], 2
    order = np.array([(dy * p + dx) * C + c for c in range(C) for dy in range(p) for dx in range(p)])
    xw = st.get("all_x_embedder.2-1.weight")[:, order]  # [D][C p p]
    w.tensor("x_embed.w", np.ascontiguousarray(xw, np.float32), "f32")
    w.tensor("x_embed.b", st.get("all_x_embedder.2-1.bias").astype(np.float32), "f32")
    w.tensor("x_pad", st.get("x_pad_token").reshape(D).astype(np.float32), "f32")
    w.tensor("cap_pad", st.get("cap_pad_token").reshape(D).astype(np.float32), "f32")
    w.tensor("cap_norm", st.get("cap_embedder.0.weight").astype(np.float32), "f32")
    w.tensor("cap_embed.w", st.get("cap_embedder.1.weight").astype(np.float32), "f32")
    w.tensor("cap_embed.b", st.get("cap_embedder.1.bias").astype(np.float32), "f32")

    # Final layer: LayerNorm (no affine) * (1 + scale(t)), then a linear map to the patch features.
    fa_w, fa_b = st.get("all_final_layer.2-1.adaLN_modulation.1.weight"), st.get(
        "all_final_layer.2-1.adaLN_modulation.1.bias")
    w.tensor("final_scale", (1.0 + silu(temb) @ fa_w.T + fa_b).astype(np.float32), "f32")  # [steps][D]
    w.tensor("final.w", np.ascontiguousarray(st.get("all_final_layer.2-1.linear.weight")[order], np.float32), "f32")
    w.tensor("final.b", np.ascontiguousarray(st.get("all_final_layer.2-1.linear.bias")[order], np.float32), "f32")
    perm = _pair_perm(H, hd)

    def put(name, kind, mat):
        codes, scales = quantize(f[kind], mat)
        w.linear(name, f[kind], mat.shape[0], mat.shape[1], codes, scales)

    def block(prefix: str, name: str, layer: int, modulated: bool):
        s_in, s_o = _signs(seed, layer, 1, D), _signs(seed, layer, 2, D)
        s_mlp, s_f = _signs(seed, layer, 3, D), _signs(seed, layer, 4, F)
        a = f"{prefix}.attention"
        q = st.get(f"{a}.to_q.weight")[perm]
        k = st.get(f"{a}.to_k.weight")[perm]
        v = st.get(f"{a}.to_v.weight") * s_o[:, None]
        put(f"{name}.qkv", "qkv", rotate_in(np.concatenate([q, k, v]), s_in))
        put(f"{name}.out", "out", rotate_in(st.get(f"{a}.to_out.0.weight"), s_o))
        w.tensor(f"{name}.q_norm", st.get(f"{a}.norm_q.weight")[perm[:hd]].astype(np.float32), "f32")
        w.tensor(f"{name}.k_norm", st.get(f"{a}.norm_k.weight")[perm[:hd]].astype(np.float32), "f32")
        gate = st.get(f"{prefix}.feed_forward.w1.weight")
        value = st.get(f"{prefix}.feed_forward.w3.weight")
        put(f"{name}.fc1", "fc1", rotate_in(np.concatenate([value * s_f[:, None], gate]), s_mlp))
        put(f"{name}.fc2", "fc2", rotate_in(st.get(f"{prefix}.feed_forward.w2.weight"), s_f))
        n1, n2 = st.get(f"{prefix}.attention_norm1.weight"), st.get(f"{prefix}.attention_norm2.weight")
        m1, m2 = st.get(f"{prefix}.ffn_norm1.weight"), st.get(f"{prefix}.ffn_norm2.weight")
        if modulated:
            aw, ab = st.get(f"{prefix}.adaLN_modulation.0.weight"), st.get(f"{prefix}.adaLN_modulation.0.bias")
            mod = (temb @ aw.T + ab).reshape(len(t_steps), 4, D)  # scale_msa, gate_msa, scale_mlp, gate_mlp
            sc_a, g_a, sc_m, g_m = (mod[:, i] for i in range(4))
            g_a, g_m = np.tanh(g_a), np.tanh(g_m)
        else:
            sc_a = sc_m = np.zeros((1, D), np.float32)
            g_a = g_m = np.ones((1, D), np.float32)
        n = sc_a.shape[0]
        pre_a = np.zeros((n, 2, D), np.float32)
        pre_m = np.zeros((n, 2, D), np.float32)
        pre_a[:, 0] = n1 * (1.0 + sc_a) * s_in   # norm_mod_q8: rmsnorm(x) * scale' + shift (shift 0)
        pre_m[:, 0] = m1 * (1.0 + sc_m) * s_mlp
        w.tensor(f"{name}.pre_attn", pre_a, "f32")
        w.tensor(f"{name}.pre_mlp", pre_m, "f32")
        w.tensor(f"{name}.post_attn", (g_a * n2).astype(np.float32), "f32")  # x += post * rmsnorm(y)
        w.tensor(f"{name}.post_mlp", (g_m * m2).astype(np.float32), "f32")

    for i in range(R):
        block(f"context_refiner.{i}", f"cap_refiner.{i}", 2000 + i, False)
        block(f"noise_refiner.{i}", f"x_refiner.{i}", 1000 + i, True)
    for i in range(L):
        block(f"layers.{i}", f"blocks.{i}", i, True)
        log(f"  block {i + 1}/{L} done ({time.time() - t0:.0f} s)")
    w.close()
    log(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB) in {time.time() - t0:.0f} s")


def rope_tables(pos: np.ndarray, axes_dims, theta: float):
    """Z-Image's multi-axis rotary angles: pos [S][3] -> cos, sin [S][hd/2] (axis 0's pairs first)."""
    angs = []
    for i, d in enumerate(axes_dims):
        inv = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
        angs.append(pos[:, i:i + 1].astype(np.float64) * inv[None, :])
    ang = np.concatenate(angs, axis=1).astype(np.float32)
    return np.cos(ang), np.sin(ang)


class ZImageDiT:
    """The converted transformer on the engine. encode_caption() once per prompt, then step() per step."""
    attn_cfg_q8 = dict(wm=10, bk=32, DB=1, PVORDER=1, VX=1)

    def __init__(self, gpu: Gpu, wf: WeightFile):
        self.gpu, self.wf = gpu, wf
        c = wf.meta["config"]
        self.cfg = c
        self.D, self.H, self.hd, self.F = c["dim"], c["heads"], c["head_dim"], c["ffn"]
        self.L, self.R = c["layers"], c["refiner_layers"]
        self.timesteps = np.array(wf.meta["timesteps"], np.float32)
        self.sigmas = np.array(wf.meta["sigmas"], np.float32)
        self.n_steps = len(self.timesteps)

        def blk(name):
            d = {k: wf.linear(f"{name}.{k}") for k in ("qkv", "out", "fc1", "fc2")}
            d.update({k: wf.tensor(f"{name}.{k}") for k in ("q_norm", "k_norm", "pre_attn", "pre_mlp",
                                                             "post_attn", "post_mlp")})
            return d
        self.blocks = [blk(f"blocks.{i}") for i in range(self.L)]
        self.x_refiner = [blk(f"x_refiner.{i}") for i in range(self.R)]
        self.cap_refiner = [blk(f"cap_refiner.{i}") for i in range(self.R)]
        self.x_embed_w, self.x_embed_b = wf.tensor("x_embed.w"), wf.tensor("x_embed.b")
        self.x_pad, self.cap_pad = wf.host("x_pad"), wf.host("cap_pad")
        self.cap_norm = wf.host("cap_norm")
        self.cap_w, self.cap_b = wf.host("cap_embed.w"), wf.host("cap_embed.b")
        self.final_scale = wf.tensor("final_scale")
        self.final_w, self.final_b = wf.tensor("final.w"), wf.tensor("final.b")
        self.zeros = gpu.upload(np.zeros(self.D, np.float32))
        self.patch_dim = self.x_embed_w.shape[1]
        self._bufs = None

    def _alloc(self, S: int):
        g, D, H, hd, F = self.gpu, self.D, self.H, self.hd, self.F
        Sp = ops.pad_rows(S, 128)
        if self._bufs and self._bufs["Sp"] >= Sp:
            return self._bufs
        b = {"Sp": Sp,
             "x": g.empty((Sp, D), "f32"), "y": g.empty((Sp, D), "f32"),
             "a8": g.empty((Sp, max(F, D)), "i8"), "sa": g.empty((Sp,), "f32"),
             "wide": g.empty((Sp, max(3 * D, 2 * F)), "bf16"), "o": g.empty((Sp, D), "bf16"),
             "Q": g.empty((H, Sp, hd), "i8"), "K": g.empty((H, Sp, hd), "i8"), "V": g.empty((H, Sp, hd), "f16"),
             "qs": g.empty((H, Sp), "f32"), "ks": g.empty((H, Sp), "f32"),
             "rt": g.empty((Sp,), "u8"),
             "cos": g.empty((Sp, hd // 2), "f32"), "sin": g.empty((Sp, hd // 2), "f32"),
             "lat": g.empty((Sp, self.patch_dim), "f32"), "vel": g.empty((Sp, self.patch_dim), "f32"),
             "cap": g.empty((Sp, D), "f32")}
        b["rt"].numpy()[:] = 0
        self._bufs = b
        return b

    def free(self):
        self._bufs = None

    def _block(self, b, blk, x: Tensor, r0: int, S: int, s: int, pending: Tensor | None = None) -> Tensor:
        """One block over rows r0 .. r0 + S of x (the rotary tables' rows r0 .. too); s: the step's table row.

        A block's last update, x += post_mlp * rmsnorm(y), is left to the caller (returned): the next block adds it
        in the same pass as its first norm (pending), or _flush() adds it on its own."""
        g, D, H, hd, F = self.gpu, self.D, self.H, self.hd, self.F
        Sp = b["Sp"]
        eps = self.cfg["norm_eps"]
        a8, sa, y = b["a8"], b["sa"], b["y"]
        xr = x.row(r0, S)
        pre_a = blk["pre_attn"].view((1, 2, D), offset_elems=min(s, blk["pre_attn"].shape[0] - 1) * 2 * D)
        pre_m = blk["pre_mlp"].view((1, 2, D), offset_elems=min(s, blk["pre_mlp"].shape[0] - 1) * 2 * D)
        post_a = blk["post_attn"].view((D,), offset_elems=min(s, blk["post_attn"].shape[0] - 1) * D)
        post_m = blk["post_mlp"].view((D,), offset_elems=min(s, blk["post_mlp"].shape[0] - 1) * D)
        fused = {"y": y, "post": pending} if pending is not None else {}
        ops.norm_mod_q8(g, xr, pre_a, b["rt"], a8.view((Sp, D)), sa, S, D, eps=eps, **fused)
        qkv = b["wide"].view((Sp, 3 * D))
        ops.gemm(g, a8.view((Sp, D)), blk["qkv"], qkv, S, sa=sa, **_ztile(blk["qkv"], S))
        Qb, Kb = b["Q"], b["K"]
        cos, sin = b["cos"].row(r0, S), b["sin"].row(r0, S)
        ops.qk_norm_rope(g, qkv, blk["q_norm"], blk["k_norm"], cos, sin, Qb, Kb, b["V"], M=S, Hq=H, Hk=H, Hv=H,
                         hd=hd, out_rows=Sp, rot=hd, eps=eps, q_scale=b["qs"], k_scale=b["ks"])
        ops.attention(g, Qb, Kb, b["V"], b["o"], Lq=S, L=S, H=H, KVH=H, D=hd, q_rows=Sp, k_rows=Sp,
                      scale=hd ** -0.5, q_scale=b["qs"], k_scale=b["ks"], **self.attn_cfg_q8)
        ops.rot_q8(g, b["o"], a8.view((Sp, D)), sa, S, D)
        ops.gemm(g, a8.view((Sp, D)), blk["out"], y, S, sa=sa, **_ztile(blk["out"], S))
        ops.norm_mod_q8(g, xr, pre_m, b["rt"], a8.view((Sp, D)), sa, S, D, eps=eps, y=y, post=post_a)
        h = b["wide"].view((Sp, 2 * F))
        ops.gemm(g, a8.view((Sp, D)), blk["fc1"], h, S, sa=sa, **_ztile(blk["fc1"], S))
        ops.rot_q8(g, h, a8.view((Sp, F)), sa, S, F, swiglu=True)
        ops.gemm(g, a8.view((Sp, F)), blk["fc2"], y, S, sa=sa, **_ztile(blk["fc2"], S))
        return post_m

    def _flush(self, b, x: Tensor, r0: int, S: int, pending: Tensor | None):
        if pending is not None:
            ops.post_norm_add(self.gpu, x.row(r0, S), b["y"], pending, S, self.D, eps=self.cfg["norm_eps"])

    def prepare(self, n_cap: int, lat_h: int, lat_w: int):
        """Per-request layout: [image rows, padded to 32][caption rows, padded to 32] and their rotary tables."""
        p = self.cfg["patch"]
        self.hp, self.wp = lat_h // p, lat_w // p
        self.n_img = self.hp * self.wp
        self.n_x = self.n_img + (-self.n_img) % SEQ_MULTI
        self.n_cap = n_cap
        self.n_c = n_cap + (-n_cap) % SEQ_MULTI
        S = self.n_x + self.n_c
        b = self._alloc(S)
        pos = np.zeros((S, 3), np.int64)
        hh, ww = np.meshgrid(np.arange(self.hp), np.arange(self.wp), indexing="ij")
        pos[:self.n_img, 0] = self.n_c + 1
        pos[:self.n_img, 1] = hh.reshape(-1)
        pos[:self.n_img, 2] = ww.reshape(-1)
        pos[self.n_x:, 0] = 1 + np.arange(self.n_c)
        cos, sin = rope_tables(pos, self.cfg["axes_dims"], self.cfg["rope_theta"])
        b["cos"].numpy()[:S] = cos
        b["sin"].numpy()[:S] = sin
        self.S = S

    def encode_caption(self, feats: np.ndarray):
        """Caption features [T][cap_dim] (the text encoder's rows) -> the refined caption rows, kept on the GPU."""
        g, D = self.gpu, self.D
        b = self._bufs
        T = feats.shape[0]
        f = feats.astype(np.float32)
        f = f / np.sqrt((f * f).mean(-1, keepdims=True) + self.cfg["norm_eps"]) * self.cap_norm
        rows = np.empty((self.n_c, D), np.float32)
        rows[:T] = f @ self.cap_w.T + self.cap_b
        rows[T:] = self.cap_pad
        x = b["x"]
        x.numpy()[self.n_x:self.n_x + self.n_c] = rows
        pending = None
        for blk in self.cap_refiner:
            pending = self._block(b, blk, x, self.n_x, self.n_c, 0, pending)
        self._flush(b, x, self.n_x, self.n_c, pending)
        g.wait()
        b["cap"].numpy()[:self.n_c] = x.numpy()[self.n_x:self.n_x + self.n_c]

    def step(self, s: int, latents: np.ndarray) -> np.ndarray:
        """The transformer's output for latents [C][H][W] at step s -> [C][H][W] (velocity = minus this)."""
        g, D, b = self.gpu, self.D, self._bufs
        x = b["x"]
        lat = b["lat"].numpy()
        lat[:self.n_img] = patchify(latents.astype(np.float32), self.cfg["patch"])
        ops.linear_f32(g, b["lat"], self.x_embed_w, self.x_embed_b, x, M=self.n_img, N=D, K=self.patch_dim)
        g.wait()
        xn = x.numpy()
        xn[self.n_img:self.n_x] = self.x_pad
        xn[self.n_x:self.n_x + self.n_c] = b["cap"].numpy()[:self.n_c]
        pending = None
        for blk in self.x_refiner:
            pending = self._block(b, blk, x, 0, self.n_x, s, pending)
        self._flush(b, x, 0, self.n_x, pending)  # the main blocks run over more rows
        pending = None
        for blk in self.blocks:
            pending = self._block(b, blk, x, 0, self.S, s, pending)
            g.commit()
        self._flush(b, x, 0, self.S, pending)
        y = b["y"]
        fs = self.final_scale.view((D,), offset_elems=s * D)
        ops.layer_norm(g, x, fs, self.zeros, y, self.n_img, D, eps=1e-6)
        ops.linear_f32(g, y, self.final_w, self.final_b, b["vel"], M=self.n_img, N=self.patch_dim, K=D)
        g.wait()
        return unpatchify(b["vel"].numpy()[:self.n_img], self.cfg["in_channels"], 2 * self.hp, 2 * self.wp,
                          self.cfg["patch"])


def _ztile(w, M: int) -> dict:
    """Tiles for Z-Image's products, from an interleaved sweep of checked candidates on an M6 at 1024 x 1024 and
    2048 x 2048 (int8: 32 x 256 tiles for the products of 3,840 inputs and for the MLP down-projection, 37 and
    35 TOPS; 64 x 128 tiles swept four row tiles at a time for the MLP up-projection, 38-39 TOPS). No K split:
    these products store their result instead of accumulating it."""
    if w.fmt == "q8":
        if w.K < 7000 and w.N >= 16384:   # MLP up
            return {"tile": (64, 128, 4), "swz": 4}
        return {"tile": (32, 256, 4)}
    t = _tile(w, M)
    t.pop("ksplit", None)
    return t


# -- text encoder ------------------------------------------------------------------------------------------------

def convert_text_encoder(src: str, out: str, *, fmt: str = "h8", log=print):
    """Z-Image's Qwen3-4B: the transformer reads hidden_states[-2], the residual stream after the second-to-last
    decoder layer, so the last layer, the final norm and the head are left out (h3metal/te.py)."""
    from .te import convert_te
    d = os.path.join(src, "text_encoder")
    with open(os.path.join(d, "config.json")) as fh:
        n = json.load(fh)["num_hidden_layers"]
    convert_te(d, out, fmt=fmt, layers=n - 1, prefix="model", log=log)


def chat_prompt(prompt: str) -> str:
    """The text Z-Image's pipeline encodes: Qwen3's chat template with one user message, the assistant turn opened
    (thinking enabled, so no empty thinking block)."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


# -- pipeline ----------------------------------------------------------------------------------------------------

MAX_TOKENS = 512  # Z-Image's pipeline truncates the templated prompt here


def is_zimage_dir(model_dir: str) -> bool:
    """A converted Z-Image directory (its transformer file says so)."""
    p = os.path.join(os.path.expanduser(model_dir), "dit.h3m")
    if not os.path.exists(p):
        return False
    from .weights import read_header
    return read_header(p)["meta"].get("kind") == "zimage-dit"


def is_zimage_checkpoint(src: str) -> bool:
    """An original Z-Image checkpoint (diffusers layout)."""
    p = os.path.join(os.path.expanduser(src), "model_index.json")
    if not os.path.exists(p):
        return False
    with open(p) as fh:
        return json.load(fh).get("_class_name") == "ZImagePipeline"


def convert_zimage(src: str, out: str, *, dit: str = "q8", te: str = "q8", parts=None, log=print):
    """The whole model: tokenizer, text encoder, transformer, VAE decoder."""
    import shutil
    from .flux_vae import convert_flux_vae
    src, out = os.path.expanduser(src), os.path.expanduser(out)
    os.makedirs(out, exist_ok=True)
    parts = parts or ["tokenizer", "vae", "te", "dit"]
    fmts = dict(zip(("qkv", "out", "fc1", "fc2"), dit.split(","))) if "," in dit else \
        {k: dit for k in ("qkv", "out", "fc1", "fc2")}
    if "tokenizer" in parts:
        shutil.copy(os.path.join(src, "tokenizer", "tokenizer.json"), os.path.join(out, "tokenizer.json"))
    if "vae" in parts:
        convert_flux_vae(src, os.path.join(out, "vae.h3m"), log=log)
    if "te" in parts:
        convert_text_encoder(src, os.path.join(out, "te.h3m"), fmt=te, log=log)
    if "dit" in parts:
        convert_transformer(src, os.path.join(out, "dit.h3m"), fmts=fmts, log=log)


class ZImagePipeline:
    """Text to image with Z-Image-Turbo: eight steps, no guidance, the FLUX decoder."""
    model_name = "Z-Image-Turbo"
    default_size = (1024, 1024)

    def __init__(self, model_dir: str, gpu: Gpu | None = None, log=print):
        from tokenizers import Tokenizer
        model_dir = os.path.expanduser(model_dir)
        self.dir = model_dir
        self.gpu = gpu or Gpu()
        self.log = log
        self.paths = {k: os.path.join(model_dir, f"{k}.h3m") for k in ("te", "dit", "vae")}
        for k, p in self.paths.items():
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing {p}: convert a Z-Image checkpoint with `h3metal convert` first")
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self._te_cache: dict[str, np.ndarray] = {}
        self._dit = self._dit_wf = None

    def tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(chat_prompt(prompt), add_special_tokens=False).ids[:MAX_TOKENS]

    def encode_prompt(self, prompt: str, timing) -> np.ndarray:
        from .pipeline import _store
        from .te import TextEncoder
        from .weights import bf16_bits, bf16_to_f32
        if prompt in self._te_cache:
            return self._te_cache[prompt]
        path = self._cache_path(prompt)
        if path and os.path.exists(path):
            emb = bf16_to_f32(np.load(path))
            timing.add("text (cached)", 0.0)
        else:
            self.release()
            t = time.time()
            wf = WeightFile(self.gpu, self.paths["te"])
            emb = TextEncoder(self.gpu, wf).encode(self.tokenize(prompt))
            wf.close()
            timing.add("text", time.time() - t)
            if path:
                _store(path, bf16_bits(emb))
        self._te_cache = {prompt: emb}
        return emb

    def _cache_path(self, prompt: str) -> str | None:
        root = os.environ.get("H3METAL_CACHE", os.path.expanduser("~/.h3metal/cache"))
        if root in ("", "0", "off"):
            return None
        import hashlib
        st = os.stat(self.paths["te"])
        key = f"zimage|{prompt}|{st.st_size}|{int(st.st_mtime)}"
        return os.path.join(root, hashlib.sha256(key.encode()).hexdigest()[:32] + ".npy")

    def release(self):
        """Unmaps the transformer (before the text encoder needs the memory)."""
        if self._dit is not None:
            self._dit.free()
            self._dit = None
            self._dit_wf.close()
            self._dit_wf = None

    def generate(self, prompt: str, width: int = 1024, height: int = 1024, seed: int = 0, progress=None, **_):
        """-> (uint8 image [height][width][3], Timing)."""
        from .flux_vae import FluxVAEDecoder
        from .pipeline import Timing, noise
        if width % 16 or height % 16:
            raise ValueError("Z-Image needs width and height that are multiples of 16")
        timing = Timing()
        emb = self.encode_prompt(prompt, timing)
        t = time.time()
        if self._dit is None:
            self._dit_wf = WeightFile(self.gpu, self.paths["dit"])
            self._dit = ZImageDiT(self.gpu, self._dit_wf)
            timing.add("load", time.time() - t)
        dit = self._dit
        h, w = height // 8, width // 8
        t = time.time()
        dit.prepare(emb.shape[0], h, w)
        dit.encode_caption(emb)
        timing.add("text rows", time.time() - t)
        x = noise(seed, (1, dit.cfg["in_channels"], h, w)).reshape(-1, h, w).astype(np.float32)
        sig = dit.sigmas
        for s in range(dit.n_steps):
            t = time.time()
            o = dit.step(s, x)
            x = (x + np.float32(sig[s] - sig[s + 1]) * o).astype(np.float32)
            timing.add("denoise", time.time() - t)
            if progress:
                progress(s + 1, dit.n_steps)
        t = time.time()
        dit.free()  # its activations (about 2 GB at 2048 x 2048) are not needed while decoding; the weights stay mapped
        wf = WeightFile(self.gpu, self.paths["vae"])
        img = FluxVAEDecoder(self.gpu, wf).decode(x)
        wf.close()
        timing.add("decode", time.time() - t)
        out = np.clip(np.rint((img / 2 + 0.5).clip(0, 1) * 255), 0, 255).astype(np.uint8)
        return out, timing
