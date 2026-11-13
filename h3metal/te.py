"""Text encoders of the Qwen3 family, run up to the layer a diffusion model reads.

MiniMax-H3-Image conditions on hidden_states[50] of its Qwen3-VL
conditioner, the residual stream after decoder layer 50, so the remaining
layers, the final norm and the language-model head are never needed. Its
prompts are tokenized verbatim (no chat template, no special tokens). For an
edit, every reference image contributes a label and a vision block whose
rows hold the vision tower's features; the three DeepStack feature sets are
added to those rows after decoder layers 1, 2 and 3.

Z-Image reads hidden_states[-2] of a plain Qwen3-4B (h3metal/zimage.py):
the same decoder with one-dimensional positions, which Qwen3-VL's
multimodal rotary embedding reduces to when its three axes agree.
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import ops
from .gpu import Gpu
from .quant import quantize, random_signs, rotate_in
from .st import Sharded
from .weights import WeightFile, Writer, bf16_bits, bf16_to_f32

N_LAYERS = 50  # MiniMax-H3 reads hidden_states[50]


def convert_te(src: str, out: str, *, fmt: str = "h8", layers: int = N_LAYERS, seed: int = 2, log=print,
               prefix: str = "model.language_model"):
    """src: directory with the Qwen3-VL text encoder (config.json + safetensors shards)."""
    with open(os.path.join(src, "config.json")) as f:
        full = json.load(f)
    tc = full.get("text_config", full)
    st = Sharded(src)
    D, Hq, Hkv, hd, F = tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"], \
        tc["intermediate_size"]
    rope = tc.get("rope_scaling") or tc.get("rope_parameters") or {}
    meta = {"kind": "te", "fmt": fmt, "seed": seed,
            "config": {"hidden": D, "layers": layers, "heads": Hq, "kv_heads": Hkv, "head_dim": hd, "ffn": F,
                       "eps": tc["rms_norm_eps"], "rope_theta": tc.get("rope_theta", rope.get("rope_theta")),
                       "mrope_section": rope.get("mrope_section", [24, 20, 20]),
                       "vocab": tc["vocab_size"], "image_token_id": full.get("image_token_id", 151655)}}
    w = Writer(out, meta)
    name = f"{prefix}.embed_tokens.weight"
    emb = st.raw(name) if st.dtype(name) == "BF16" else bf16_bits(st.get(name))  # stored as bfloat16 bits
    w.tensor("embed", np.ascontiguousarray(emb), "bf16", host_only=True)
    rep = Hq // Hkv
    for l in range(layers):
        p = f"{prefix}.layers.{l}"
        s_in = random_signs(D, seed * 7919 + l * 4 + 0)
        s_v = random_signs(Hkv * hd, seed * 7919 + l * 4 + 1)
        s_mlp = random_signs(D, seed * 7919 + l * 4 + 2)
        s_f = random_signs(F, seed * 7919 + l * 4 + 3)
        s_o = np.tile(s_v.reshape(Hkv, 1, hd), (1, rep, 1)).reshape(Hq * hd)  # query head h reads kv head h // rep
        q = st.get(f"{p}.self_attn.q_proj.weight")
        k = st.get(f"{p}.self_attn.k_proj.weight")
        v = st.get(f"{p}.self_attn.v_proj.weight") * s_v[:, None]
        _lin(w, f"layers.{l}.qkv", fmt, rotate_in(np.concatenate([q, k, v]), s_in))
        del q, k, v
        _lin(w, f"layers.{l}.out", fmt, rotate_in(st.get(f"{p}.self_attn.o_proj.weight"), s_o))
        up = st.get(f"{p}.mlp.up_proj.weight") * s_f[:, None]
        gate = st.get(f"{p}.mlp.gate_proj.weight")
        _lin(w, f"layers.{l}.fc1", fmt, rotate_in(np.concatenate([up, gate]), s_mlp))
        del up, gate
        _lin(w, f"layers.{l}.fc2", fmt, rotate_in(st.get(f"{p}.mlp.down_proj.weight"), s_f))
        n1 = st.get(f"{p}.input_layernorm.weight")
        n2 = st.get(f"{p}.post_attention_layernorm.weight")
        w.tensor(f"layers.{l}.mod1", np.stack([n1 * s_in, np.zeros(D, np.float32)])[None].astype(np.float32), "f32")
        w.tensor(f"layers.{l}.mod2", np.stack([n2 * s_mlp, np.zeros(D, np.float32)])[None].astype(np.float32), "f32")
        w.tensor(f"layers.{l}.q_norm", st.get(f"{p}.self_attn.q_norm.weight").astype(np.float32), "f32")
        w.tensor(f"layers.{l}.k_norm", st.get(f"{p}.self_attn.k_norm.weight").astype(np.float32), "f32")
        log(f"  text encoder layer {l + 1}/{layers}")
    w.tensor("ones", np.ones((1, D), np.float32), "f32")
    w.close()
    log(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB)")


def convert_vision(src: str, out: str, log=print, prefix: str = "model.visual."):
    """The Qwen3-VL vision tower (edits only), stored in float16 under its checkpoint names."""
    with open(os.path.join(src, "config.json")) as f:
        vcfg = json.load(f)["vision_config"]
    st = Sharded(src)
    w = Writer(out, {"kind": "vision", "config": vcfg})
    for k in sorted(st.keys()):
        if k.startswith(prefix):
            w.tensor(k[len(prefix):], np.ascontiguousarray(st.get(k), np.float16), "f16")
    w.close()
    log(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB)")


def _lin(w: Writer, name: str, fmt: str, mat: np.ndarray):
    codes, scales = quantize(fmt, mat)
    w.linear(name, fmt, mat.shape[0], mat.shape[1], codes, scales)


def mrope_angles(pos3: np.ndarray, hd: int, theta: float, section) -> np.ndarray:
    """Qwen3-VL interleaved multimodal rotary angles: pos3 [3][T] (t, h, w) -> [T][hd/2] float32."""
    inv = (1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd))).astype(np.float32)
    freqs = (pos3.astype(np.float32)[:, :, None] * inv[None, None, :]).astype(np.float32)  # [3][T][hd/2]
    out = freqs[0].copy()
    for dim, offset in ((1, 1), (2, 2)):
        length = section[dim] * 3
        out[:, offset:length:3] = freqs[dim][:, offset:length:3]
    return out


# Stored integer weights run with either int8 or half-precision activations: the codes are the same.
HALF_ACT = {"q8": "h8", "q4": "h4", "q4g": "h4g", "h8": "h8", "h4": "h4", "h4g": "h4g"}
INT8_ACT = {"h8": "q8", "h4": "q4", "h4g": "q4g", "q8": "q8", "q4": "q4", "q4g": "q4g"}


class TextEncoder:
    def __init__(self, gpu: Gpu, wf: WeightFile, act: str | None = None):
        """act: "f16" (half-precision activations, the default) or "i8" (rotated int8 activations)."""
        self.gpu, self.wf = gpu, wf
        c = wf.meta["config"]
        self.cfg = c
        self.D, self.Hq, self.Hkv, self.hd, self.F = c["hidden"], c["heads"], c["kv_heads"], c["head_dim"], c["ffn"]
        self.L = c["layers"]
        self.embed = wf.host("embed")
        stored = wf.meta["fmt"]
        act = act or ("i8" if stored.startswith("q") and os.environ.get("H3METAL_TE_ACT") == "i8" else "f16")
        if stored in ("f16", "bf16"):
            act = "f16"
        self.fmt = (HALF_ACT if act == "f16" else INT8_ACT).get(stored, stored)
        lin = wf.linear
        self.layers = []
        for l in range(self.L):
            ly = {k: lin(f"layers.{l}.{k}") for k in ("qkv", "out", "fc1", "fc2")}
            for w in ly.values():
                w.fmt = self.fmt
            ly |= {k: wf.tensor(f"layers.{l}.{k}") for k in ("mod1", "mod2", "q_norm", "k_norm")}
            self.layers.append(ly)
        self.ones = wf.tensor("ones")
        self.half_act = act == "f16"

    def encode(self, ids, pos3: np.ndarray | None = None, vision: np.ndarray | None = None,
               vision_rows: np.ndarray | None = None, deepstack: list | None = None, profile=None,
               taps: tuple | None = None, n_valid: int | None = None) -> np.ndarray:
        """Token ids -> the hidden states after the stored layers as float32 [T][D] (rounded to bfloat16 like
        the reference).

        pos3: [3][T] multimodal positions (default: text positions 0..T-1 on all three axes).
        vision: [n][D] features replacing the rows `vision_rows`; deepstack: three [n][D] arrays added to
        those rows after the first three layers.
        taps: hidden_states indices k (the residual stream after k layers) to return instead, concatenated
        per row [T][len(taps) D] (FLUX.2 reads three); the layers after the last tap are skipped.
        n_valid: the first n_valid rows are the prompt and the rest padding (FLUX.2 pads to 512 tokens and
        masks the padding as keys): a padding row attends to the n_valid prompt rows only, itself excluded,
        as with the reference's padding mask.
        """
        g, D, Hq, Hkv, hd, F = self.gpu, self.D, self.Hq, self.Hkv, self.hd, self.F
        ids = np.asarray(ids, dtype=np.int64)
        T = len(ids)
        Tp = ops.pad_rows(T, 128)
        x = g.empty((Tp, D), "f32")
        x.numpy()[:T] = bf16_to_f32(self.embed[ids])
        if vision is not None:
            x.numpy()[vision_rows] = vision
        if pos3 is None:
            pos3 = np.tile(np.arange(T), (3, 1))
        ang = mrope_angles(pos3, hd, self.cfg["rope_theta"], self.cfg["mrope_section"])
        cos, sin = g.upload(np.cos(ang).astype(np.float32)), g.upload(np.sin(ang).astype(np.float32))
        act = "f16" if self.half_act else "i8"
        a = g.empty((Tp, max(D, Hq * hd, F)), act)
        sa = g.empty((Tp,), "f32")
        wide = g.empty((Tp, max((Hq + 2 * Hkv) * hd, 2 * F)), "bf16")
        o = g.empty((Tp, Hq * hd), "bf16")
        Q = g.empty((Hq, Tp, hd), "f16")
        K = g.empty((Hkv, Tp, hd), "f16")
        V = g.empty((Hkv, Tp, hd), "f16")
        rt = g.empty((Tp,), "u8")
        eps = self.cfg["eps"]
        h = self.half_act
        tapped = []
        for l, ly in enumerate(self.layers):
            ops.norm_mod_q8(g, x, ly["mod1"], rt, a.view((Tp, D)), None if h else sa, T, D, eps=eps, out_hr=h)
            qkv = wide.view((Tp, (Hq + 2 * Hkv) * hd))
            ops.gemm(g, a.view((Tp, D)), ly["qkv"], qkv, T, sa=None if h else sa)
            ops.qk_norm_rope(g, qkv, ly["q_norm"], ly["k_norm"], cos, sin, Q, K, V, M=T, Hq=Hq, Hk=Hkv, Hv=Hkv,
                             hd=hd, out_rows=Tp, rot=hd, eps=eps)
            if n_valid is None or n_valid >= T:
                ops.attention(g, Q, K, V, o, Lq=T, L=T, H=Hq, KVH=Hkv, D=hd, q_rows=Tp, k_rows=Tp,
                              scale=hd ** -0.5, causal=True, wm=8, bk=32)
            else:
                ops.attention(g, Q, K, V, o, Lq=n_valid, L=n_valid, H=Hq, KVH=Hkv, D=hd, q_rows=Tp, k_rows=Tp,
                              scale=hd ** -0.5, causal=True, wm=8, bk=32)
                ops.attention(g, Q.view(Q.shape, offset_elems=n_valid * hd), K, V, o.row(n_valid, T - n_valid),
                              Lq=T - n_valid, L=n_valid, H=Hq, KVH=Hkv, D=hd, q_rows=Tp, k_rows=Tp,
                              scale=hd ** -0.5, causal=False, wm=8, bk=32)
            ops.rot_q8(g, o, a.view((Tp, Hq * hd)), None if h else sa, T, Hq * hd, out_hr=h)
            ops.gemm(g, a.view((Tp, Hq * hd)), ly["out"], x, T, sa=None if h else sa, epi=1, rtype=rt,
                     gate=self.ones)
            ops.norm_mod_q8(g, x, ly["mod2"], rt, a.view((Tp, D)), None if h else sa, T, D, eps=eps, out_hr=h)
            ff = wide.view((Tp, 2 * F))
            ops.gemm(g, a.view((Tp, D)), ly["fc1"], ff, T, sa=None if h else sa)
            ops.rot_q8(g, ff, a.view((Tp, F)), None if h else sa, T, F, swiglu=True, out_hr=h)
            ops.gemm(g, a.view((Tp, F)), ly["fc2"], x, T, sa=None if h else sa, epi=1, rtype=rt, gate=self.ones)
            if deepstack is not None and l < len(deepstack):
                g.wait()
                x.numpy()[vision_rows] += deepstack[l]
            else:
                g.commit()  # one command buffer per layer: only this layer's weights are pinned
            if profile is not None:
                profile(l)
            if taps and l + 1 in taps:
                g.wait()
                tapped.append(bf16_to_f32(bf16_bits(x.numpy()[:T])))
                if l + 1 == max(taps):
                    return np.concatenate(tapped, axis=1)
        if taps:
            raise ValueError(f"taps {taps}: only {len(self.layers)} layers are stored")
        g.wait()
        return bf16_to_f32(bf16_bits(x.numpy()[:T]))
