"""Converts the official MiniMax-H3-Image weights into h3metal's .h3m files.

The transformer (DiT) conversion does three things beyond quantization:

* The AdaLN modulation of every block depends only on the timestep and the
  row's modality, and the checkpoint is distilled to a fixed six-step grid, so
  every modulation vector the model can ever use is computed here, once
  (6 steps x 2 modalities + the reference rows' fixed timestep). The 13.4 B
  parameters of the AdaLN projections never reach the device.
* Each block's norm weights are folded into those tables.
* Every quantized matrix is rotated on its input side (h3metal/quant.py), and
  the random sign flips of the rotations are folded into the tables and into
  the rows of the producing matrix wherever a sign can pass through.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from .quant import quantize, random_signs, rotate_in
from .st import SafeTensors, Sharded
from .weights import Writer, bf16_bits, bf16_to_f32

REF_T = 0.999  # the timestep reference rows of an edit are held at


def silu(x):
    return x / (1.0 + np.exp(-x))


def bf16_round(x: np.ndarray) -> np.ndarray:
    return bf16_to_f32(bf16_bits(x))


def timestep_embedding(t: np.ndarray, dim: int = 256, max_period: float = 10000.0) -> np.ndarray:
    """diffusers' get_timestep_embedding with flip_sin_to_cos=True, downscale_freq_shift=0, in float32."""
    half = dim // 2
    exponent = (-math.log(max_period) * np.arange(half, dtype=np.float32) / np.float32(half)).astype(np.float32)
    freqs = np.exp(exponent).astype(np.float32)
    emb = (t.astype(np.float32)[:, None] * freqs[None, :]).astype(np.float32)
    return np.concatenate([np.cos(emb), np.sin(emb)], axis=-1).astype(np.float32)


def schedule(sigmas: list[float]):
    """The timesteps and sigmas the released sampler actually uses, float32 round trips included."""
    t0 = np.array([1.0 - s for s in sigmas[:-1]], dtype=np.float32)
    sig = np.concatenate([(np.float32(1) - t0).astype(np.float32), np.zeros(1, np.float32)])
    t = (np.float32(1) - sig[:-1]).astype(np.float32)
    return t, sig


def _signs(seed: int, layer: int, role: int, n: int) -> np.ndarray:
    return random_signs(n, seed * 1_000_003 + layer * 101 + role)


def convert_dit(src: str, out: str, *, fmts=None, refiner_fmt: str = "q8", seed: int = 1, layers: int | None = None,
                log=print):
    """src: the MiniMax-H3-Image directory (transformer/, pdd_*). fmts: format per matrix kind."""
    fmts = {"qkv": "q8", "out": "q8", "fc1": "q8", "fc2": "q8", **(fmts or {})}
    with open(os.path.join(src, "transformer", "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(src, "pdd_config.json")) as f:
        pdd = json.load(f)
    st = Sharded(os.path.join(src, "transformer"))
    heads = SafeTensors(os.path.join(src, "pdd_fused_heads.safetensors"))

    D, H, hd, F = cfg["hidden_size"], cfg["num_attention_heads"], cfg["attention_head_dim"], cfg["ffn_hidden_size"]
    n_layers = cfg["num_layers"] if layers is None else layers
    inner = H * hd
    t_steps, sigmas = schedule(pdd["sigmas"])
    n_steps = len(t_steps)
    ts = np.concatenate([t_steps, np.array([REF_T], np.float32)])  # the last one is the reference rows'

    # Timestep embedding, float32 like the checkpoint's time embedder.
    w1, b1 = st.get("time_embedder.proj_in.weight"), st.get("time_embedder.proj_in.bias")
    w2, b2 = st.get("time_embedder.proj_out.weight"), st.get("time_embedder.proj_out.bias")
    temb = timestep_embedding(ts, cfg["timestep_input_dim"]) @ w1.T + b1
    temb = (silu(temb) @ w2.T + b2).astype(np.float32)
    # Every AdaLN projection activates at float32 and casts to bfloat16 before its bfloat16 product.
    act = bf16_round(silu(temb))

    meta = {
        "kind": "dit",
        "config": {"hidden": D, "layers": n_layers, "heads": H, "head_dim": hd, "ffn": F,
                   "refiner_layers": cfg["token_refiner_num_layers"], "text_dim": cfg["text_dim"],
                   "latents": cfg["latents_dim"], "patch": cfg["patch_size"], "rope_freq": cfg["rope_inv_freq_len"],
                   "rope_theta": 10000.0, "norm_eps": cfg["norm_eps"], "qk_eps": cfg["qk_norm_eps"],
                   "final_eps": cfg["final_norm_eps"]},
        "timesteps": t_steps.tolist(), "sigmas": sigmas.tolist(), "ref_t": REF_T,
        "fmts": fmts, "refiner_fmt": refiner_fmt, "seed": seed,
    }
    w = Writer(out, meta)
    t_start = time.time()

    # Patch embedding, text projection, per-step heads.
    w.tensor("x_embed.w", st.get("video_patch_proj.weight").astype(np.float32), "f32")
    w.tensor("x_embed.b", st.get("video_patch_proj.bias").astype(np.float32), "f32")
    cw = st.get("condition_proj.weight")
    w.linear("ctx_embed", "bf16", cw.shape[0], cw.shape[1], bf16_bits(cw), None, bias=st.get("condition_proj.bias"))
    hw, hb = heads.get("video_out.weight"), heads.get("video_out.bias")
    w.tensor("heads.w", np.ascontiguousarray(hw, np.float32), "f32")
    w.tensor("heads.b", np.ascontiguousarray(hb, np.float32), "f32")

    # Final norm modulation: [steps][3 row types][2][D] (scale' = norm weight * (1 + scale), shift).
    fw, fb = st.get("final_layer.adaln_proj.linear.weight"), st.get("final_layer.adaln_proj.linear.bias")
    fmod = bf16_round(act @ fw.T + fb)  # [n_t][2D]: shift, scale
    shift, scale = fmod[:, :D], fmod[:, D:]
    nw = st.get("final_layer.norm.weight")
    final = np.zeros((n_steps, 3, 2, D), np.float32)
    for s in range(n_steps):
        for ty, ti in ((0, s), (1, s), (2, n_steps)):
            final[s, ty, 0] = nw * (1.0 + scale[ti])
            final[s, ty, 1] = shift[ti]
    w.tensor("final_mod", final, "f32")

    def block(prefix: str, name: str, layer: int, fmt_of, tables=None):
        """One transformer block's matrices; returns the input sign vectors folded into the norms."""
        s_in, s_o = _signs(seed, layer, 1, D), _signs(seed, layer, 2, inner)
        s_mlp, s_f = _signs(seed, layer, 3, D), _signs(seed, layer, 4, F)
        qkv = st.get(f"{prefix}.attn.qkv_proj.weight")  # per-head interleaved [h: q k v]
        qkv = qkv.reshape(H, 3, hd, D).transpose(1, 0, 2, 3).reshape(3 * inner, D)
        qkv[2 * inner:] *= s_o[:, None]
        _lin(w, f"{name}.qkv", fmt_of("qkv"), rotate_in(qkv, s_in))
        del qkv
        o = st.get(f"{prefix}.attn.out_proj.weight")
        _lin(w, f"{name}.out", fmt_of("out"), rotate_in(o, s_o))
        del o
        fc1 = st.get(f"{prefix}.mlp.fc1.weight")  # [gate; value]
        fc1 = np.concatenate([fc1[F:] * s_f[:, None], fc1[:F]])  # -> [value * sign; gate]
        _lin(w, f"{name}.fc1", fmt_of("fc1"), rotate_in(fc1, s_mlp))
        del fc1
        fc2 = st.get(f"{prefix}.mlp.fc2.weight")
        _lin(w, f"{name}.fc2", fmt_of("fc2"), rotate_in(fc2, s_f))
        del fc2
        w.tensor(f"{name}.q_norm", st.get(f"{prefix}.attn.q_norm.weight").astype(np.float32), "f32")
        w.tensor(f"{name}.k_norm", st.get(f"{prefix}.attn.k_norm.weight").astype(np.float32), "f32")
        return s_in, s_mlp

    # Token refiner: plain pre-norm blocks over the text rows (no modulation, no rotary embedding).
    for i in range(cfg["token_refiner_num_layers"]):
        prefix = f"token_refiner.blocks.{i}"
        s_in, s_mlp = block(prefix, f"refiner.{i}", 1000 + i, lambda kind: refiner_fmt)
        m1 = np.stack([st.get(f"{prefix}.norm1.weight") * s_in, np.zeros(D, np.float32)])[None]
        m2 = np.stack([st.get(f"{prefix}.norm2.weight") * s_mlp, np.zeros(D, np.float32)])[None]
        w.tensor(f"refiner.{i}.mod1", m1.astype(np.float32), "f32")
        w.tensor(f"refiner.{i}.mod2", m2.astype(np.float32), "f32")
    w.tensor("refiner.final_norm", st.get("token_refiner.final_norm.weight").astype(np.float32), "f32")
    w.tensor("ones", np.ones((1, D), np.float32), "f32")

    # Main blocks with their precomputed modulation.
    mod_attn = np.zeros((n_steps, n_layers, 3, 2, D), np.float32)
    mod_mlp = np.zeros((n_steps, n_layers, 3, 2, D), np.float32)
    gate_attn = np.zeros((n_steps, n_layers, 3, D), np.float32)
    gate_mlp = np.zeros((n_steps, n_layers, 3, D), np.float32)
    for l in range(n_layers):
        prefix = f"blocks.{l}"
        s_in, s_mlp = block(prefix, f"blocks.{l}", l, lambda kind: fmts[kind])
        aw = st.get(f"{prefix}.adaln_proj.linear.weight")
        ab = st.get(f"{prefix}.adaln_proj.linear.bias")
        mod = bf16_round(act @ aw.T + ab).reshape(len(ts), 3, 6, D)  # [t][modality][6][D]
        del aw
        n1, n2 = st.get(f"{prefix}.norm1.weight"), st.get(f"{prefix}.norm2.weight")
        for s in range(n_steps):
            for ty, (ti, mo) in enumerate(((s, 0), (s, 1), (n_steps, 0))):
                sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = mod[ti, mo]
                mod_attn[s, l, ty, 0] = n1 * (1.0 + sc_msa) * s_in
                mod_attn[s, l, ty, 1] = sh_msa * s_in
                mod_mlp[s, l, ty, 0] = n2 * (1.0 + sc_mlp) * s_mlp
                mod_mlp[s, l, ty, 1] = sh_mlp * s_mlp
                gate_attn[s, l, ty] = g_msa
                gate_mlp[s, l, ty] = g_mlp
        log(f"  block {l + 1}/{n_layers} done ({time.time() - t_start:.0f} s)")
    w.tensor("mod_attn", mod_attn, "f32")
    w.tensor("mod_mlp", mod_mlp, "f32")
    w.tensor("gate_attn", gate_attn, "f32")
    w.tensor("gate_mlp", gate_mlp, "f32")
    w.close()
    log(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB) in {time.time() - t_start:.0f} s")


def _lin(w: Writer, name: str, fmt: str, mat: np.ndarray, bias=None, G: int = 128):
    codes, scales = quantize(fmt, mat, G)
    w.linear(name, fmt, mat.shape[0], mat.shape[1], codes, scales, bias=bias, G=G)
