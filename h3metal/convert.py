"""Converts the official MiniMax-H3-Image weights into h3metal's .h3m files.

The transformer (DiT) conversion does three things beyond quantization:

* The AdaLN modulation of every block depends only on the timestep and the
  row's modality, and the sampler runs a fixed grid of timesteps, so every
  modulation vector the model can ever use is computed here, once (each step
  x 2 modalities + the reference rows' fixed timestep). The 13.4 B parameters
  of the AdaLN projections never reach the device. The checkpoint is distilled
  to a six-step grid with one output head per step; `flow=` converts instead
  for MiniMax-H3's plain flow sampler, which few-step adapters such as pDMD's
  2-step LoRA are made for (see convert_dit).
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


def flow_sigmas(n: int, shift: float) -> np.ndarray:
    """The sigma grid of MiniMax-H3's flow sampler (diffusers' MiniMaxH3Scheduler) for n steps: torch.linspace(1, 0,
    n + 1) in float32, through the shift s*x / (1 + (s - 1)*x), consecutive duplicates removed. Computed in float32
    in torch's order (its linspace fills the second half from the end), so the timesteps match bit for bit."""
    f = np.float32
    count = n + 1
    step = f(f(0.0) - f(1.0)) / f(count - 1)
    half = count // 2
    base = np.array([f(1.0) + step * f(i) if i < half else f(0.0) - step * f(count - i - 1) for i in range(count)],
                    np.float32)
    sig = ((f(shift) * base) / (f(1.0) + f(shift - 1.0) * base)).astype(np.float32)
    return sig[np.concatenate([[True], sig[1:] != sig[:-1]])]


# LoRA modules (diffusers' MiniMax-H3 names, as the pDMD release has them, or the checkpoint's own) -> the
# checkpoint tensor they change and how their rows sit in it.
_RENAMES = (("token_refiner.refiner_blocks.", "token_refiner.blocks."), ("transformer_blocks.", "blocks."),
            ("time_embedder.linear_1", "time_embedder.proj_in"), ("time_embedder.linear_2", "time_embedder.proj_out"),
            ("norm_out.linear", "final_layer.adaln_proj.linear"), ("context_embedder", "condition_proj"))
_WHOLE = {"proj_in": "video_patch_proj", "proj_out": "final_layer.video_out"}


def _lora_target(module: str):
    """-> (checkpoint key, how) with how None (the whole matrix), "q"/"k"/"v" (that third of the per-head
    interleaved fused QKV) or "swap" (diffusers' [value; gate] SwiGLU rows over the checkpoint's [gate; value])."""
    if module in _WHOLE:
        return _WHOLE[module] + ".weight", None
    for a, b in _RENAMES:
        if module.startswith(a):
            module = b + module[len(a):]
    for suffix, key, how in ((".attn.to_q", ".attn.qkv_proj", "q"), (".attn.to_k", ".attn.qkv_proj", "k"),
                             (".attn.to_v", ".attn.qkv_proj", "v"), (".attn.to_out.0", ".attn.out_proj", None),
                             (".ff.net.0.proj", ".mlp.fc1", "swap"), (".ff.net.2", ".mlp.fc2", None)):
        if module.endswith(suffix):
            return module[:-len(suffix)] + key + ".weight", how
    return module + ".weight", None


def lora_deltas(adapters, cfg: dict, keys) -> dict:
    """adapters: [(path, strength)]. -> {checkpoint key: [add(w) adding that adapter's delta to the float32 matrix
    in place]}, every module checked against the checkpoint's keys and shapes before any work is done."""
    from .lora import LoraError, read
    H, hd = cfg["num_attention_heads"], cfg["attention_head_dim"]
    F = cfg["ffn_hidden_size"]
    keys = set(keys)
    out: dict = {}
    for path, strength in adapters:
        unknown = []
        for module, layer in read(path).items():
            key, how = _lora_target(module)
            if key not in keys:
                unknown.append(module)
                continue
            scale = np.float32(layer.scale * strength)
            A, B = layer.A.astype(np.float32), layer.B.astype(np.float32)

            def add(w, A=A, B=B, scale=scale, how=how, module=module):
                d = (B * scale) @ A
                if how in ("q", "k", "v"):
                    j = "qkv".index(how)
                    if d.shape != (H * hd, w.shape[1]):
                        raise LoraError(f"{module}: delta {d.shape} does not fit one third of {w.shape}")
                    w.reshape(H, 3, hd, w.shape[1])[:, j] += d.reshape(H, hd, w.shape[1])
                elif how == "swap":
                    if d.shape != w.shape or w.shape[0] != 2 * F:
                        raise LoraError(f"{module}: delta {d.shape} does not fit {w.shape}")
                    w[:F] += d[F:]
                    w[F:] += d[:F]
                else:
                    if d.shape != w.shape:
                        raise LoraError(f"{module}: delta {d.shape} does not fit {w.shape}")
                    w += d
            out.setdefault(key, []).append(add)
        if unknown:
            raise LoraError(f"{os.path.basename(path)}: {len(unknown)} modules are not in MiniMax-H3-Image's "
                            f"transformer, e.g. {', '.join(sorted(unknown)[:3])}")
    return out


class _Merged:
    """The checkpoint's tensors with LoRA deltas added, in float32. Remembers which keys it changed."""

    def __init__(self, st, deltas: dict):
        self.st, self.deltas, self.done = st, deltas, set()

    def get(self, key: str) -> np.ndarray:
        w = self.st.get(key)
        if key in self.deltas:
            w = np.array(w, dtype=np.float32)
            for add in self.deltas[key]:
                add(w)
            self.done.add(key)
        return w


def _signs(seed: int, layer: int, role: int, n: int) -> np.ndarray:
    return random_signs(n, seed * 1_000_003 + layer * 101 + role)


def convert_dit(src: str, out, *, fmts=None, refiner_fmt: str = "q8", seed: int = 1, layers: int | None = None,
                log=print, flow: dict | None = None, lora=None):
    """src: the MiniMax-H3-Image directory (transformer/, pdd_*). fmts: format per matrix kind.
    out may also be a dict {path: fmts} to write several formats in one pass over the checkpoint.

    flow: {"shift": s, "steps": [n, ...]} converts for MiniMax-H3's plain flow sampler instead of the checkpoint's
    six-step grid: the transformer's own output head rather than the six per-step heads, the grids of
    flow_sigmas(n, s) for every n (the first is the default), and the Euler update in its x_t / x0 blend form, as
    diffusers' MiniMaxH3Scheduler runs it. lora: [(path, strength)], merged into the checkpoint's matrices in
    float32 before they are quantized."""
    outs = out if isinstance(out, dict) else {out: fmts}
    outs = {p: {"qkv": "q8", "out": "q8", "fc1": "q8", "fc2": "q8", **(f or {})} for p, f in outs.items()}
    with open(os.path.join(src, "transformer", "config.json")) as f:
        cfg = json.load(f)
    raw = Sharded(os.path.join(src, "transformer"))
    deltas = lora_deltas(lora, cfg, raw.keys()) if lora else {}
    st = _Merged(raw, deltas)

    D, H, hd, F = cfg["hidden_size"], cfg["num_attention_heads"], cfg["attention_head_dim"], cfg["ffn_hidden_size"]
    n_layers = cfg["num_layers"] if layers is None else layers
    inner = H * hd
    if flow is None:
        with open(os.path.join(src, "pdd_config.json")) as f:
            pdd = json.load(f)
        t_steps, sigmas = schedule(pdd["sigmas"])
        schedules = {str(len(t_steps)): {"sigmas": sigmas.tolist(), "first": 0}}
        heads = SafeTensors(os.path.join(src, "pdd_fused_heads.safetensors"))
        hw, hb = heads.get("video_out.weight"), heads.get("video_out.bias")
        if "final_layer.video_out.weight" in deltas:
            from .lora import LoraError
            raise LoraError("the six-step sampler uses its own per-step heads, which a LoRA of proj_out cannot change")
        sampler = {"sampler": "pdd", "update": "velocity", "default_steps": len(t_steps)}
    else:
        t_all, schedules = [], {}
        for n in flow["steps"]:
            sig = flow_sigmas(n, flow["shift"])
            schedules[str(n)] = {"sigmas": sig.tolist(), "first": len(t_all)}
            t_all += list((np.float32(1) - sig[:-1]).astype(np.float32))
        t_steps = np.array(t_all, np.float32)
        hw0 = np.asarray(st.get("final_layer.video_out.weight"), np.float32)
        hb0 = np.asarray(st.get("final_layer.video_out.bias"), np.float32)
        hw, hb = np.repeat(hw0[None], len(t_steps), 0), np.repeat(hb0[None], len(t_steps), 0)
        sigmas = np.array(schedules[str(flow["steps"][0])]["sigmas"], np.float32)
        sampler = {"sampler": "flow", "update": "blend", "default_steps": int(flow["steps"][0]),
                   "shift": float(flow["shift"])}
    n_steps = len(t_steps)  # table entries: every step of every grid
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
        "refiner_fmt": refiner_fmt, "seed": seed, "schedules": schedules, **sampler,
        "lora": [{"file": os.path.basename(p), "strength": float(k)} for p, k in (lora or [])],
    }
    writers = [(Writer(p, dict(meta, fmts=f)), f) for p, f in outs.items()]

    class _All:
        """Writes every tensor to all the outputs."""
        def tensor(self, *a, **k):
            for wr, _ in writers:
                wr.tensor(*a, **k)

        def linear(self, *a, **k):
            for wr, _ in writers:
                wr.linear(*a, **k)

    w = _All()
    t_start = time.time()

    # Patch embedding, text projection, per-step heads.
    w.tensor("x_embed.w", st.get("video_patch_proj.weight").astype(np.float32), "f32")
    w.tensor("x_embed.b", st.get("video_patch_proj.bias").astype(np.float32), "f32")
    cw = st.get("condition_proj.weight")
    w.linear("ctx_embed", "bf16", cw.shape[0], cw.shape[1], bf16_bits(cw), None, bias=st.get("condition_proj.bias"))
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

    def put(name: str, kind: str, mat: np.ndarray, refiner: bool):
        cache = {}
        for wr, f in writers:
            fmt = refiner_fmt if refiner else f[kind]
            if fmt not in cache:
                cache[fmt] = quantize(fmt, mat)
            codes, scales = cache[fmt]
            wr.linear(name, fmt, mat.shape[0], mat.shape[1], codes, scales)

    def block(prefix: str, name: str, layer: int, refiner: bool):
        """One transformer block's matrices; returns the input sign vectors folded into the norms."""
        s_in, s_o = _signs(seed, layer, 1, D), _signs(seed, layer, 2, inner)
        s_mlp, s_f = _signs(seed, layer, 3, D), _signs(seed, layer, 4, F)
        qkv = st.get(f"{prefix}.attn.qkv_proj.weight")  # per-head interleaved [h: q k v]
        qkv = qkv.reshape(H, 3, hd, D).transpose(1, 0, 2, 3).reshape(3 * inner, D)
        qkv[2 * inner:] *= s_o[:, None]
        put(f"{name}.qkv", "qkv", rotate_in(qkv, s_in), refiner)
        del qkv
        o = st.get(f"{prefix}.attn.out_proj.weight")
        put(f"{name}.out", "out", rotate_in(o, s_o), refiner)
        del o
        fc1 = st.get(f"{prefix}.mlp.fc1.weight")  # [gate; value]
        fc1 = np.concatenate([fc1[F:] * s_f[:, None], fc1[:F]])  # -> [value * sign; gate]
        put(f"{name}.fc1", "fc1", rotate_in(fc1, s_mlp), refiner)
        del fc1
        fc2 = st.get(f"{prefix}.mlp.fc2.weight")
        put(f"{name}.fc2", "fc2", rotate_in(fc2, s_f), refiner)
        del fc2
        w.tensor(f"{name}.q_norm", st.get(f"{prefix}.attn.q_norm.weight").astype(np.float32), "f32")
        w.tensor(f"{name}.k_norm", st.get(f"{prefix}.attn.k_norm.weight").astype(np.float32), "f32")
        return s_in, s_mlp

    # Token refiner: plain pre-norm blocks over the text rows (no modulation, no rotary embedding).
    for i in range(cfg["token_refiner_num_layers"]):
        prefix = f"token_refiner.blocks.{i}"
        s_in, s_mlp = block(prefix, f"refiner.{i}", 1000 + i, True)
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
        s_in, s_mlp = block(prefix, f"blocks.{l}", l, False)
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
    if lora and layers is None and set(deltas) - st.done:
        from .lora import LoraError
        raise LoraError(f"LoRA modules never reached the converter: {sorted(set(deltas) - st.done)[:3]}")
    for wr, _ in writers:
        wr.close()
    for p in outs:
        log(f"wrote {p} ({os.path.getsize(p) / 1e9:.2f} GB) in {time.time() - t_start:.0f} s")


def _lin(w: Writer, name: str, fmt: str, mat: np.ndarray, bias=None, G: int = 128):
    codes, scales = quantize(fmt, mat, G)
    w.linear(name, fmt, mat.shape[0], mat.shape[1], codes, scales, bias=bias, G=G)
