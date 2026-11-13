"""FLUX.2 [klein] (black-forest-labs/FLUX.2-klein-4B, Apache-2.0) on h3metal's engine: the transformer.

FLUX.2's transformer runs the caption's rows and the image's rows through double-stream blocks (each stream
with its own weights, one attention over both) and then through single-stream blocks (one set of weights; the
attention and the MLP read the same normalized input and their outputs share one projection, as in ViT-22B).
Every block normalizes with LayerNorm (no affine), modulates with (1 + scale, shift) and gates what it adds to
the residual stream. Unlike Z-Image's and MiniMax-H3's, the modulation is global: one set per stream for all
double-stream blocks and one for all single-stream blocks, from the timestep alone, which depends on the
image's size (the schedule's shift does). So it is computed per step, on the CPU, with the sign flips below.

The conversion (h3metal/zimage.py and h3metal/convert.py do the same for their models):

* Every block matrix is rotated on its input side and quantized (h3metal/quant.py). The rotations' random
  sign flips go into the modulation tables and into the rows of the matrix producing the input. A block's two
  streams share the sign flips of the attention values: the attention mixes the rows of both streams.
* FLUX.2's rotary embedding rotates adjacent channel pairs; permuting the query and key rows of every head
  turns that into the engine's half-split rotation and leaves every score unchanged.
* The MLPs' input matrices are ordered (value, gate) as the engine's SwiGLU reads them.
* A single-stream block's output matrix is split into its attention part and its MLP part. Their inputs are
  quantized separately, each row with its own scale, so they run as two products, both accumulating into the
  residual stream.

The caption always has 512 rows (the pipeline pads it), a multiple of the products' row tiles, so the rows of
the two streams can share buffers: caption rows first, then the image rows.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from . import lora, ops
from .convert import silu, timestep_embedding
from .gpu import Gpu, Tensor
from .lora import LinearPlan, Part
from .quant import quantize, random_signs
from .st import Sharded
from .weights import WeightFile, Writer
from .zimage import _pair_perm, _ztile, rope_tables

TXT_MULTIPLE = 128   # caption rows: a multiple of every product's row tile (FLUX.2 pads captions to 512)


def empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """diffusers' compute_empirical_mu for FLUX.2: the schedule's shift for this many image rows and steps."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def schedule(steps: int, image_seq_len: int):
    """klein's sampler: sigmas linspace(1, 1/steps, steps) shifted exponentially by mu (FlowMatchEulerDiscrete-
    Scheduler with dynamic shifting), float32 like the scheduler. -> (sigmas [steps + 1], the last 0)."""
    mu = empirical_mu(image_seq_len, steps)
    base = np.linspace(1.0, 1.0 / steps, steps)
    sig = (np.exp(mu) / (np.exp(mu) + (1.0 / base - 1.0))).astype(np.float32)
    return np.concatenate([sig, np.zeros(1, np.float32)])


def _ftile(w, M: int) -> dict:
    """Tiles for FLUX.2 [klein]'s int8 products, from an interleaved sweep on an M6 at 1024 x 1024 (4608 rows; the
    double-stream blocks' image and caption rows, 4096 and 512, run as separate products)."""
    if w.fmt != "q8":
        return _ztile(w, M)
    if w.K < 7000 and w.N == w.K:          # attention output: 34 TOPS
        return {"tile": (64, 128, 4), "swz": 4}
    if w.K < 7000 and w.N < 16384:         # query, key and value: 37-39 TOPS
        return {"tile": (64, 256, 8), "swz": 4 if M >= 2048 else 1}
    if w.K < 7000:                         # MLP up, fused projections: 36-38 TOPS
        return {"tile": (64, 128, 4), "swz": 4}
    if M >= 2048:                          # MLP down, accumulating into the residual stream: 34 TOPS
        return {"tile": (32, 256, 4), "ksplit": 2}
    return {"tile": (64, 128, 4), "swz": 2}


def _signs(seed: int, layer: int, role: int, n: int) -> np.ndarray:
    return random_signs(n, seed * 1_000_003 + layer * 101 + role)


# Queries and keys rotated before their int8 rounding (ops.qk_norm_rope): against diffusers, one step's error goes
# from 4.4% to 3.9% and the image from the same embeddings from 38.5 to 39.0 dB (docs/flux2.md). H3METAL_QKROT=0
# turns it off.
QK_ROT = os.environ.get("H3METAL_QKROT", "1") != "0"

HOST = {"t_lin1": "time_guidance_embed.timestep_embedder.linear_1",
        "t_lin2": "time_guidance_embed.timestep_embedder.linear_2",
        "mod_img": "double_stream_modulation_img.linear", "mod_txt": "double_stream_modulation_txt.linear",
        "mod_single": "single_stream_modulation.linear", "mod_final": "norm_out.linear"}
HALF = {"x_embed": "x_embedder", "ctx_embed": "context_embedder", "proj_out": "proj_out"}


def model_plans(c: dict, seed: int) -> tuple:
    """Every converted linear layer (h3metal/lora.py): which rows of which original layers, in which order,
    with which signs, and the rotation of their inputs; the converter and LoRA adapters both use these. And the
    aliases: the original checkpoint's names (fused attention projections), which most FLUX.2 LoRAs use."""
    D, H, hd, F = c["dim"], c["heads"], c["head_dim"], c["ffn"]
    L2, L1, C = c["double_layers"], c["single_layers"], c["in_channels"]
    perm = _pair_perm(H, hd)
    rng = np.arange
    plans = {
        "t_lin1": LinearPlan("host", [Part(HOST["t_lin1"], D)], c["t_channels"], rotate=False),
        "t_lin2": LinearPlan("host", [Part(HOST["t_lin2"], D)], D, rotate=False),
        "mod_img": LinearPlan("host", [Part(HOST["mod_img"], 6 * D)], D, rotate=False),
        "mod_txt": LinearPlan("host", [Part(HOST["mod_txt"], 6 * D)], D, rotate=False),
        "mod_single": LinearPlan("host", [Part(HOST["mod_single"], 3 * D)], D, rotate=False),
        "mod_final": LinearPlan("host", [Part(HOST["mod_final"], 2 * D)], D, rotate=False),
        "x_embed": LinearPlan("f16", [Part(HALF["x_embed"], D)], C, rotate=False),
        "ctx_embed": LinearPlan("f16", [Part(HALF["ctx_embed"], D)], c["joint_dim"], rotate=False),
        "proj_out": LinearPlan("f16", [Part(HALF["proj_out"], C)], D, rotate=False),
    }
    aliases = {"img_in": [(HALF["x_embed"], 0)], "txt_in": [(HALF["ctx_embed"], 0)],
               "time_in.in_layer": [(HOST["t_lin1"], 0)], "time_in.out_layer": [(HOST["t_lin2"], 0)],
               "final_layer.linear": [(HALF["proj_out"], 0)],
               # the original final layer's modulation is (shift, scale); diffusers' norm_out (scale, shift)
               "final_layer.adaLN_modulation.1": [(HOST["mod_final"], np.r_[D:2 * D, 0:D])],
               "single_stream_modulation.lin": [(HOST["mod_single"], 0)],
               "double_stream_modulation_img.lin": [(HOST["mod_img"], 0)],
               "double_stream_modulation_txt.lin": [(HOST["mod_txt"], 0)]}
    for i in range(L2):
        a = f"transformer_blocks.{i}.attn"
        s_o = _signs(seed, i, 3, D)  # the attention values: shared by both streams
        for stream, (q, k, v, o), role, ff, orig in (
                ("img", ("to_q", "to_k", "to_v", "to_out.0"), 1, "ff", "img"),
                ("txt", ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"), 2, "ff_context", "txt")):
            name = f"double.{i}.{stream}"
            s_in = _signs(seed, i, role, D)
            s_m, s_f = _signs(seed, i, 4 + 2 * role, D), _signs(seed, i, 5 + 2 * role, F)
            li, lo = f"transformer_blocks.{i}.{ff}.linear_in", f"transformer_blocks.{i}.{ff}.linear_out"
            plans[f"{name}.qkv"] = LinearPlan("qkv", [Part(f"{a}.{q}", D, rows=perm), Part(f"{a}.{k}", D, rows=perm),
                                                      Part(f"{a}.{v}", D, sign=s_o)], D, s_in)
            plans[f"{name}.out"] = LinearPlan("out", [Part(f"{a}.{o}", D)], D, s_o)
            # linear_in [2F][D]: the SiLU'd half first, then the multiplied half
            plans[f"{name}.fc1"] = LinearPlan("fc1", [Part(li, F, rows=rng(F, 2 * F), sign=s_f),
                                                      Part(li, F, rows=rng(F))], D, s_m)
            plans[f"{name}.fc2"] = LinearPlan("fc2", [Part(lo, D)], F, s_f)
            b = f"double_blocks.{i}.{orig}"
            aliases[f"{b}_attn.qkv"] = [(f"{a}.{q}", 0), (f"{a}.{k}", D), (f"{a}.{v}", 2 * D)]
            aliases[f"{b}_attn.proj"] = [(f"{a}.{o}", 0)]
            aliases[f"{b}_mlp.0"] = [(li, 0)]
            aliases[f"{b}_mlp.2"] = [(lo, 0)]
    for j in range(L1):
        layer = 1000 + j
        a = f"single_transformer_blocks.{j}.attn"
        s_in, s_o, s_f = _signs(seed, layer, 1, D), _signs(seed, layer, 3, D), _signs(seed, layer, 6, F)
        fused = f"{a}.to_qkv_mlp_proj"  # [3D + 2F][D]: q, k, v, the SiLU'd half, the other half
        plans[f"single.{j}.qkv_mlp"] = LinearPlan("qkv", [
            Part(fused, D, rows=perm), Part(fused, D, rows=D + perm), Part(fused, D, rows=rng(2 * D, 3 * D), sign=s_o),
            Part(fused, F, rows=rng(3 * D + F, 3 * D + 2 * F), sign=s_f), Part(fused, F, rows=rng(3 * D, 3 * D + F))],
            D, s_in)
        # to_out [D][D + F]: the attention's columns, then the MLP's, as two products
        plans[f"single.{j}.out_attn"] = LinearPlan("out", [Part(f"{a}.to_out", D, cols=(0, D))], D, s_o)
        plans[f"single.{j}.out_mlp"] = LinearPlan("fc2", [Part(f"{a}.to_out", D, cols=(D, D + F))], F, s_f)
        aliases[f"single_blocks.{j}.linear1"] = [(fused, 0)]
        aliases[f"single_blocks.{j}.linear2"] = [(f"{a}.to_out", 0)]
    return plans, aliases


def convert_transformer(src: str, out: str, *, fmts=None, seed: int = 1, log=print):
    """src: the FLUX.2 directory (transformer/ with config.json and safetensors)."""
    f = {"qkv": "q8", "out": "q8", "fc1": "q8", "fc2": "q8", **(fmts or {})}
    with open(os.path.join(src, "transformer", "config.json")) as fh:
        cfg = json.load(fh)
    if cfg.get("guidance_embeds") or cfg.get("patch_size", 1) != 1:
        raise ValueError("h3metal runs FLUX.2 transformers without guidance embeddings and with patch size 1")
    st = Sharded(os.path.join(src, "transformer"))
    H, hd = cfg["num_attention_heads"], cfg["attention_head_dim"]
    D = H * hd
    F = int(D * cfg["mlp_ratio"])
    L2, L1 = cfg["num_layers"], cfg["num_single_layers"]
    meta = {"kind": "flux2-dit",
            "config": {"dim": D, "heads": H, "head_dim": hd, "ffn": F, "double_layers": L2, "single_layers": L1,
                       "in_channels": cfg["in_channels"], "joint_dim": cfg["joint_attention_dim"],
                       "axes_dims": cfg["axes_dims_rope"], "rope_theta": cfg["rope_theta"], "eps": cfg["eps"],
                       "t_channels": cfg["timestep_guidance_channels"]},
            "seed": seed, "fmts": f}
    w = Writer(out, meta)
    t0 = time.time()
    plans, _ = model_plans(meta["config"], seed)

    def get(m):
        return np.ascontiguousarray(st.get(f"{m}.weight"), np.float32)

    # Read on the CPU every step: the timestep MLP, the three modulation projections, the final modulation.
    for name in HOST:
        w.tensor(name, plans[name].assemble(get), "f32", host_only=True)
    for name in HALF:  # half-precision products on the matrix units
        m = plans[name].assemble(get)
        codes, _ = quantize("f16", m)
        w.linear(name, "f16", m.shape[0], m.shape[1], codes, None)

    perm = _pair_perm(H, hd)

    def put(name):
        plan = plans[name]
        mat = plan.assemble(get)
        codes, scales = quantize(f[plan.kind], mat)
        w.linear(name, f[plan.kind], mat.shape[0], mat.shape[1], codes, scales)

    def sign_vec(name, s):
        w.tensor(name, s.astype(np.float32), "f32", host_only=True)

    for i in range(L2):
        a = f"transformer_blocks.{i}.attn"
        for stream, (nq, nk) in (("img", ("norm_q", "norm_k")), ("txt", ("norm_added_q", "norm_added_k"))):
            name = f"double.{i}.{stream}"
            put(f"{name}.qkv")
            put(f"{name}.out")
            w.tensor(f"{name}.q_norm", st.get(f"{a}.{nq}.weight")[perm[:hd]].astype(np.float32), "f32")
            w.tensor(f"{name}.k_norm", st.get(f"{a}.{nk}.weight")[perm[:hd]].astype(np.float32), "f32")
            sign_vec(f"{name}.s_in", plans[f"{name}.qkv"].s_in)
            put(f"{name}.fc1")
            put(f"{name}.fc2")
            sign_vec(f"{name}.s_mlp", plans[f"{name}.fc1"].s_in)
        log(f"  double block {i + 1}/{L2} done ({time.time() - t0:.0f} s)")
    for j in range(L1):
        a = f"single_transformer_blocks.{j}.attn"
        name = f"single.{j}"
        put(f"{name}.qkv_mlp")
        put(f"{name}.out_attn")
        put(f"{name}.out_mlp")
        w.tensor(f"{name}.q_norm", st.get(f"{a}.norm_q.weight")[perm[:hd]].astype(np.float32), "f32")
        w.tensor(f"{name}.k_norm", st.get(f"{a}.norm_k.weight")[perm[:hd]].astype(np.float32), "f32")
        sign_vec(f"{name}.s_in", plans[f"{name}.qkv_mlp"].s_in)
        if (j + 1) % 5 == 0:
            log(f"  single block {j + 1}/{L1} done ({time.time() - t0:.0f} s)")
    w.close()
    log(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB) in {time.time() - t0:.0f} s")


class Flux2DiT:
    """The converted transformer on the engine: prepare() per request, then step() per step."""
    attn_cfg_q8 = dict(wm=10, bk=32, DB=1, PVORDER=1, VX=1)

    def __init__(self, gpu: Gpu, wf: WeightFile):
        self.gpu, self.wf = gpu, wf
        c = wf.meta["config"]
        self.cfg = c
        self.D, self.H, self.hd, self.F = c["dim"], c["heads"], c["head_dim"], c["ffn"]
        self.L2, self.L1 = c["double_layers"], c["single_layers"]
        self.eps = c["eps"]
        host = wf.host
        self.t_lin1, self.t_lin2 = host("t_lin1"), host("t_lin2")
        self.w_mod_img, self.w_mod_txt = host("mod_img"), host("mod_txt")
        self.w_mod_single, self.w_mod_final = host("mod_single"), host("mod_final")
        self.x_embed, self.ctx_embed, self.proj_out = (wf.linear(k) for k in ("x_embed", "ctx_embed", "proj_out"))

        def stream(name):
            d = {k: wf.linear(f"{name}.{k}") for k in ("qkv", "out", "fc1", "fc2")}
            d.update({k: wf.tensor(f"{name}.{k}") for k in ("q_norm", "k_norm")})
            d.update({k: host(f"{name}.{k}") for k in ("s_in", "s_mlp")})
            return d
        self.double = [{"img": stream(f"double.{i}.img"), "txt": stream(f"double.{i}.txt")} for i in range(self.L2)]
        self.single = []
        for j in range(self.L1):
            n = f"single.{j}"
            d = {k: wf.linear(f"{n}.{k}") for k in ("qkv_mlp", "out_attn", "out_mlp")}
            d.update({k: wf.tensor(f"{n}.{k}") for k in ("q_norm", "k_norm")})
            d["s_in"] = host(f"{n}.s_in")
            self.single.append(d)
        D = self.D
        # Per-step tables, written by the CPU: per double block mod1, mod2 [2 types][2][D] (image rows are type 0,
        # caption rows type 1) and gate1, gate2 [2][D]; per single block mod [1][2][D] and gate [1][D]; the final
        # norm's scale and shift [2][D].
        self.tab_layout = {}
        n = 0
        for i in range(self.L2):
            for k, size in (("mod1", 4 * D), ("mod2", 4 * D), ("gate1", 2 * D), ("gate2", 2 * D)):
                self.tab_layout[(i, k)] = (n, size)
                n += size
        for j in range(self.L1):
            for k, size in (("mod", 2 * D), ("gate", D)):
                self.tab_layout[(1000 + j, k)] = (n, size)
                n += size
        self.tab_layout["final"] = (n, 2 * D)
        n += 2 * D
        self.tables = gpu.empty((n,), "f32")
        self._bufs = None
        self.plans, self.aliases = model_plans(c, wf.meta.get("seed", 1))
        self.linears = {"x_embed": self.x_embed, "ctx_embed": self.ctx_embed, "proj_out": self.proj_out}
        for i, blk in enumerate(self.double):
            for stream in ("img", "txt"):
                for k in ("qkv", "out", "fc1", "fc2"):
                    self.linears[f"double.{i}.{stream}.{k}"] = blk[stream][k]
        for j, blk in enumerate(self.single):
            for k in ("qkv_mlp", "out_attn", "out_mlp"):
                self.linears[f"single.{j}.{k}"] = blk[k]
        self.host_low = {}  # host matrix -> (A, B) of LoRA adapters

    def set_loras(self, adapters):
        """LoRA adapters [(layers from h3metal.lora.read, strength)] for the next requests ([] removes them):
        low-rank terms on the products (h3metal/ops.py, low_rank) and on the CPU's modulation matrices. The
        caption embedding runs in prepare(), so set them before it."""
        for w in self.linears.values():
            w.low = None
        self.host_low = {}
        if not adapters:
            return
        scratch = ops.Scratch(self.gpu)
        for name, terms in lora.terms(adapters, self.plans, self.aliases, model="FLUX.2 [klein]").items():
            A, B = lora.stack(terms)
            if self.plans[name].kind == "host":
                self.host_low[name] = (A.astype(np.float64), B.astype(np.float64))
            else:
                act = "f16" if self.plans[name].kind == "f16" else "q8"
                self.linears[name].low = ops.low_rank(self.gpu, A, B, scratch, act=act)

    def _host(self, name: str, w: np.ndarray, x: np.ndarray) -> np.ndarray:
        """x W^T for a matrix the CPU applies, with its adapters' term."""
        y = x @ w.T
        if name in self.host_low:
            A, B = self.host_low[name]
            y = (y + (x.astype(np.float64) @ A.T) @ B.T).astype(np.float32)
        return y

    def _tab(self, key, shape) -> Tensor:
        off, size = self.tab_layout[key]
        return self.tables.view(shape, offset_elems=off)

    def modulation(self, sigma: float):
        """The step's modulation tables (CPU): FLUX.2's timestep embedding is that of sigma * 1000."""
        t = timestep_embedding(np.array([np.float32(sigma) * np.float32(1000)], np.float32), self.cfg["t_channels"])
        h = self._host
        temb = h("t_lin2", self.t_lin2, silu(h("t_lin1", self.t_lin1, t)))  # [1][D]
        st = silu(temb)
        m_img, m_txt = h("mod_img", self.w_mod_img, st)[0], h("mod_txt", self.w_mod_txt, st)[0]  # [6D]
        m_single, m_final = h("mod_single", self.w_mod_single, st)[0], h("mod_final", self.w_mod_final, st)[0]
        D = self.D
        tab = self.tables.numpy()

        def put(key, arr):
            off, size = self.tab_layout[key]
            tab[off:off + size] = np.asarray(arr, np.float32).reshape(-1)

        def chunk(m, i):
            return m[i * D:(i + 1) * D]
        for i, blk in enumerate(self.double):
            mods = {"img": m_img, "txt": m_txt}
            for which, (sh, sc, gt), sk in (("mod1", (0, 1, 2), "s_in"), ("mod2", (3, 4, 5), "s_mlp")):
                t2 = np.empty((2, 2, D), np.float32)
                g2 = np.empty((2, D), np.float32)
                for ty, stream in ((0, "img"), (1, "txt")):
                    s = blk[stream][sk]
                    m = mods[stream]
                    t2[ty, 0] = (1.0 + chunk(m, sc)) * s
                    t2[ty, 1] = chunk(m, sh) * s
                    g2[ty] = chunk(m, gt)
                put((i, which), t2)
                put((i, "gate1" if which == "mod1" else "gate2"), g2)
        for j, blk in enumerate(self.single):
            s = blk["s_in"]
            put((1000 + j, "mod"), np.stack([(1.0 + chunk(m_single, 1)) * s, chunk(m_single, 0) * s]))
            put((1000 + j, "gate"), chunk(m_single, 2))
        put("final", np.stack([1.0 + chunk(m_final, 0), chunk(m_final, 1)]))  # AdaLayerNormContinuous: scale, shift

    def _alloc(self, S: int):
        g, D, H, hd, F = self.gpu, self.D, self.H, self.hd, self.F
        Sp = ops.pad_rows(S, 128)
        if self._bufs and self._bufs["Sp"] >= Sp:
            return self._bufs
        b = {"Sp": Sp,
             "x": g.empty((Sp, D), "f32"), "y": g.empty((Sp, D), "f32"),
             "a8": g.empty((Sp, max(F, D)), "i8"), "sa": g.empty((Sp,), "f32"),
             "wide": g.empty((Sp, 3 * D + 2 * F), "bf16"), "o": g.empty((Sp, D), "bf16"),
             "Q": g.empty((H, Sp, hd), "i8"), "K": g.empty((H, Sp, hd), "i8"), "V": g.empty((H, Sp, hd), "f16"),
             "qs": g.empty((H, Sp), "f32"), "ks": g.empty((H, Sp), "f32"),
             "rt": g.empty((Sp,), "u8"), "rt0": g.empty((Sp,), "u8"),
             "cos": g.empty((Sp, hd // 2), "f32"), "sin": g.empty((Sp, hd // 2), "f32"),
             "lat": g.empty((Sp, self.cfg["in_channels"]), "f16"), "vel": g.empty((Sp, self.cfg["in_channels"]), "f32"),
             "yh": g.empty((Sp, D), "f16"),
             "txt": g.empty((Sp, D), "f32")}
        b["rt0"].numpy()[:] = 0
        self._bufs = b
        return b

    def free(self):
        self._bufs = None

    def prepare(self, txt_feats: np.ndarray, img_pos: np.ndarray):
        """Per request: the caption's features [T][joint_dim] (T a multiple of 128) and the image rows' positions
        [n_img][4] (t, h, w, layer). Embeds the caption once and sets the rotary tables of [caption, image]."""
        T = txt_feats.shape[0]
        if T % TXT_MULTIPLE:
            raise ValueError(f"FLUX.2 caption rows must be a multiple of {TXT_MULTIPLE} (got {T})")
        n_img = img_pos.shape[0]
        S = T + n_img
        b = self._alloc(S)
        self.T, self.n_img, self.S = T, n_img, S
        pos = np.zeros((S, 4), np.int64)
        pos[:T, 3] = np.arange(T)
        pos[T:] = img_pos
        cos, sin = rope_tables(pos, self.cfg["axes_dims"], self.cfg["rope_theta"])
        b["cos"].numpy()[:S] = cos
        b["sin"].numpy()[:S] = sin
        rt = b["rt"].numpy()
        rt[:] = 0
        rt[:T] = 1
        g = self.gpu
        f = g.empty((ops.pad_rows(T, 128), txt_feats.shape[1]), "f16")
        f.numpy()[:T] = txt_feats
        ops.gemm(g, f, self.ctx_embed, b["txt"], T, tile=(64, 128, 4))
        g.wait()

    def _attend(self, b, S: int):
        g, H, hd = self.gpu, self.H, self.hd
        ops.attention(g, b["Q"], b["K"], b["V"], b["o"], Lq=S, L=S, H=H, KVH=H, D=hd, q_rows=b["Sp"],
                      k_rows=b["Sp"], scale=hd ** -0.5, q_scale=b["qs"], k_scale=b["ks"], **self.attn_cfg_q8)

    def _qk(self, b, src: Tensor, blk, r0: int, M: int, in_stride: int):
        """Queries, keys (int8, with row scales) and values of rows r0 .. r0 + M, from their projections."""
        hd, Sp = self.hd, b["Sp"]
        ops.qk_norm_rope(self.gpu, src, blk["q_norm"], blk["k_norm"], b["cos"].row(r0, M), b["sin"].row(r0, M),
                         b["Q"].view(b["Q"].shape, offset_elems=r0 * hd), b["K"].view(b["K"].shape, offset_elems=r0 * hd),
                         b["V"].view(b["V"].shape, offset_elems=r0 * hd), M=M, Hq=self.H, Hk=self.H, Hv=self.H, hd=hd,
                         out_rows=Sp, rot=hd, eps=self.eps, in_stride=in_stride,
                         q_scale=b["qs"].view(b["qs"].shape, offset_elems=r0),
                         k_scale=b["ks"].view(b["ks"].shape, offset_elems=r0),
                         qk_rot=QK_ROT)

    def _double(self, b, i: int, blk):
        g, D, F, T, S, Sp = self.gpu, self.D, self.F, self.T, self.S, b["Sp"]
        x, a8, sa, eps = b["x"], b["a8"], b["sa"], self.eps
        rows = {"txt": (0, T), "img": (T, S - T)}
        ops.norm_mod_q8(g, x, self._tab((i, "mod1"), (2, 2, D)), b["rt"], a8.view((Sp, D)), sa, S, D, eps=eps, ln=True)
        qkv = b["wide"].view((Sp, 3 * D))
        for stream in ("txt", "img"):
            r0, M = rows[stream]
            w = blk[stream]["qkv"]
            ops.gemm(g, a8.view((Sp, D)).row(r0, M), w, qkv.row(r0, M), M, sa=sa.view((M,), offset_elems=r0),
                     **_ftile(w, M))
            self._qk(b, qkv.row(r0, M), blk[stream], r0, M, 3 * D)
        self._attend(b, S)
        ops.rot_q8(g, b["o"], a8.view((Sp, D)), sa, S, D)
        gate1 = self._tab((i, "gate1"), (2, D))
        for ty, stream in ((1, "txt"), (0, "img")):
            r0, M = rows[stream]
            w = blk[stream]["out"]
            ops.gemm(g, a8.view((Sp, D)).row(r0, M), w, x.row(r0, M), M, sa=sa.view((M,), offset_elems=r0), epi=1,
                     rtype=b["rt0"], gate=gate1.view((D,), offset_elems=ty * D), **_ftile(w, M))
        ops.norm_mod_q8(g, x, self._tab((i, "mod2"), (2, 2, D)), b["rt"], a8.view((Sp, D)), sa, S, D, eps=eps, ln=True)
        h = b["wide"].view((Sp, 2 * F))
        for stream in ("txt", "img"):
            r0, M = rows[stream]
            w = blk[stream]["fc1"]
            ops.gemm(g, a8.view((Sp, D)).row(r0, M), w, h.row(r0, M), M, sa=sa.view((M,), offset_elems=r0),
                     **_ftile(w, M))
        ops.rot_q8(g, h, a8.view((Sp, F)), sa, S, F, swiglu=True)
        gate2 = self._tab((i, "gate2"), (2, D))
        for ty, stream in ((1, "txt"), (0, "img")):
            r0, M = rows[stream]
            w = blk[stream]["fc2"]
            ops.gemm(g, a8.view((Sp, F)).row(r0, M), w, x.row(r0, M), M, sa=sa.view((M,), offset_elems=r0), epi=1,
                     rtype=b["rt0"], gate=gate2.view((D,), offset_elems=ty * D), **_ftile(w, M))

    def _single(self, b, j: int, blk):
        g, D, F, S, Sp = self.gpu, self.D, self.F, self.S, b["Sp"]
        x, a8, sa, eps = b["x"], b["a8"], b["sa"], self.eps
        W = 3 * D + 2 * F
        ops.norm_mod_q8(g, x, self._tab((1000 + j, "mod"), (1, 2, D)), b["rt0"], a8.view((Sp, D)), sa, S, D, eps=eps,
                        ln=True)
        wide = b["wide"].view((Sp, W))
        ops.gemm(g, a8.view((Sp, D)), blk["qkv_mlp"], wide, S, sa=sa, **_ftile(blk["qkv_mlp"], S))
        self._qk(b, wide, blk, 0, S, W)
        self._attend(b, S)
        gate = self._tab((1000 + j, "gate"), (1, D))
        ops.rot_q8(g, b["o"], a8.view((Sp, D)), sa, S, D)
        ops.gemm(g, a8.view((Sp, D)), blk["out_attn"], x, S, sa=sa, epi=1, rtype=b["rt0"], gate=gate,
                 **_ftile(blk["out_attn"], S))
        ops.rot_q8(g, wide.view((Sp, W), offset_elems=3 * D), a8.view((Sp, F)), sa, S, F, swiglu=True, x_stride=W)
        ops.gemm(g, a8.view((Sp, F)), blk["out_mlp"], x, S, sa=sa, epi=1, rtype=b["rt0"], gate=gate,
                 **_ftile(blk["out_mlp"], S))

    def step(self, sigma: float, lat_rows: np.ndarray) -> np.ndarray:
        """The transformer's output (the velocity) for the image rows' latents [n_img][in_channels] at sigma."""
        g, D, b = self.gpu, self.D, self._bufs
        T, n_img = self.T, self.n_img
        self.modulation(sigma)
        x = b["x"]
        b["lat"].numpy()[:n_img] = lat_rows
        ops.copy_f32(g, b["txt"], x, T * D)
        ops.gemm(g, b["lat"], self.x_embed, x.row(T, n_img), n_img, tile=(64, 128, 4))
        for i, blk in enumerate(self.double):
            self._double(b, i, blk)
            g.commit()
        for j, blk in enumerate(self.single):
            self._single(b, j, blk)
            g.commit()
        fin = self._tab("final", (2, D))
        y = b["yh"]
        ops.layer_norm(g, x.row(T, n_img), fin.view((D,)), fin.view((D,), offset_elems=D), y, n_img, D, eps=self.eps)
        ops.gemm(g, y, self.proj_out, b["vel"], n_img, tile=(64, 128, 4))
        g.wait()
        return b["vel"].numpy()[:n_img].copy()


# -- the whole model ---------------------------------------------------------------------------------------------

MAX_TOKENS = 512      # klein pads (and truncates) every prompt to this many tokens
TAPS = (9, 18, 27)    # the text encoder's hidden states the transformer reads, stacked per token
PAD_ID = 151643       # Qwen3's <|endoftext|>, the tokenizer's padding


def chat_prompt(prompt: str) -> str:
    """The text klein encodes: Qwen3's chat template with one user message and thinking disabled."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def image_positions(hp: int, wp: int, t: int = 0) -> np.ndarray:
    """Rotary positions (t, h, w, layer) of an hp x wp grid of image rows, row-major."""
    hh, ww = np.meshgrid(np.arange(hp), np.arange(wp), indexing="ij")
    pos = np.zeros((hp * wp, 4), np.int64)
    pos[:, 0], pos[:, 1], pos[:, 2] = t, hh.reshape(-1), ww.reshape(-1)
    return pos


def prepare_reference(image) -> np.ndarray:
    """A reference image (a PIL image) as klein's pipeline prepares it: scaled down to 1024 x 1024 pixels of area
    if larger (Lanczos), center-cropped to multiples of 16 -> float [h][w][3] in [-1, 1]."""
    import math
    from PIL import Image
    im = image.convert("RGB")
    w, h = im.size
    if w * h > 1024 * 1024:
        s = math.sqrt(1024 * 1024 / (w * h))
        im = im.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
        w, h = im.size
    cw, ch = w // 16 * 16, h // 16 * 16
    left, top = (w - cw) // 2, (h - ch) // 2
    im = im.crop((left, top, left + cw, top + ch))
    return (np.asarray(im, np.float32) / np.float32(255) * 2 - 1).astype(np.float32)


VARIANTS = {3072: "FLUX.2 [klein] 4B", 4096: "FLUX.2 [klein] 9B"}  # by the transformer's width


def variant_name(dim: int) -> str:
    return VARIANTS.get(dim, f"FLUX.2 [klein] (width {dim})")


def flux2_name(model_dir: str) -> str:
    """A converted directory's model name, from its transformer's width."""
    from .weights import read_header
    meta = read_header(os.path.join(os.path.expanduser(model_dir), "dit.h3m"))["meta"]
    return variant_name(meta["config"]["dim"])


def checkpoint_name(src: str) -> str:
    """An original checkpoint's model name, from its transformer's config."""
    with open(os.path.join(os.path.expanduser(src), "transformer", "config.json")) as fh:
        c = json.load(fh)
    return variant_name(c["num_attention_heads"] * c["attention_head_dim"])


def is_flux2_dir(model_dir: str) -> bool:
    """A converted FLUX.2 directory (its transformer file says so)."""
    p = os.path.join(os.path.expanduser(model_dir), "dit.h3m")
    if not os.path.exists(p):
        return False
    from .weights import read_header
    return read_header(p)["meta"].get("kind") == "flux2-dit"


def is_flux2_checkpoint(src: str) -> bool:
    """An original FLUX.2 [klein] checkpoint (diffusers layout)."""
    p = os.path.join(os.path.expanduser(src), "model_index.json")
    if not os.path.exists(p):
        return False
    with open(p) as fh:
        return json.load(fh).get("_class_name") == "Flux2KleinPipeline"


def convert_flux2(src: str, out: str, *, dit: str = "q8", te: str = "q8", parts=None, log=print):
    """The whole model: tokenizer, text encoder (the layers up to the last tap), transformer, VAE decoder."""
    import shutil
    from .flux_vae import convert_flux_vae
    from .te import convert_te
    src, out = os.path.expanduser(src), os.path.expanduser(out)
    os.makedirs(out, exist_ok=True)
    parts = parts or ["tokenizer", "vae", "te", "dit"]
    fmts = dict(zip(("qkv", "out", "fc1", "fc2"), dit.split(","))) if "," in dit else \
        {k: dit for k in ("qkv", "out", "fc1", "fc2")}
    if "tokenizer" in parts:
        shutil.copy(os.path.join(src, "tokenizer", "tokenizer.json"), os.path.join(out, "tokenizer.json"))
    if "vae" in parts:
        convert_flux_vae(src, os.path.join(out, "vae.h3m"), log=log, encoder=True)  # the encoder for edits
    if "te" in parts:
        convert_te(os.path.join(src, "text_encoder"), os.path.join(out, "te.h3m"), fmt=te, layers=max(TAPS),
                   prefix="model", log=log)
    if "dit" in parts:
        convert_transformer(src, os.path.join(out, "dit.h3m"), fmts=fmts, log=log)


class Flux2Pipeline:
    """Text to image, and edits from reference images, with FLUX.2 [klein]: four steps, no guidance, the FLUX.2
    decoder. A reference image's latents join the image rows (after the generated image's) at their own place on
    the rotary embedding's first axis, 10 per reference."""
    model_name = "FLUX.2 [klein] 4B"  # each pipeline reads its own (flux2_name)
    default_size = (1024, 1024)
    steps = 4

    def __init__(self, model_dir: str, gpu: Gpu | None = None, log=print):
        from tokenizers import Tokenizer
        model_dir = os.path.expanduser(model_dir)
        self.dir = model_dir
        self.gpu = gpu or Gpu()
        self.log = log
        self.paths = {k: os.path.join(model_dir, f"{k}.h3m") for k in ("te", "dit", "vae")}
        for k, p in self.paths.items():
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing {p}: `h3metal download --model flux2-klein-4b` fetches 4B ready to "
                                        "run; 9B takes `h3metal download --model flux2-klein-9b --accept-license` "
                                        "and `h3metal convert`")
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.model_name = flux2_name(model_dir)
        self._te_cache: dict[str, np.ndarray] = {}
        self._dit = self._dit_wf = None
        self.loras = lora.Loaded()

    def tokenize(self, prompt: str) -> tuple[np.ndarray, int]:
        """-> (ids padded to MAX_TOKENS, the number of prompt tokens)."""
        ids = self.tokenizer.encode(chat_prompt(prompt), add_special_tokens=False).ids[:MAX_TOKENS]
        out = np.full(MAX_TOKENS, PAD_ID, np.int64)
        out[:len(ids)] = ids
        return out, len(ids)

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
            ids, n = self.tokenize(prompt)
            emb = TextEncoder(self.gpu, wf).encode(ids, taps=TAPS, n_valid=n)
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
        key = f"flux2|{prompt}|{st.st_size}|{int(st.st_mtime)}"
        return os.path.join(root, hashlib.sha256(key.encode()).hexdigest()[:32] + ".npy")

    def release(self):
        """Unmaps the transformer (before the text encoder needs the memory)."""
        if self._dit is not None:
            self._dit.free()
            self._dit = None
            self._dit_wf.close()
            self._dit_wf = None

    def generate(self, prompt: str, width: int | None = None, height: int | None = None, seed: int = 0,
                 progress=None, latents: np.ndarray | None = None, images=None, loras=(), **_):
        """-> (uint8 image [height][width][3], Timing). images: reference images (PIL) for an edit; without a size
        the image takes the first reference's (prepared) size, as in the official pipeline. latents: the starting
        noise [C][height/16][width/16] instead of the seed's (tests). loras: [(path to a LoRA file, strength)]."""
        from .flux_vae import FluxVAEDecoder, FluxVAEEncoder
        from .pipeline import Timing, noise
        adapters = self.loras.read(loras)  # before the work: a file that cannot be applied stops here
        refs = [prepare_reference(im) for im in images or []]
        if refs and (width is None or height is None):
            height, width = refs[0].shape[:2]
        width, height = width or self.default_size[0], height or self.default_size[1]
        if width % 16 or height % 16:
            raise ValueError("FLUX.2 needs width and height that are multiples of 16")
        timing = Timing()
        emb = self.encode_prompt(prompt, timing)
        ref_rows, ref_pos = [], []
        if refs:
            t = time.time()
            self.release()
            wf = WeightFile(self.gpu, self.paths["vae"])
            enc = FluxVAEEncoder(self.gpu, wf)
            for i, im in enumerate(refs):
                p = enc.encode_packed(im)  # [C][h][w]
                ref_rows.append(p.reshape(p.shape[0], -1).T)
                ref_pos.append(image_positions(p.shape[1], p.shape[2], t=10 + 10 * i))
            enc.release()
            wf.close()
            timing.add("references", time.time() - t)
        t = time.time()
        if self._dit is None:
            self._dit_wf = WeightFile(self.gpu, self.paths["dit"])
            self._dit = Flux2DiT(self.gpu, self._dit_wf)
            self.loras.applied = None
            timing.add("load", time.time() - t)
        dit = self._dit
        t = time.time()
        if self.loras.apply(dit, adapters) and adapters:
            timing.add("lora", time.time() - t)
        hp, wp = height // 16, width // 16
        C = dit.cfg["in_channels"]
        t = time.time()
        dit.prepare(emb, np.concatenate([image_positions(hp, wp)] + ref_pos))
        timing.add("text rows", time.time() - t)
        x = latents if latents is not None else noise(seed, (1, C, hp, wp)).reshape(C, hp, wp)
        x = np.ascontiguousarray(x.reshape(C, hp * wp).T, np.float32)  # rows [hp*wp][C]
        fixed = np.concatenate(ref_rows).astype(np.float32) if ref_rows else None
        sig = schedule(self.steps, hp * wp)  # the shift follows the generated image's rows only
        for s in range(self.steps):
            t = time.time()
            v = dit.step(float(sig[s]), x if fixed is None else np.concatenate([x, fixed]))[:hp * wp]
            x = (x + np.float32(sig[s + 1] - sig[s]) * v).astype(np.float32)
            timing.add("denoise", time.time() - t)
            if progress:
                progress(s + 1, self.steps)
        t = time.time()
        dit.free()
        wf = WeightFile(self.gpu, self.paths["vae"])
        img = FluxVAEDecoder(self.gpu, wf).decode(x.T.reshape(C, hp, wp))
        wf.close()
        timing.add("decode", time.time() - t)
        out = np.clip(np.rint((img / 2 + 0.5).clip(0, 1) * 255), 0, 255).astype(np.uint8)
        return out, timing
