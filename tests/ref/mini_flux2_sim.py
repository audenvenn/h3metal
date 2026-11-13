"""The error h3metal's quantization alone gives the mini FLUX.2 model, simulated in float with diffusers.

Every block matrix gets the weights h3metal stores (rotated, quantized, dequantized, rotated back) and every
block product's input the rounding h3metal's kernels apply (rotated, int8 per row, rotated back; a
single-stream block's output projection quantizes its attention and MLP parts separately); the attention's
queries and keys are int8 per row and head, its values half precision. If the engine's error on the same
inputs (tests/test_flux2_mini.py) matches this, the engine adds no error of its own.

    python tests/ref/mini_flux2_sim.py REF_DIR [q8|q4]
"""
import json
import os
import sys

import numpy as np
import torch
import diffusers.models.transformers.transformer_flux2 as tf2
from diffusers import Flux2Transformer2DModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from h3metal.quant import RB, dequant, hadamard, quantize, random_signs, rotate_in  # noqa: E402

src = sys.argv[1]
fmt = sys.argv[2] if len(sys.argv) > 2 else "q8"
r = np.load(os.path.join(src, "ref.npz"))
model = Flux2Transformer2DModel.from_pretrained(os.path.join(src, "transformer")).float().eval()
H = torch.from_numpy(hadamard()).double()
D = model.config.num_attention_heads * model.config.attention_head_dim


def rot(x, s):  # rows of x (..., K) -> blockwise Hadamard of x * s
    y = (x.double() * s).reshape(*x.shape[:-1], -1, RB) @ H
    return y.reshape(x.shape)


def unrot(y, s):
    return ((y.double().reshape(*y.shape[:-1], -1, RB) @ H).reshape(y.shape)) * s


def q8_rows(u):
    sc = u.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
    return torch.round(u / sc).clamp(-127, 127) * sc


def act_quant(x, s):
    return unrot(q8_rows(rot(x, s)), s).to(x.dtype)


seed = 0
for name, mod in model.named_modules():
    if not isinstance(mod, torch.nn.Linear) or not name.startswith(("transformer_blocks", "single_transformer_blocks")):
        continue
    W = mod.weight.detach().double().numpy()
    K = W.shape[1]
    split = D if name.endswith("attn.to_out") and name.startswith("single") else None
    parts = [(0, split), (split, K)] if split else [(0, K)]
    s_all = np.concatenate([random_signs(b - a, seed := seed + 1) for a, b in parts])
    Wr = rotate_in(W.astype(np.float32), s_all.astype(np.float32))
    codes, scales = quantize(fmt, Wr)
    Wd = dequant(fmt, codes, scales, W.shape[0], K)
    Wback = (torch.from_numpy(Wd.astype(np.float64)).reshape(W.shape[0], -1, RB) @ H).reshape(W.shape) * \
        torch.from_numpy(s_all)
    with torch.no_grad():
        mod.weight.copy_(Wback.float())
    s_t = torch.from_numpy(s_all.astype(np.float64))

    def hook(m, args, s_t=s_t, parts=parts):
        x = args[0]
        return (torch.cat([act_quant(x[..., a:b], s_t[a:b]) for a, b in parts], dim=-1),)
    mod.register_forward_pre_hook(hook)

real_attn = tf2.dispatch_attention_fn


def attn_q(query, key, value, **kw):  # (B, S, H, hd): int8 queries and keys per row and head, half values
    return real_attn(q8_rows(query.double()).float(), q8_rows(key.double()).float(),
                     value.half().float(), **kw)


tf2.dispatch_attention_fn = attn_q
n_txt = r["txt"].shape[0]
t_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(n_txt))[None]
i_ids = torch.from_numpy(r["i_ids"])[None]
txt = torch.from_numpy(r["txt"])[None]
worst, errs = 0.0, []
with torch.no_grad():
    for s in range(4):
        t = torch.tensor([float(r["timesteps"][s])])
        o = model(hidden_states=torch.from_numpy(r[f"x{s}"])[None], encoder_hidden_states=txt, timestep=t / 1000,
                  img_ids=i_ids, txt_ids=t_ids, guidance=None, return_dict=False)[0][0].numpy()
        ref = r[f"out{s}"]
        rel = float(np.linalg.norm(o - ref) / np.linalg.norm(ref))
        worst = max(worst, rel)
        errs.append(rel)
        print(f"simulated {fmt} step {s}: rel L2 err {rel:.4f}")
print(f"simulated {fmt}: worst {worst:.4f}")
json.dump(errs, open(os.path.join(src, f"sim_{fmt}.json"), "w"))  # read by tests/test_flux2_mini.py
