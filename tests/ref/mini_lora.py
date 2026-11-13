"""Reference side of the LoRA tests (runs in an environment with torch and diffusers).

Makes a random LoRA adapter for a mini model of tests/ref/mini_zimage.py or tests/ref/mini_flux2.py, writes it
in the file layouts h3metal reads, and records the model's outputs with the adapter merged into its float32
weights (W + strength * alpha / r * B A, which is what a LoRA is) on the mini reference's trajectory, at
strengths 1 and 0.5.

    python tests/ref/mini_lora.py zimage|flux2 MODEL_DIR

MODEL_DIR is the mini model's directory (with transformer/ and ref.npz); the files go to MODEL_DIR/lora/:
  peft.safetensors     diffusers names, lora_A / lora_B, the ai-toolkit prefix diffusion_model.
  kohya.safetensors    lora_unet_ names with underscores, lora_down / lora_up and an alpha of r / 2
  fused.safetensors    the original checkpoint's names (fused attention projections), PEFT's base_model.model.
  ref.npz              out{s}_1, out{s}_05: the outputs per step; base{s}: without the adapter
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors.numpy import save_file

which, mdir = sys.argv[1], sys.argv[2]
out = os.path.join(mdir, "lora")
os.makedirs(out, exist_ok=True)
torch.manual_seed(1)
rng = np.random.default_rng(7)
R = 8
ref = np.load(os.path.join(mdir, "ref.npz"))

if which == "zimage":
    from diffusers import ZImageTransformer2DModel
    model = ZImageTransformer2DModel.from_pretrained(os.path.join(mdir, "transformer")).float().eval()
    cfg = json.load(open(os.path.join(mdir, "mini_config.json")))
    prefixes = [f"layers.{i}" for i in range(cfg["n_layers"])] + \
        [f"noise_refiner.{i}" for i in range(cfg["n_refiner_layers"])] + \
        [f"context_refiner.{i}" for i in range(cfg["n_refiner_layers"])]
    targets = []
    for p in prefixes:
        targets += [f"{p}.attention.{m}" for m in ("to_q", "to_k", "to_v", "to_out.0")]
        targets += [f"{p}.feed_forward.{m}" for m in ("w1", "w2", "w3")]
        if not p.startswith("context_refiner"):
            targets.append(f"{p}.adaLN_modulation.0")
    shared = {f"layers.0.attention.{m}": "layers.0.attention.qkv" for m in ("to_q", "to_k", "to_v")}
    fused_names = {"layers.0.attention.to_out.0": "layers.0.attention.out"}
else:
    from diffusers import Flux2Transformer2DModel
    model = Flux2Transformer2DModel.from_pretrained(os.path.join(mdir, "transformer")).float().eval()
    cfg = json.load(open(os.path.join(mdir, "mini_config.json")))
    targets = ["x_embedder", "context_embedder", "proj_out", "time_guidance_embed.timestep_embedder.linear_1",
               "time_guidance_embed.timestep_embedder.linear_2", "double_stream_modulation_img.linear",
               "double_stream_modulation_txt.linear", "single_stream_modulation.linear", "norm_out.linear"]
    shared, fused_names = {}, {}
    for i in range(cfg["num_layers"]):
        a = f"transformer_blocks.{i}.attn"
        for s_, (q, k, v, o), ff in (("img", ("to_q", "to_k", "to_v", "to_out.0"), "ff"),
                                     ("txt", ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"), "ff_context")):
            targets += [f"{a}.{m}" for m in (q, k, v, o)]
            targets += [f"transformer_blocks.{i}.{ff}.linear_in", f"transformer_blocks.{i}.{ff}.linear_out"]
            for m in (q, k, v):
                shared[f"{a}.{m}"] = f"double_blocks.{i}.{s_}_attn.qkv"
            fused_names[f"{a}.{o}"] = f"double_blocks.{i}.{s_}_attn.proj"
            fused_names[f"transformer_blocks.{i}.{ff}.linear_in"] = f"double_blocks.{i}.{s_}_mlp.0"
            fused_names[f"transformer_blocks.{i}.{ff}.linear_out"] = f"double_blocks.{i}.{s_}_mlp.2"
    for j in range(cfg["num_single_layers"]):
        a = f"single_transformer_blocks.{j}.attn"
        targets += [f"{a}.to_qkv_mlp_proj", f"{a}.to_out"]
        fused_names[f"{a}.to_qkv_mlp_proj"] = f"single_blocks.{j}.linear1"
        fused_names[f"{a}.to_out"] = f"single_blocks.{j}.linear2"
    fused_names.update({"x_embedder": "img_in", "context_embedder": "txt_in", "proj_out": "final_layer.linear",
                        "time_guidance_embed.timestep_embedder.linear_1": "time_in.in_layer",
                        "time_guidance_embed.timestep_embedder.linear_2": "time_in.out_layer",
                        "double_stream_modulation_img.linear": "double_stream_modulation_img.lin",
                        "double_stream_modulation_txt.linear": "double_stream_modulation_txt.lin",
                        "single_stream_modulation.linear": "single_stream_modulation.lin"})
    # norm_out.linear stays under its diffusers name in the fused file: the original final layer orders its
    # modulation (shift, scale), which a separate check covers (fused_final.safetensors)

mods = dict(model.named_modules())
lora = {}
shared_A = {}
for t in targets:
    w = mods[t].weight
    n_out, n_in = w.shape
    if t in shared:
        A = shared_A.setdefault(shared[t], rng.standard_normal((R, n_in)) / np.sqrt(n_in))
    else:
        A = rng.standard_normal((R, n_in)) / np.sqrt(n_in)
    # sized so that B A is a third of W's typical entry: the adapter moves every layer visibly
    w_rms = float(w.detach().pow(2).mean().sqrt())
    B = rng.standard_normal((n_out, R)) * w_rms * np.sqrt(n_in) / np.sqrt(R) / 3
    lora[t] = (A.astype(np.float32), B.astype(np.float32))

base_w = {t: mods[t].weight.detach().clone() for t in targets}


def set_strength(s):
    with torch.no_grad():
        for t, (A, B) in lora.items():
            mods[t].weight.copy_(base_w[t] + s * torch.from_numpy(B.astype(np.float64) @ A.astype(np.float64)).float())


def outputs():
    res = []
    with torch.no_grad():
        for s in range(len([k for k in ref.files if k.startswith("x") and k[1:].isdigit()])):
            x = torch.from_numpy(ref[f"x{s}"])
            if which == "zimage":
                tt = (1000 - torch.tensor([float(ref["timesteps"][s])])) / 1000
                o = model([x.unsqueeze(1)], tt, [torch.from_numpy(ref["cap"])], return_dict=False)[0][0][:, 0]
            else:
                n_txt = ref["txt"].shape[0]
                t_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1),
                                             torch.arange(n_txt))[None]
                o = model(hidden_states=x[None], encoder_hidden_states=torch.from_numpy(ref["txt"])[None],
                          timestep=torch.tensor([float(ref["timesteps"][s])]) / 1000,
                          img_ids=torch.from_numpy(ref["i_ids"])[None], txt_ids=t_ids, guidance=None,
                          return_dict=False)[0][0]
            res.append(o.numpy())
    return res


rec = {}
set_strength(0.0)
base = outputs()
for s, o in enumerate(base):
    rec[f"base{s}"] = o
    print(f"step {s}: without the adapter vs the mini reference: "
          f"{np.linalg.norm(o - ref[f'out{s}']) / np.linalg.norm(ref[f'out{s}']):.2e}")
for strength, tag in ((1.0, "1"), (0.5, "05")):
    set_strength(strength)
    for s, o in enumerate(outputs()):
        rec[f"out{s}_{tag}"] = o
        print(f"strength {strength} step {s}: the adapter moves the output by "
              f"{np.linalg.norm(o - base[s]) / np.linalg.norm(base[s]):.3f} (rel L2)")
np.savez(os.path.join(out, "ref.npz"), **rec)

# The files: the same adapter in three layouts.
peft = {}
for t, (A, B) in lora.items():
    peft[f"diffusion_model.{t}.lora_A.weight"] = A
    peft[f"diffusion_model.{t}.lora_B.weight"] = B
save_file(peft, os.path.join(out, "peft.safetensors"))

kohya = {}
for t, (A, B) in lora.items():
    k = "lora_unet_" + t.replace(".", "_")
    kohya[f"{k}.lora_down.weight"] = A
    kohya[f"{k}.lora_up.weight"] = (B * 2).astype(np.float32)   # alpha r / 2 halves the product
    kohya[f"{k}.alpha"] = np.array(R / 2, np.float32)
save_file(kohya, os.path.join(out, "kohya.safetensors"))

fused, groups = {}, {}
for t, (A, B) in lora.items():
    if t in shared:
        groups.setdefault(shared[t], []).append((t, A, B))
        continue
    name = fused_names.get(t, t)
    fused[f"base_model.model.{name}.lora_A.weight"] = A
    fused[f"base_model.model.{name}.lora_B.weight"] = B
for name, parts in groups.items():
    parts.sort(key=lambda p: next(i for i, s_ in enumerate(("_q", "_k", "_v")) if s_ in p[0].rsplit(".", 1)[1]))
    fused[f"base_model.model.{name}.lora_A.weight"] = parts[0][1]
    fused[f"base_model.model.{name}.lora_B.weight"] = np.concatenate([p[2] for p in parts])
save_file(fused, os.path.join(out, "fused.safetensors"))
if which == "flux2":  # the original final layer's modulation is (shift, scale): the same adapter, its B swapped
    D = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    A, B = lora["norm_out.linear"]
    final = {k: v for k, v in fused.items() if ".norm_out.linear." not in k}
    final["base_model.model.final_layer.adaLN_modulation.1.lora_A.weight"] = A
    final["base_model.model.final_layer.adaLN_modulation.1.lora_B.weight"] = np.concatenate([B[D:], B[:D]])
    save_file(final, os.path.join(out, "fused_final.safetensors"))
print("saved", out, len(lora), "adapted layers")
