"""Reference side of the real-weights check of MiniMax-H3-Image's few-step transformer (convert --pdmd): the token
refiner, the first block, every embedding and the output layer with pDMD's LoRA merged in float32, run in float32 on
the CPU by diffusers' MiniMax-H3 transformer, at the timesteps of the 2-step grid.

    python tests/ref/minimax_truncated.py SRC LORA OUT_DIR CAPTION.npy

SRC is the original checkpoint (only the first transformer shard is read: it holds all of the above). LORA is
pDMD's file, as released (diffusers' module names). CAPTION.npy holds real caption features [T][5120] (h3metal's
text encoder on a prompt). diffusers' scripts/convert_minimax_h3_to_diffusers.py does the key conversion; its
directory comes from DIFFUSERS_SCRIPTS (default ~/project/h3-ref/diffusers-src). OUT_DIR receives ref.npz: the
latents (one 1024 x 1024 image), the caption features, the grid and both steps' velocities.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from diffusers import MiniMaxH3Scheduler
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3 import modular_pipeline as video_pipe
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep, patchify_video_latents
from safetensors import safe_open
from safetensors.torch import load_file

src, lora_path, out, cap = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
scripts = os.environ.get("DIFFUSERS_SCRIPTS", os.path.expanduser("~/project/h3-ref/diffusers-src"))
spec = importlib.util.spec_from_file_location("conv", os.path.join(scripts, "convert_minimax_h3_to_diffusers.py"))
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

c = json.load(open(os.path.join(src, "transformer", "config.json")))
cfg = dict(num_attention_heads=c["num_attention_heads"], attention_head_dim=c["attention_head_dim"],
           hidden_size=c["hidden_size"], num_layers=1, num_refiner_layers=c["token_refiner_num_layers"],
           ffn_dim=c["ffn_hidden_size"], in_channels=c["latents_dim"], audio_in_channels=c["audio_latents_dim"],
           patch_size=tuple(c["patch_size"]), text_dim=c["text_dim"], freq_dim=c["timestep_input_dim"],
           time_embed_hidden_dim=c["time_embed_hidden_size"], time_embed_dim=c["time_embed_dim"],
           rope_freq_dim=c["rope_inv_freq_len"], rope_theta=10000.0, norm_eps=c["norm_eps"],
           qk_norm_eps=c["qk_norm_eps"], final_norm_eps=c["final_norm_eps"])
model = MiniMaxH3Transformer3DModel(**cfg).float().eval()

# 1. The checkpoint's tensors for that model, converted by diffusers' own script.
index = json.load(open(os.path.join(src, "transformer", "model.safetensors.index.json")))["weight_map"]
wanted = [k for k in index if not k.startswith("blocks.") or k.startswith("blocks.0.")]
sd = {}
for shard in sorted({index[k] for k in wanted}):
    with safe_open(os.path.join(src, "transformer", shard), framework="pt") as f:
        for k in wanted:
            if index[k] != shard:
                continue
            t = f.get_tensor(k).float()
            if k.endswith(".attn.qkv_proj.weight"):
                t = conv.reorder_interleaved_qkv(t, cfg["num_attention_heads"], cfg["attention_head_dim"])
            for nk, nt in conv.convert_transformer_key(k, t, cfg):
                sd[nk] = nt
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not unexpected and all(k.startswith("audio_") for k in missing), (unexpected, missing)

# 2. pDMD's LoRA merged in float32, as its card does it (W += B @ A, scale 1), on the modules this model has.
lora = load_file(lora_path)
params = dict(model.named_parameters())
merged = 0
with torch.no_grad():
    for k, a in lora.items():
        if not k.endswith(".lora_A.weight"):
            continue
        name = k[len("transformer."):].replace(".lora_A.weight", ".weight")
        if name in params:
            params[name].add_(lora[k.replace(".lora_A.", ".lora_B.")].float() @ a.float())
            merged += 1
print(f"merged {merged} LoRA pairs (the refiner's and block 0's)")
assert merged == 12, merged

# 3. One 1024 x 1024 image (64 x 64 latents), packed by the video pipeline as one frame; the 2-step grid.
txt = torch.from_numpy(np.load(cap).astype(np.float32)).to(torch.bfloat16).float()[None]
T = txt.shape[1]
lat_h = lat_w = 64
rng = np.random.default_rng(0)
lat = torch.from_numpy(rng.standard_normal((1, cfg["in_channels"], 1, lat_h, lat_w)).astype(np.float32))
target = patchify_video_latents(lat, cfg["patch_size"])
tags = torch.ones(T, dtype=torch.long)
pos, ttags, v_idx, a_idx, t_idx, _, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
    tags, 1, lat_h, lat_w, 0, cfg["patch_size"], video_pipe.MINIMAX_H3_AUDIO_CHANNELS,
    video_pipe.MINIMAX_H3_AUDIO_TAG, video_pipe.MINIMAX_H3_VIDEO_TAG)
sched = MiniMaxH3Scheduler(shift=12.0, shift_provided_sigmas=False)
sched.set_timesteps(3)
rec = {"latents": target.numpy(), "txt": txt[0].numpy(), "lat_hw": np.array([lat_h, lat_w]),
       "sigmas": sched.sigmas.numpy()}
for s, t in enumerate(sched.timesteps):
    uniq, tix = torch.unique(torch.full((pos.shape[0],), float(t)), sorted=True, return_inverse=True)
    with torch.no_grad():
        v = model(hidden_states=target[None], audio_hidden_states=torch.zeros(1, 0, cfg["audio_in_channels"]),
                  encoder_hidden_states=txt, timestep=uniq, timestep_indices=tix, token_tags=ttags,
                  position_ids=pos.float(), video_indices=v_idx, audio_indices=a_idx, text_indices=t_idx,
                  return_dict=False)[0]
    rec[f"vel{s}"] = v[0].numpy()
np.savez(os.path.join(out, "ref.npz"), **rec)
print("saved", out, {k: getattr(v, "shape", v) for k, v in rec.items()})
