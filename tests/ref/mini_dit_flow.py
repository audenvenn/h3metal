"""Reference side of the mini-DiT test of the flow sampler with a merged LoRA (runs with torch and diffusers).

MiniMax-H3-Image's few-step adapters (pDMD's 2-step LoRA) were trained on MiniMax-H3, the video model, and are
used with MiniMax-H3-Image as a one-frame video: its transformer's own output head (not the six per-step heads),
a plain flow-matching grid and the Euler blend update. This builds a small MiniMaxH3Transformer3DModel with random
weights and a random LoRA in the release's layout (`transformer.<module>.lora_A/B.weight`, diffusers module
names), writes the model in the original checkpoint's layout next to the LoRA file, and saves the merged model's
velocities at every step of the 2-, 3- and 4-step grids for a text-to-image layout, packed by the video pipeline.

It also checks the premise h3metal relies on: MiniMax-H3-Image's transformer class with the video model's output
head computes the same as the video transformer on a one-frame image.

    python tests/ref/mini_dit_flow.py OUT_DIR
"""
import json
import os
import sys

import numpy as np
import torch
from diffusers import MiniMaxH3ImageTransformer2DModel, MiniMaxH3Scheduler
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3 import modular_pipeline as video_pipe
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep, patchify_video_latents
from diffusers.modular_pipelines.minimax_h3_image import modular_pipeline as image_pipe
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import build_image_packed_sequence
from safetensors.torch import save_file

out = sys.argv[1]
os.makedirs(os.path.join(out, "transformer"), exist_ok=True)
torch.manual_seed(0)
H, HD, D, F, TXT, TE, TEH, RANK, SHIFT = 4, 128, 512, 1024, 256, 256, 512, 8, 12.0
cfg = dict(num_attention_heads=H, attention_head_dim=HD, hidden_size=D, num_layers=2, num_refiner_layers=2, ffn_dim=F,
           in_channels=24, patch_size=(1, 2, 2), text_dim=TXT, freq_dim=256, time_embed_hidden_dim=TEH,
           time_embed_dim=TE, rope_freq_dim=16, rope_theta=10000.0)
model = MiniMaxH3Transformer3DModel(**cfg).float().eval()
fp32_prefix = ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out")
with torch.no_grad():
    for name, p in model.named_parameters():
        std = 1.0 if "norm" in name else (0.5 / p.shape[-1] ** 0.5 if p.ndim == 2 else 0.05)
        v = torch.randn_like(p) * std + (1.0 if "norm" in name else 0.0)
        if "adaln_proj" in name or "norm_out.linear" in name:
            v = v * 0.5
        if not name.startswith(fp32_prefix):
            v = v.to(torch.bfloat16).float()  # the checkpoint stores these in bfloat16
        p.copy_(v)
sd = {k: v.clone() for k, v in model.state_dict().items()}

# 1. The base model in the original checkpoint's layout (as tests/ref/mini_dit.py), with the plain output head.
def orig_block(src, dst, adaln):
    o = {}
    q, k, v = sd[f"{src}.attn.to_q.weight"], sd[f"{src}.attn.to_k.weight"], sd[f"{src}.attn.to_v.weight"]
    o[f"{dst}.attn.qkv_proj.weight"] = torch.stack([q.view(H, HD, D), k.view(H, HD, D), v.view(H, HD, D)], 1).reshape(3 * H * HD, D)
    o[f"{dst}.attn.q_norm.weight"] = sd[f"{src}.attn.norm_q.weight"]
    o[f"{dst}.attn.k_norm.weight"] = sd[f"{src}.attn.norm_k.weight"]
    o[f"{dst}.attn.out_proj.weight"] = sd[f"{src}.attn.to_out.0.weight"]
    vg = sd[f"{src}.ff.net.0.proj.weight"]  # [value; gate]
    o[f"{dst}.mlp.fc1.weight"] = torch.cat([vg[F:], vg[:F]])  # original: [gate; value]
    o[f"{dst}.mlp.fc2.weight"] = sd[f"{src}.ff.net.2.weight"]
    o[f"{dst}.norm1.weight"] = sd[f"{src}.norm1.weight"]
    o[f"{dst}.norm2.weight"] = sd[f"{src}.norm2.weight"]
    if adaln:
        o[f"{dst}.adaln_proj.linear.weight"] = sd[f"{src}.adaln_proj.linear.weight"]
        o[f"{dst}.adaln_proj.linear.bias"] = sd[f"{src}.adaln_proj.linear.bias"]
    return o

ck = {
    "video_patch_proj.weight": sd["proj_in.weight"], "video_patch_proj.bias": sd["proj_in.bias"],
    "condition_proj.weight": sd["context_embedder.weight"], "condition_proj.bias": sd["context_embedder.bias"],
    "time_embedder.proj_in.weight": sd["time_embedder.linear_1.weight"], "time_embedder.proj_in.bias": sd["time_embedder.linear_1.bias"],
    "time_embedder.proj_out.weight": sd["time_embedder.linear_2.weight"], "time_embedder.proj_out.bias": sd["time_embedder.linear_2.bias"],
    "token_refiner.final_norm.weight": sd["token_refiner.final_norm.weight"],
    "final_layer.norm.weight": sd["norm_out.norm.weight"],
    "final_layer.adaln_proj.linear.weight": sd["norm_out.linear.weight"], "final_layer.adaln_proj.linear.bias": sd["norm_out.linear.bias"],
    "final_layer.video_out.weight": sd["proj_out.weight"], "final_layer.video_out.bias": sd["proj_out.bias"],
}
for i in range(2):
    ck.update(orig_block(f"token_refiner.refiner_blocks.{i}", f"token_refiner.blocks.{i}", False))
for l in range(2):
    ck.update(orig_block(f"transformer_blocks.{l}", f"blocks.{l}", True))
f32 = ("video_patch_proj", "time_embedder", "final_layer.video_out")
ck = {k: (v.float() if k.startswith(f32) else v.to(torch.bfloat16)).contiguous() for k, v in ck.items()}
save_file(ck, os.path.join(out, "transformer", "model.safetensors"))
json.dump({"_class_name": "MiniMaxH3DiTModel", "hidden_size": D, "num_layers": 2, "token_refiner_num_layers": 2,
           "num_attention_heads": H, "attention_head_dim": HD, "ffn_hidden_size": F, "latents_dim": 24,
           "patch_size": [1, 2, 2], "text_dim": TXT, "timestep_input_dim": 256, "time_embed_hidden_size": TEH,
           "time_embed_dim": TE, "rope_inv_freq_len": 16, "norm_eps": 1e-5, "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5},
          open(os.path.join(out, "transformer", "config.json"), "w"))
# The six per-step heads and their grid, which the flow sampler does not use (the converter's default reads them).
g = torch.Generator().manual_seed(1)
save_file({"video_out.weight": (torch.randn(6, 96, D, generator=g) * 0.02).contiguous(),
           "video_out.bias": torch.zeros(6, 96).contiguous()}, os.path.join(out, "pdd_fused_heads.safetensors"))
sig6 = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
json.dump({"num_inference_steps": 6, "sigmas": sig6}, open(os.path.join(out, "pdd_config.json"), "w"))

# 2. A random LoRA on every attention and feed-forward matrix, as the pDMD release has it. Each delta is as large as
# its matrix: the blocks of a random mini model move the output little, and a wrong row order or a missed matrix has
# to stand out against int8 rounding in the comparison.
modules = []
for prefix in [f"token_refiner.refiner_blocks.{i}" for i in range(2)] + [f"transformer_blocks.{l}" for l in range(2)]:
    modules += [f"{prefix}.attn.to_q", f"{prefix}.attn.to_k", f"{prefix}.attn.to_v", f"{prefix}.attn.to_out.0",
                f"{prefix}.ff.net.0.proj", f"{prefix}.ff.net.2"]
lora = {}
with torch.no_grad():
    for m in modules:
        w = sd[f"{m}.weight"]
        a = torch.randn(RANK, w.shape[1], generator=g) / w.shape[1] ** 0.5
        b = torch.randn(w.shape[0], RANK, generator=g) * (0.5 / RANK ** 0.5)
        lora[f"transformer.{m}.lora_A.weight"] = a.contiguous()
        lora[f"transformer.{m}.lora_B.weight"] = b.contiguous()
        model.get_parameter(f"{m}.weight").add_(b @ a)
save_file(lora, os.path.join(out, "lora.safetensors"))

# 3. MiniMax-H3-Image's class on the same merged weights, its six heads set to the video model's head.
assert video_pipe.MINIMAX_H3_VIDEO_TAG == image_pipe.MINIMAX_H3_IMAGE_TAG
image = MiniMaxH3ImageTransformer2DModel(**cfg, num_pdd_heads=6).float().eval()
msd = model.state_dict()
isd = {k: v for k, v in msd.items() if not k.startswith(("audio_", "proj_out"))}
for i in range(6):
    isd[f"pdd_proj_out.{i}.weight"], isd[f"pdd_proj_out.{i}.bias"] = msd["proj_out.weight"], msd["proj_out.bias"]
missing, unexpected = image.load_state_dict(isd, strict=False)
assert not unexpected and all("rope" in k for k in missing), (missing, unexpected)

# 4. Velocities at every step of the 2-, 3- and 4-step grids, from the scheduler the adapter's users run
# (set_timesteps(N + 1) for N steps), one text-to-image layout, one set of latents for every step; and the scheduler's
# own update from the first velocity, for the sampler's arithmetic.
rng = np.random.default_rng(5)
T, lat_h, lat_w = 37, 16, 20
tags = torch.ones(T, dtype=torch.long)
text = torch.from_numpy(rng.standard_normal((1, T, TXT)).astype(np.float32)).to(torch.bfloat16).float()
target = patchify_video_latents(torch.from_numpy(rng.standard_normal((1, 24, 1, lat_h, lat_w)).astype(np.float32)), (1, 2, 2))
pos, ttags, v_idx, a_idx, t_idx, _, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
    tags, 1, lat_h, lat_w, 0, (1, 2, 2), video_pipe.MINIMAX_H3_AUDIO_CHANNELS, video_pipe.MINIMAX_H3_AUDIO_TAG,
    video_pipe.MINIMAX_H3_VIDEO_TAG)
ipos, itags, i_idx, it_idx, n_ref = build_image_packed_sequence(tags, [], lat_h, lat_w, (1, 2, 2), image_pipe.MINIMAX_H3_IMAGE_TAG)
assert torch.equal(pos, ipos) and torch.equal(ttags, itags) and torch.equal(v_idx, i_idx) and n_ref == 0
res = {"text": text[0].numpy(), "tags": tags.numpy(), "latents": target.numpy(), "lat_hw": np.array([lat_h, lat_w]),
       "shift": np.float32(SHIFT)}
worst = 0.0
for n in (2, 3, 4):
    sched = MiniMaxH3Scheduler(shift=SHIFT, shift_provided_sigmas=False)
    sched.set_timesteps(n + 1)
    sigmas, timesteps = sched.sigmas, sched.timesteps
    assert len(timesteps) == n
    res[f"sigmas{n}"] = sigmas.numpy()
    for s in range(n):
        uniq, tix = torch.unique(torch.full((pos.shape[0],), float(timesteps[s])), sorted=True, return_inverse=True)
        with torch.no_grad():
            v = model(hidden_states=target[None], audio_hidden_states=torch.zeros(1, 0, 32), encoder_hidden_states=text,
                      timestep=uniq, timestep_indices=tix, token_tags=ttags, position_ids=pos.float(),
                      video_indices=v_idx, audio_indices=a_idx, text_indices=t_idx, return_dict=False)[0]
            vi = image(hidden_states=target[None], encoder_hidden_states=text, timestep=uniq, timestep_indices=tix,
                       token_tags=itags, position_ids=ipos.float(), image_indices=i_idx, text_indices=it_idx,
                       pdd_step_index=s, return_dict=False)[0]
        worst = max(worst, float((v - vi).norm() / v.norm()))
        res[f"vel{n}_{s}"] = v[0].numpy()
        if s == 0:
            res[f"next{n}"] = sched.step(v[0], timesteps[0], target, return_dict=False)[0].numpy()
np.savez(os.path.join(out, "ref_flow.npz"), **res)
print(f"video transformer vs image transformer with the video head: worst rel diff {worst:.2e}")
assert worst < 1e-5
print("saved", out, {k: getattr(v, "shape", v) for k, v in res.items() if k.startswith(("sigmas", "vel2"))})
