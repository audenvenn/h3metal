"""Reference side of the mini-DiT test (runs in an environment with torch and diffusers).

Builds a small MiniMaxH3ImageTransformer2DModel with random weights, writes it in
the layout of the original MiniMax checkpoint (transformer/, pdd_fused_heads,
configs) and saves one forward pass of every step for a text-to-image layout
and for an edit layout. tests/test_dit_mini.py converts the checkpoint with
h3metal and compares.

    python tests/ref/mini_dit.py OUT_DIR
"""
import json
import os
import sys
import numpy as np
import torch
from safetensors.torch import save_file
from diffusers import MiniMaxH3ImageTransformer2DModel
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import build_image_packed_sequence, patchify_video_latents

out = sys.argv[1]
os.makedirs(os.path.join(out, "transformer"), exist_ok=True)
torch.manual_seed(0)
H, HD, D, F, TXT, TE, TEH = 4, 128, 512, 1024, 256, 256, 512
model = MiniMaxH3ImageTransformer2DModel(num_attention_heads=H, attention_head_dim=HD, hidden_size=D, num_layers=2,
    num_refiner_layers=2, ffn_dim=F, in_channels=24, patch_size=(1, 2, 2), text_dim=TXT, freq_dim=256,
    time_embed_hidden_dim=TEH, time_embed_dim=TE, rope_freq_dim=16, rope_theta=10000.0, num_pdd_heads=6).float().eval()
fp32_prefix = ("proj_in", "time_embedder", "pdd_proj_out")
with torch.no_grad():
    for name, p in model.named_parameters():
        std = 1.0 if "norm" in name else (0.5 / p.shape[-1] ** 0.5 if p.ndim == 2 else 0.05)
        v = torch.randn_like(p) * std + (1.0 if "norm" in name else 0.0)
        if "adaln_proj" in name or "norm_out.linear" in name:
            v = v * 0.5
        if not name.startswith(fp32_prefix):
            v = v.to(torch.bfloat16).float()  # the checkpoint stores these in bfloat16
        p.copy_(v)
sd = model.state_dict()

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
}
for i in range(2):
    ck.update(orig_block(f"token_refiner.refiner_blocks.{i}", f"token_refiner.blocks.{i}", False))
for l in range(2):
    ck.update(orig_block(f"transformer_blocks.{l}", f"blocks.{l}", True))
ck = {k: (v.float() if k.startswith(("video_patch_proj", "time_embedder")) else v.to(torch.bfloat16)).contiguous() for k, v in ck.items()}
save_file(ck, os.path.join(out, "transformer", "model.safetensors"))
heads = {"video_out.weight": torch.stack([sd[f"pdd_proj_out.{i}.weight"] for i in range(6)]).float().contiguous(),
         "video_out.bias": torch.stack([sd[f"pdd_proj_out.{i}.bias"] for i in range(6)]).float().contiguous()}
save_file(heads, os.path.join(out, "pdd_fused_heads.safetensors"))
json.dump({"_class_name": "MiniMaxH3DiTModel", "hidden_size": D, "num_layers": 2, "token_refiner_num_layers": 2,
           "num_attention_heads": H, "attention_head_dim": HD, "ffn_hidden_size": F, "latents_dim": 24,
           "patch_size": [1, 2, 2], "text_dim": TXT, "timestep_input_dim": 256, "time_embed_hidden_size": TEH,
           "time_embed_dim": TE, "rope_inv_freq_len": 16, "norm_eps": 1e-5, "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5},
          open(os.path.join(out, "transformer", "config.json"), "w"))
sig = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
json.dump({"num_inference_steps": 6, "sigmas": sig}, open(os.path.join(out, "pdd_config.json"), "w"))

# Forward passes. Timesteps exactly as the pipeline builds them.
t0 = torch.tensor([1.0 - s for s in sig[:-1]], dtype=torch.float32)
sigmas = torch.cat([1.0 - t0, torch.zeros(1)])
timesteps = 1.0 - sigmas[:-1]
rng = np.random.default_rng(5)
cases = {}
for case, refs in (("t2i", []), ("edit", [(8, 12)])):
    T = 37
    tags = torch.ones(T, dtype=torch.long)
    if refs:
        tags[5:15] = 0  # a vision block inside the text
    text = torch.from_numpy(rng.standard_normal((1, T, TXT)).astype(np.float32)).to(torch.bfloat16).float()
    lat_h, lat_w = 16, 20
    pos, token_tags, img_idx, txt_idx, n_ref = build_image_packed_sequence(tags, refs, lat_h, lat_w, (1, 2, 2), 0)
    rows = [patchify_video_latents(torch.from_numpy(rng.standard_normal((1, 24, 1, h, w)).astype(np.float32)), (1, 2, 2)) for h, w in refs]
    target = patchify_video_latents(torch.from_numpy(rng.standard_normal((1, 24, 1, lat_h, lat_w)).astype(np.float32)), (1, 2, 2))
    lat = torch.cat(rows + [target]) if rows else target
    res = {"text": text[0].numpy(), "tags": tags.numpy(), "latents": lat.numpy(), "refs": np.array(refs).reshape(-1, 2),
           "lat_hw": np.array([lat_h, lat_w])}
    for s in range(6):
        row_t = torch.full((pos.shape[0],), float(timesteps[s]), dtype=torch.float32)
        row_t[img_idx[:n_ref]] = 0.999
        uniq, tidx = torch.unique(row_t, sorted=True, return_inverse=True)
        with torch.no_grad():
            v = model(hidden_states=lat[None], encoder_hidden_states=text, timestep=uniq, timestep_indices=tidx,
                      token_tags=token_tags, position_ids=pos.float(), image_indices=img_idx, text_indices=txt_idx,
                      pdd_step_index=s, return_dict=False)[0]
        res[f"vel{s}"] = v[0].numpy()
    np.savez(os.path.join(out, f"ref_{case}.npz"), **res)
    print(case, "saved", {k: v.shape for k, v in res.items()})
