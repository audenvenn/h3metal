"""Reference side of the mini Z-Image test (runs in an environment with torch and diffusers).

Builds a small ZImageTransformer2DModel with random weights, saves it as a diffusers checkpoint
(OUT_DIR/transformer) and records, for an eight-step schedule from fixed latents and caption features, the
transformer's output at every step on the reference trajectory, plus the schedule itself.

    python tests/ref/mini_zimage.py OUT_DIR
"""
import json
import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, ZImageTransformer2DModel

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
# dim and the MLP's int(dim / 3 * 8) must be multiples of 256 (h3metal rotates blocks of 256 channels).
cfg = dict(all_patch_size=(2,), all_f_patch_size=(1,), in_channels=16, dim=768, n_layers=2, n_refiner_layers=2,
           n_heads=6, n_kv_heads=6, norm_eps=1e-5, qk_norm=True, cap_feat_dim=256, rope_theta=256.0, t_scale=1000.0,
           axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512])
model = ZImageTransformer2DModel(**cfg).float().eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name and p.ndim == 1:
            v = 1.0 + 0.1 * torch.randn_like(p)
        elif "pad_token" in name:
            v = torch.randn_like(p)
        elif p.ndim == 2:
            v = torch.randn_like(p) * (0.5 / p.shape[1] ** 0.5)
        else:
            v = torch.randn_like(p) * 0.05
        p.copy_(v.to(torch.bfloat16).float())  # the released checkpoint is float32, but keep values bf16-exact
model.save_pretrained(os.path.join(out, "transformer"), safe_serialization=True)

# Inputs: an 18 x 22 latent (9 x 11 = 99 patches, padded to 128) and 37 caption rows (padded to 64).
lat_h, lat_w, n_cap = 18, 22, 37
latents = torch.randn(1, 16, lat_h, lat_w)
cap = torch.randn(n_cap, cfg["cap_feat_dim"])

sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False)
steps = 8
sigmas = torch.linspace(1.0, 1 / steps, steps).tolist()
sched.set_timesteps(sigmas=sigmas)
rec = {"latents": latents[0].numpy(), "cap": cap.numpy(), "timesteps": sched.timesteps.numpy(),
       "sigmas": sched.sigmas.numpy()}
x = latents.clone()
with torch.no_grad():
    for i, t in enumerate(sched.timesteps):
        tt = (1000 - t.expand(1)) / 1000
        o = model([x.unsqueeze(2)[0]], tt, [cap], return_dict=False)[0][0]  # (C, 1, H, W)
        rec[f"x{i}"] = x[0].numpy()
        rec[f"out{i}"] = o[:, 0].numpy()
        x = sched.step(-o[:, 0].unsqueeze(0).float(), t, x, return_dict=False)[0]
rec["final"] = x[0].numpy()
np.savez(os.path.join(out, "ref.npz"), **rec)
json.dump(cfg, open(os.path.join(out, "mini_config.json"), "w"), indent=1)
print("saved", out, "timesteps", [round(float(t), 2) for t in sched.timesteps])
