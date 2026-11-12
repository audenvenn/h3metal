"""Reference run of the released Z-Image-Turbo with diffusers (torch, on the Mac's GPU in bfloat16, as the
model card runs it), recording what h3metal is compared with part by part:

    prompt_embeds   the text encoder's rows (hidden_states[-2], valid tokens)
    x0, t0, out0    the transformer's first-step input, timestep and output
    final           the latents after the last step
    image           the decoded image, float in [0, 1]

    python tests/ref/zimage_reference.py Z_IMAGE_DIR OUT.npz [--size 1024] [--seed 42] [--prompt ...]
"""
import argparse

import numpy as np
import torch
from diffusers import ZImagePipeline

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--prompt", default="A red fox sitting in fresh snow at sunrise, photorealistic, soft golden light.")
ap.add_argument("--device", default="mps")
a = ap.parse_args()

pipe = ZImagePipeline.from_pretrained(a.src, torch_dtype=torch.bfloat16).to(a.device)
rec = {}
orig_forward = pipe.transformer.forward


def spy(x, t, cap_feats, *args, **kw):
    out = orig_forward(x, t, cap_feats, *args, **kw)
    if "out0" not in rec:
        rec["x0"] = x[0][:, 0].float().cpu().numpy()
        rec["t0"] = t.float().cpu().numpy()
        rec["cap"] = cap_feats[0].float().cpu().numpy()
        rec["out0"] = out[0][0][:, 0].float().cpu().numpy()
    return out


pipe.transformer.forward = spy
gen = torch.Generator("cpu").manual_seed(a.seed)
with torch.no_grad():
    res = pipe(prompt=a.prompt, height=a.size, width=a.size, num_inference_steps=8, guidance_scale=0.0,
               generator=gen, output_type="latent")
    latents = res.images if hasattr(res, "images") else res[0]
    rec["final"] = latents[0].float().cpu().numpy()
    z = latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    img = pipe.vae.decode(z, return_dict=False)[0]
    rec["image"] = (img[0].float().cpu().permute(1, 2, 0).numpy() / 2 + 0.5).clip(0, 1)
np.savez(a.out, prompt=a.prompt, seed=a.seed, size=a.size, **rec)
print("saved", a.out, {k: v.shape for k, v in rec.items()})
