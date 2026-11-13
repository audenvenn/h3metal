"""Reference run of the released Z-Image-Turbo with diffusers (torch, on the Mac's GPU in bfloat16, as the
model card runs it), recording what h3metal is compared with part by part:

    cap             the text encoder's rows (hidden_states[-2], valid tokens): the prompt embeddings
    x0, t0, out0    the transformer's first-step input, timestep and output
    final           the latents after the last step
    image           the decoded image, float in [0, 1]

The text encoder (8 GB) and the transformer (12 GB) do not fit a 24 GB Mac's GPU together, so the prompt is
encoded first with the text encoder alone, which is then freed; the pipeline takes the embeddings.

    python tests/ref/zimage_reference.py Z_IMAGE_DIR OUT.npz [--size 1024] [--seed 42] [--prompt ...]
"""
import argparse
import gc
import os

import numpy as np
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel
from transformers import AutoTokenizer, Qwen3Model

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--prompt", default="A red fox sitting in fresh snow at sunrise, photorealistic, soft golden light.")
ap.add_argument("--device", default="mps")
ap.add_argument("--lora", help="a LoRA file, loaded with diffusers' load_lora_weights")
ap.add_argument("--lora-scale", type=float, default=1.0)
a = ap.parse_args()
dt = torch.bfloat16


def empty_cache():
    gc.collect()
    if a.device == "mps":
        torch.mps.empty_cache()


# 1. The prompt, with the text encoder alone.
tok = AutoTokenizer.from_pretrained(os.path.join(a.src, "tokenizer"))
te = Qwen3Model.from_pretrained(os.path.join(a.src, "text_encoder"), torch_dtype=dt).to(a.device).eval()
enc = ZImagePipeline(scheduler=None, vae=None, text_encoder=te, tokenizer=tok, transformer=None)
with torch.no_grad():
    embeds, _ = enc.encode_prompt(a.prompt, device=a.device, do_classifier_free_guidance=False)
embeds = [e.detach().clone() for e in embeds]
del enc, te
empty_cache()

# 2. The transformer and the decoder, fed the embeddings.
tr = ZImageTransformer2DModel.from_pretrained(os.path.join(a.src, "transformer"), torch_dtype=dt).to(a.device).eval()
vae = AutoencoderKL.from_pretrained(os.path.join(a.src, "vae"), torch_dtype=dt).to(a.device).eval()
sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(a.src, "scheduler"))
pipe = ZImagePipeline(scheduler=sched, vae=vae, text_encoder=None, tokenizer=None, transformer=tr)
rec = {"cap": embeds[0].float().cpu().numpy()}
if a.lora:
    pipe.load_lora_weights(a.lora, adapter_name="lora")
    pipe.set_adapters(["lora"], adapter_weights=[a.lora_scale])
    rec.update(lora=os.path.abspath(a.lora), lora_scale=a.lora_scale)
orig_forward = pipe.transformer.forward


def spy(x, t, cap_feats, *args, **kw):
    out = orig_forward(x, t, cap_feats, *args, **kw)
    if "out0" not in rec:
        rec["x0"] = x[0][:, 0].float().cpu().numpy()
        rec["t0"] = t.float().cpu().numpy()
        rec["out0"] = out[0][0][:, 0].float().cpu().numpy()
    return out


pipe.transformer.forward = spy
gen = torch.Generator("cpu").manual_seed(a.seed)
with torch.no_grad():
    res = pipe(prompt_embeds=embeds, height=a.size, width=a.size, num_inference_steps=8, guidance_scale=0.0,
               generator=gen, output_type="latent")
    latents = res.images if hasattr(res, "images") else res[0]
    rec["final"] = latents[0].float().cpu().numpy()
    z = latents.to(vae.dtype) / vae.config.scaling_factor + vae.config.shift_factor
    img = vae.decode(z, return_dict=False)[0]
    rec["image"] = (img[0].float().cpu().permute(1, 2, 0).numpy() / 2 + 0.5).clip(0, 1)
np.savez(a.out, prompt=a.prompt, seed=a.seed, size=a.size, **rec)
print("saved", a.out, {k: getattr(v, "shape", v) for k, v in rec.items()})
