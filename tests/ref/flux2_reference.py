"""Reference run of the released FLUX.2 [klein] 4B with diffusers (torch, on the Mac's GPU in bfloat16, as the
model card runs it), recording what h3metal is compared with part by part:

    text, ids, n_valid  the chat-templated prompt, its tokens padded to 512, the number of prompt tokens
    embeds              the text encoder's rows (hidden states 9, 18 and 27 stacked): the prompt embeddings
    noise               the starting latents [128][h/16][w/16] (bfloat16 values, as the pipeline casts them)
    x0, out0            the transformer's first-step input rows and output
    final               the packed latents after the last step
    image               the decoded image, float in [0, 1]
    ref_pixels, ref_rows   with --image: the reference image as the pipeline prepares it, and its packed,
                           normalized latents (the rows the transformer reads)

The text encoder runs alone first and is freed; the pipeline takes the embeddings.

    python tests/ref/flux2_reference.py FLUX2_DIR OUT.npz [--size 1024] [--seed 42] [--prompt ...]
"""
import argparse
import gc
import os

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
from transformers import AutoTokenizer, Qwen3ForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--prompt", default="A red fox sitting in fresh snow at sunrise, photorealistic, soft golden light.")
ap.add_argument("--device", default="mps")
ap.add_argument("--lora", help="a LoRA file, loaded with diffusers' load_lora_weights")
ap.add_argument("--lora-scale", type=float, default=1.0)
ap.add_argument("--image", help="a reference image: an edit")
a = ap.parse_args()
dt = torch.bfloat16


def empty_cache():
    gc.collect()
    if a.device == "mps":
        torch.mps.empty_cache()


# 1. The prompt, with the text encoder alone.
tok = AutoTokenizer.from_pretrained(os.path.join(a.src, "tokenizer"))
text = tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False, add_generation_prompt=True,
                               enable_thinking=False)
enc = tok(text, padding="max_length", truncation=True, max_length=512)
te = Qwen3ForCausalLM.from_pretrained(os.path.join(a.src, "text_encoder"), torch_dtype=dt).to(a.device).eval()
with torch.no_grad():
    embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(te, tok, a.prompt, dtype=dt, device=a.device)
rec = {"text": text, "ids": np.array(enc["input_ids"]), "n_valid": int(sum(enc["attention_mask"])),
       "embeds": embeds[0].float().cpu().numpy()}
del te
empty_cache()

# 2. The transformer and the decoder, fed the embeddings.
tr = Flux2Transformer2DModel.from_pretrained(os.path.join(a.src, "transformer"), torch_dtype=dt).to(a.device).eval()
vae = AutoencoderKLFlux2.from_pretrained(os.path.join(a.src, "vae"), torch_dtype=dt).to(a.device).eval()
sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(a.src, "scheduler"))
pipe = Flux2KleinPipeline(scheduler=sched, vae=vae, text_encoder=None, tokenizer=None, transformer=tr,
                          is_distilled=True)
if a.lora:
    pipe.load_lora_weights(a.lora, adapter_name="lora")
    pipe.set_adapters(["lora"], adapter_weights=[a.lora_scale])
    rec.update(lora=os.path.abspath(a.lora), lora_scale=a.lora_scale)
orig_forward = pipe.transformer.forward


def spy(*args, **kw):
    out = orig_forward(*args, **kw)
    if "out0" not in rec:
        rec["x0"] = kw["hidden_states"][0].float().cpu().numpy()
        rec["t0"] = kw["timestep"].float().cpu().numpy()
        rec["out0"] = out[0][0].float().cpu().numpy()
    return out


def on_step(p, i, t, kw):
    rec["final"] = kw["latents"][0].float().cpu().numpy()
    return {}


pipe.transformer.forward = spy
images = None
if a.image:
    from PIL import Image
    im = Image.open(a.image).convert("RGB")
    im = pipe.image_processor._resize_if_exceeds_area(im, 1024 * 1024)
    w16, h16 = im.size[0] // 16 * 16, im.size[1] // 16 * 16
    px = pipe.image_processor.preprocess(im, height=h16, width=w16, resize_mode="crop")
    rec["ref_pixels"] = px[0].permute(1, 2, 0).float().numpy()
    with torch.no_grad():
        rows, ids = pipe.prepare_image_latents([px], 1, None, a.device, dt)
    rec["ref_rows"] = rows[0].float().cpu().numpy()
    images = [Image.open(a.image).convert("RGB")]
hp = wp = a.size // 16
noise = torch.randn((1, 128, hp, wp), generator=torch.Generator("cpu").manual_seed(a.seed)).to(dt)
rec["noise"] = noise[0].float().numpy()
with torch.no_grad():
    res = pipe(prompt_embeds=embeds, height=a.size, width=a.size, num_inference_steps=4, guidance_scale=1.0,
               latents=noise.to(a.device), output_type="pt", callback_on_step_end=on_step, image=images,
               callback_on_step_end_tensor_inputs=["latents"])
img = res.images if hasattr(res, "images") else res[0]
rec["image"] = img[0].float().cpu().permute(1, 2, 0).numpy().clip(0, 1)
np.savez(a.out, prompt=a.prompt, seed=a.seed, size=a.size, **rec)
print("saved", a.out, {k: getattr(v, "shape", v) for k, v in rec.items() if k != "text"})
