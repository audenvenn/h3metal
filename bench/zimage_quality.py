"""h3metal's Z-Image against a diffusers reference run (tests/ref/zimage_reference.py), part by part:

    text encoder   our rows against the reference's prompt_embeds
    transformer    one step from the reference's first input, fed the reference's caption rows
    decoder        our decoder on the reference's final latents, against its image
    whole          our image from the same noise and prompt, against the reference image (PSNR, SSIM)

    python bench/zimage_quality.py MODEL_DIR REF.npz [--save side_by_side.png]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality import ssim  # noqa: E402
from h3metal.flux_vae import FluxVAEDecoder  # noqa: E402
from h3metal.pipeline import Timing  # noqa: E402
from h3metal.weights import WeightFile  # noqa: E402
from h3metal.zimage import ZImagePipeline  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


def psnr(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))


ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("ref")
ap.add_argument("--save")
ap.add_argument("--ref-embeds", action="store_true", help="feed the reference's text embeddings (isolates the "
                "transformer, its adapters and the decoder from the text encoder)")
a = ap.parse_args()
r = np.load(a.ref)
prompt, seed, size = str(r["prompt"]), int(r["seed"]), int(r["size"])
loras = [(str(r["lora"]), float(r["lora_scale"]))] if "lora" in r else []
if loras:
    print(f"LoRA {os.path.basename(loras[0][0])} at {loras[0][1]} (the reference run's)")
os.environ["H3METAL_CACHE"] = "off"
pipe = ZImagePipeline(a.model, log=lambda *x: None)

emb = pipe.encode_prompt(prompt, Timing())
print(f"text encoder: {emb.shape[0]} rows (reference {r['cap'].shape[0]}), rel err {rel(emb, r['cap']):.4f}")

ref_img = np.rint(r["image"] * 255).astype(np.uint8)
dec = FluxVAEDecoder(pipe.gpu, WeightFile(pipe.gpu, pipe.paths["vae"]))
img_dec = np.rint((dec.decode(r["final"]) / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8)
print(f"decoder on the reference latents: PSNR {psnr(img_dec, ref_img):.1f} dB")

if a.ref_embeds:
    pipe.encode_prompt = lambda p, t: r["cap"]
img, timing = pipe.generate(prompt, size, size, seed=seed, loras=loras)
dit = pipe._dit
h = w = size // 8
dit.prepare(r["cap"].shape[0], h, w)
dit.encode_caption(r["cap"])
out0 = dit.step(0, r["x0"])
print(f"transformer, first step from the reference input and caption rows: rel err {rel(out0, r['out0']):.4f}")
print(f"whole image ({timing}): PSNR {psnr(img, ref_img):.1f} dB, SSIM {ssim(img, ref_img):.4f}")
if a.save:
    from PIL import Image
    Image.fromarray(np.concatenate([ref_img, img], axis=1)).save(a.save)
    print(f"reference (left) and h3metal (right): {a.save}")
