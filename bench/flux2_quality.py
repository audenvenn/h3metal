"""h3metal's FLUX.2 [klein] against a diffusers reference run (tests/ref/flux2_reference.py), part by part:

    prompt         our chat template and padded tokens against the reference tokenizer's
    text encoder   our rows against the reference's prompt embeddings (prompt rows and padding rows)
    transformer    one step from the reference's first input, fed the reference's embeddings
    decoder        our decoder on the reference's final latents, against its image
    whole          our image from the same noise and prompt, against the reference image (PSNR, SSIM)
    edit           with --image (the reference run's): the prepared reference pixels and its encoded rows

    python bench/flux2_quality.py MODEL_DIR REF.npz [--image REF_IMAGE] [--save side_by_side.png]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality import ssim  # noqa: E402
from h3metal.flux2 import Flux2Pipeline, chat_prompt, image_positions, prepare_reference, schedule  # noqa: E402
from h3metal.flux_vae import FluxVAEDecoder, FluxVAEEncoder  # noqa: E402
from h3metal.pipeline import Timing  # noqa: E402
from h3metal.weights import WeightFile  # noqa: E402


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
ap.add_argument("--image")
a = ap.parse_args()
r = np.load(a.ref)
prompt, seed, size = str(r["prompt"]), int(r["seed"]), int(r["size"])
loras = [(str(r["lora"]), float(r["lora_scale"]))] if "lora" in r else []
if loras:
    print(f"LoRA {os.path.basename(loras[0][0])} at {loras[0][1]} (the reference run's)")
os.environ["H3METAL_CACHE"] = "off"
pipe = Flux2Pipeline(a.model, log=lambda *x: None)

ids, n = pipe.tokenize(prompt)
same_text = chat_prompt(prompt) == str(r["text"])
same_ids = n == int(r["n_valid"]) and np.array_equal(ids, r["ids"])
print(f"prompt: template {'identical' if same_text else 'DIFFERS'}, tokens {'identical' if same_ids else 'DIFFER'} "
      f"({n} prompt tokens of {len(ids)})")
emb = pipe.encode_prompt(prompt, Timing())
ref_emb = r["embeds"]
print(f"text encoder: rel err {rel(emb, ref_emb):.4f} (prompt rows {rel(emb[:n], ref_emb[:n]):.4f}, "
      f"padding rows {rel(emb[n:], ref_emb[n:]):.4f})")

ref_img = np.rint(r["image"] * 255).astype(np.uint8)
hp = wp = size // 16
dec = FluxVAEDecoder(pipe.gpu, WeightFile(pipe.gpu, pipe.paths["vae"]))
img_dec = np.rint((dec.decode(r["final"].T.reshape(-1, hp, wp)) / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8)
print(f"decoder on the reference latents: PSNR {psnr(img_dec, ref_img):.1f} dB")

images = None
if a.image:
    from PIL import Image
    images = [Image.open(a.image)]
    px = prepare_reference(images[0])
    print(f"reference image prepared: {px.shape}, max abs diff {np.abs(px - r['ref_pixels']).max():.2e}")
    enc = FluxVAEEncoder(pipe.gpu, WeightFile(pipe.gpu, pipe.paths["vae"])).encode_packed(px)
    rows = enc.reshape(enc.shape[0], -1).T
    print(f"reference image encoded: rel err {rel(rows, r['ref_rows']):.4f}")
if a.ref_embeds:
    pipe.encode_prompt = lambda p, t: ref_emb
img, timing = pipe.generate(prompt, size, size, seed=seed, latents=r["noise"], images=images, loras=loras)
dit = pipe._dit
if images is None:
    dit.prepare(ref_emb, image_positions(hp, wp))
else:
    rh, rw = px.shape[0] // 16, px.shape[1] // 16
    dit.prepare(ref_emb, np.concatenate([image_positions(hp, wp), image_positions(rh, rw, t=10)]))
out0 = dit.step(float(schedule(4, hp * wp)[0]), r["x0"])[:hp * wp]
print(f"transformer, first step from the reference input and embeddings: rel err "
      f"{rel(out0, r['out0'][:hp * wp]):.4f}")
print(f"whole image ({timing}): PSNR {psnr(img, ref_img):.1f} dB, SSIM {ssim(img, ref_img):.4f}")
if a.save:
    from PIL import Image
    Image.fromarray(np.concatenate([ref_img, img], axis=1)).save(a.save)
    print(f"reference (left) and h3metal (right): {a.save}")
