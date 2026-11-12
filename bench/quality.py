"""Image quality against a reference run of the official pipeline (tests/ref/reference.py).

Same prompt, same starting noise; the reference saved its text embeddings,
the final latents and the decoded image. This runs h3metal from the same
point and reports how far it lands:

    latent rel. error   ||ours - ref|| / ||ref|| over the final latents
    PSNR, SSIM          on the 8-bit images

--ref-emb feeds the reference's text embeddings to our transformer, which
isolates the transformer and VAE from the text encoder.

    python bench/quality.py --ref REF.npz --model DIR [--ref-emb] [--save side_by_side.png]
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from h3metal.pipeline import Pipeline, Timing, to_image  # noqa: E402


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM over the luma of two uint8 RGB images (Gaussian window 11, sigma 1.5)."""
    def luma(x):
        x = x.astype(np.float64)
        return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]

    def blur(x):
        k = np.exp(-0.5 * (np.arange(11) - 5) ** 2 / 1.5 ** 2)
        k /= k.sum()
        x = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 0, x)
        return np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, x)

    x, y = luma(a), luma(b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(m.mean())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref-emb", action="store_true", help="use the reference's text embeddings")
    ap.add_argument("--save", help="write reference | ours side by side")
    a = ap.parse_args()
    r = np.load(a.ref)
    pipe = Pipeline(a.model, log=lambda *m: None)
    timing = Timing()
    t = time.time()
    if a.ref_emb:
        emb = r["emb"].astype(np.float32)
    else:
        from h3metal.te import TextEncoder
        from h3metal.weights import WeightFile
        wf = WeightFile(pipe.gpu, pipe.paths["te"])
        emb = TextEncoder(pipe.gpu, wf).encode(r["ids"])
        wf.close()
        e = np.linalg.norm(emb - r["emb"]) / np.linalg.norm(r["emb"])
        print(f"text embeddings: rel error {e:.4f}")
    ch, cw = (int(v) for v in r["canvas"])
    noise = r["noise"].reshape(-1, ch // 16, cw // 16)
    lat = pipe.denoise(emb, ch // 16, cw // 16, 0, timing, latents=noise)
    img = to_image(pipe.decode(lat, timing), cw, ch)
    ref_lat, ref_img = r["latents"], r["image"]
    le = np.linalg.norm(lat - ref_lat) / np.linalg.norm(ref_lat)
    print(f"{os.path.basename(a.ref)}: latent rel error {le:.4f}, PSNR {psnr(img, ref_img):.2f} dB, "
          f"SSIM {ssim(img, ref_img):.4f}  ({timing}; {time.time() - t:.1f}s)")
    if a.save:
        from PIL import Image
        Image.fromarray(np.concatenate([ref_img, img], axis=1)).save(a.save)


if __name__ == "__main__":
    main()
